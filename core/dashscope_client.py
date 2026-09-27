"""Minimal asynchronous client for the DashScope non-realtime Qwen-TTS API.

I. Scope

1. Only the non-streaming route is implemented on purpose. AstrBot sends one
   voice message per reply, so incremental PCM playback would not reach the user
   any sooner; a single request that returns a complete audio file is both
   simpler and more reliable.
2. Non-streaming responses carry either a download URL (valid for 24 hours) or
   inline base64 audio. Both are handled, URL first.

II. Reliability

1. Transport failures, timeouts and the retryable status codes (408/425/429/5xx)
   are retried with exponential backoff and jitter.
2. Business errors (``code`` set in the payload) are surfaced immediately as
   :class:`~core.errors.TtsApiError` because retrying them cannot help.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import random
from dataclasses import dataclass, field
from typing import Any

from .config import DASHSCOPE_SYNTHESIS_ROUTE
from .errors import TtsApiError, TtsConfigError, TtsNetworkError

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Errors returned by DashScope that indicate a bad API key or no access.
AUTH_ERROR_CODES = frozenset(
    {
        "InvalidApiKey",
        "InvalidApiKeyFormat",
        "AccessDenied",
        "Unauthorized",
        "Model.AccessDenied",
    },
)


@dataclass(slots=True)
class SynthesisResult:
    """Audio bytes produced by one synthesis request."""

    audio: bytes
    content_type: str = "audio/wav"
    request_id: str = ""
    source: str = "url"
    usage: dict[str, Any] = field(default_factory=dict)


class DashScopeTtsClient:
    """Thin async wrapper around the Qwen-TTS HTTP endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        backoff: float = 1.5,
        session: Any | None = None,
        sleep: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff = max(0.0, backoff)
        self._session = session
        self._owns_session = session is None
        self._sleep = sleep or asyncio.sleep
        self._logger = logger
        self._session_lock = asyncio.Lock()

    # ----------------------------------------------------------- plumbing ---

    @property
    def synthesis_url(self) -> str:
        """Absolute URL of the synthesis endpoint."""
        return f"{self.base_url}{DASHSCOPE_SYNTHESIS_ROUTE}"

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self._logger is None:
            return
        log_method = getattr(self._logger, level, None)
        if callable(log_method):
            log_method(message, *args)

    async def _ensure_session(self) -> Any:
        """Lazily create the shared ``aiohttp`` session inside the event loop."""
        if self._session is not None and not getattr(self._session, "closed", False):
            return self._session
        async with self._session_lock:
            if self._session is not None and not getattr(self._session, "closed", False):
                return self._session
            try:
                import aiohttp
            except ImportError as exc:  # pragma: no cover - aiohttp ships with AstrBot
                raise TtsConfigError(
                    "aiohttp is not installed",
                    hint=(
                        "缺少 aiohttp 依赖，请在插件目录执行 "
                        "pip install -r requirements.txt 后重载插件。"
                    ),
                ) from exc

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
            return self._session

    async def aclose(self) -> None:
        """Close the owned session; an injected session is left untouched."""
        session, self._session = self._session, None
        if session is not None and self._owns_session:
            close = getattr(session, "close", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff with full jitter, honouring ``Retry-After``."""
        if retry_after is not None and retry_after > 0:
            return min(retry_after, 30.0)
        base = self.backoff * (2**attempt)
        return min(base, 30.0) * (0.5 + random.random() / 2)

    # ------------------------------------------------------------ request ---

    async def synthesize(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        language_type: str = "Auto",
        instructions: str = "",
        optimize_instructions: bool = False,
    ) -> SynthesisResult:
        """Synthesize ``text`` and return the complete audio payload."""
        if not self.api_key:
            raise TtsConfigError(
                "missing api_key",
                hint="未配置 DashScope API Key，无法调用语音合成。",
            )
        if not text.strip():
            raise TtsConfigError("empty text", hint="待合成的文本为空。")

        payload = self.build_payload(
            text=text,
            model=model,
            voice=voice,
            language_type=language_type,
            instructions=instructions,
            optimize_instructions=optimize_instructions,
        )
        data = await self._post_json(self.synthesis_url, payload)
        return await self._extract_audio(data)

    @staticmethod
    def build_payload(
        *,
        text: str,
        model: str,
        voice: str,
        language_type: str = "Auto",
        instructions: str = "",
        optimize_instructions: bool = False,
    ) -> dict[str, Any]:
        """Build the JSON body documented for Qwen-TTS.

        ``instructions``/``optimize_instructions`` are only meaningful for the
        ``*-instruct-*`` model family, so they are omitted when unset rather than
        sent as empty values (some deployments reject unknown empty fields).
        """
        input_payload: dict[str, Any] = {
            "text": text,
            "voice": voice,
            "language_type": language_type or "Auto",
        }
        if instructions:
            input_payload["instructions"] = instructions
            if optimize_instructions:
                input_payload["optimize_instructions"] = True
        return {"model": model, "input": input_payload}

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = await self._ensure_session()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._auth_headers(),
                ) as response:
                    body = await response.text()
                    status = response.status
                    if status in RETRYABLE_STATUS:
                        retry_after = _parse_retry_after(
                            response.headers.get("Retry-After"),
                        )
                        last_error = TtsApiError(
                            f"HTTP {status}",
                            hint=f"语音服务暂时不可用（HTTP {status}），已重试 {attempt} 次。",
                            status_code=status,
                        )
                        if attempt < self.max_retries:
                            delay = self._backoff_delay(attempt, retry_after)
                            self._log(
                                "warning",
                                "DashScope TTS HTTP %s, retrying in %.1fs (attempt %s/%s)",
                                status,
                                delay,
                                attempt + 1,
                                self.max_retries,
                            )
                            await self._sleep(delay)
                            continue
                        raise last_error

                    parsed = _loads_or_raise(body, status)
                    if status != 200:
                        raise _api_error_from_payload(parsed, status, body)
                    return parsed
            except TtsApiError as exc:
                if exc.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    last_error = exc
                    await self._sleep(self._backoff_delay(attempt))
                    continue
                raise
            except TimeoutError as exc:
                last_error = TtsNetworkError(
                    f"timeout after {self.timeout}s",
                    hint=f"语音合成超时（{self.timeout:.0f} 秒），请稍后重试或缩短文本。",
                )
                if attempt < self.max_retries:
                    await self._sleep(self._backoff_delay(attempt))
                    continue
                raise last_error from exc
            except Exception as exc:  # aiohttp.ClientError and friends
                if not _is_transport_error(exc):
                    raise
                last_error = TtsNetworkError(
                    f"transport error: {exc!r}",
                    hint="无法连接语音合成服务，请检查服务器网络或 DashScope 地址配置。",
                )
                if attempt < self.max_retries:
                    await self._sleep(self._backoff_delay(attempt))
                    continue
                raise last_error from exc

        raise last_error or TtsNetworkError("request failed")  # pragma: no cover

    # ------------------------------------------------------------- result ---

    async def _extract_audio(self, data: dict[str, Any]) -> SynthesisResult:
        if _as_int(data.get("status_code"), 200) not in (200, 0):
            raise _api_error_from_payload(data, _as_int(data.get("status_code"), 0), "")

        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        audio = output.get("audio") if isinstance(output.get("audio"), dict) else {}
        url = str(audio.get("url") or "").strip()
        inline = str(audio.get("data") or "").strip()
        request_id = str(data.get("request_id") or "").strip()
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}

        if url:
            try:
                content, content_type = await self._download(url)
            except TtsNetworkError:
                # Fall back to the inline copy when the CDN is unreachable.
                if not inline:
                    raise
                self._log("warning", "Audio download failed, using inline base64 payload")
            else:
                return SynthesisResult(
                    audio=content,
                    content_type=content_type or "audio/wav",
                    request_id=request_id,
                    source="url",
                    usage=usage,
                )

        if inline:
            return SynthesisResult(
                audio=_decode_base64(inline),
                content_type="audio/wav",
                request_id=request_id,
                source="base64",
                usage=usage,
            )

        raise TtsApiError(
            "no audio in response",
            hint="语音服务返回了空音频，请检查文本内容或模型名称是否正确。",
            code=str(data.get("code") or ""),
            request_id=request_id,
        )

    async def _download(self, url: str) -> tuple[bytes, str]:
        session = await self._ensure_session()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with session.get(url) as response:
                    if response.status in RETRYABLE_STATUS or response.status >= 400:
                        last_error = TtsNetworkError(
                            f"download HTTP {response.status}",
                            hint=f"下载合成音频失败（HTTP {response.status}）。",
                        )
                        if attempt < self.max_retries:
                            await self._sleep(self._backoff_delay(attempt))
                            continue
                        raise last_error
                    content = await response.read()
                    content_type = response.headers.get("Content-Type", "")
                    return bytes(content), str(content_type or "")
            except TtsNetworkError:
                raise
            except TimeoutError as exc:
                last_error = TtsNetworkError(
                    "download timeout",
                    hint="下载合成音频超时。",
                )
                if attempt < self.max_retries:
                    await self._sleep(self._backoff_delay(attempt))
                    continue
                raise last_error from exc
            except Exception as exc:
                if not _is_transport_error(exc):
                    raise
                last_error = TtsNetworkError(
                    f"download transport error: {exc!r}",
                    hint="下载合成音频失败，请检查网络。",
                )
                if attempt < self.max_retries:
                    await self._sleep(self._backoff_delay(attempt))
                    continue
                raise last_error from exc
        raise last_error or TtsNetworkError("download failed")  # pragma: no cover


# ------------------------------------------------------------- helpers ---


def _is_transport_error(exc: Exception) -> bool:
    """Return True for network-ish exceptions without importing aiohttp."""
    if isinstance(exc, asyncio.TimeoutError | TimeoutError | OSError):
        return True
    module = type(exc).__module__ or ""
    return module.startswith("aiohttp") and "Error" in type(exc).__name__


def _loads_or_raise(body: str, status: int) -> dict[str, Any]:
    if not body:
        if status == 200:
            raise TtsApiError(
                "empty response body",
                hint="语音服务返回了空响应。",
                status_code=status,
            )
        raise TtsApiError(
            f"HTTP {status}",
            hint=f"语音服务返回 HTTP {status}，且响应体为空。",
            status_code=status,
        )
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TtsApiError(
            f"invalid JSON response (HTTP {status})",
            hint=f"语音服务返回了无法解析的响应（HTTP {status}）。",
            status_code=status,
        ) from exc
    if not isinstance(parsed, dict):
        raise TtsApiError(
            "unexpected response shape",
            hint="语音服务返回了非预期的响应结构。",
            status_code=status,
        )
    return parsed


def _api_error_from_payload(
    payload: dict[str, Any],
    status: int,
    body: str = "",
) -> TtsApiError:
    code = str(payload.get("code") or "").strip()
    message = str(payload.get("message") or "").strip()
    request_id = str(payload.get("request_id") or "").strip()
    detail = message or (body[:200] if body else "")
    lowered = detail.lower()

    if code in AUTH_ERROR_CODES or status in (401, 403):
        hint = (
            "DashScope 鉴权失败。请检查 API Key 是否正确，以及 Key 与所选地域是否匹配"
            "（北京/新加坡的 Key 不通用）。"
        )
    elif status == 429 or code in {"Throttling", "Throttling.RateQuota", "LimitExceeded"}:
        hint = "调用频率超限，请稍后重试或在插件配置中调低并发。"
    elif (
        "url error" in lowered
        or "model not exist" in lowered
        or "model not found" in lowered
        or "modelnotfound" in lowered
    ):
        # Verified behaviour: the Qwen-Audio-TTS family (qwen-audio-3.x-tts-*) is
        # served over the WebSocket realtime API only, so the HTTP route answers
        # with "url error" instead of a clear "unsupported model".
        hint = (
            f"模型在所选接口地址上不可用：{detail or code}。\n"
            "请确认「地域/接口地址」与 model 是否配套。HTTP 非实时接口可用模型："
            "qwen3-tts-flash、qwen3-tts-instruct-flash、qwen-tts。\n"
            "注意：qwen-audio-3.1-tts-flash 等 Qwen-Audio-TTS 系列只提供实时"
            "（WebSocket）接口，无法通过本插件当前的 HTTP 通道调用，请改用上述模型。"
        )
    elif "voice" in lowered or "Voice" in code:
        hint = f"音色（voice）不被支持：{detail}。请换成文档中的系统音色，例如 Cherry。"
    elif "text" in lowered and ("length" in lowered or "token" in lowered):
        hint = f"文本超出模型限制：{detail}。请在插件配置中调小「单段最大字符数」。"
    elif "instructions" in lowered:
        hint = (
            f"语气指令不被当前模型支持：{detail}。"
            "instructions 仅适用于 qwen3-tts-instruct-flash 系列，请清空该配置或更换模型。"
        )
    else:
        hint = f"语音合成失败：{detail or code or f'HTTP {status}'}"

    return TtsApiError(
        f"DashScope error code={code or 'n/a'} status={status} message={detail}",
        hint=hint,
        status_code=status,
        code=code,
        request_id=request_id,
    )


def _parse_retry_after(raw: Any) -> float | None:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decode_base64(payload: str) -> bytes:
    """Decode base64 audio, tolerating data-URI prefixes and padding loss."""
    data = payload.strip()
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    if data.startswith("base64://"):
        data = data[len("base64://") :]
    padding = (-len(data)) % 4
    try:
        return base64.b64decode(data + "=" * padding, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise TtsApiError(
            "invalid base64 audio",
            hint="语音服务返回的音频数据无法解码。",
        ) from exc
