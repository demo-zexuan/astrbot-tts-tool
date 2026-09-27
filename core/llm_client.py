"""OpenAI-compatible chat-completions client for the pre-synthesis LLM stage.

I. Why a second HTTP client

1. The recommended way to reach an LLM from a plugin is AstrBot's provider
   system (see :class:`main.AstrBotProviderLlmClient`), because AstrBot already
   owns the API base, the credentials and — critically — the provider's
   ``custom_headers``.
2. Some deployments (a gateway such as Opencode, a self-hosted proxy, a provider
   that is not registered in AstrBot) still need a direct call. This client
   covers that case and exposes every header knob explicitly so a gateway that
   requires non-standard headers can be reached.
3. Both classes satisfy the same tiny protocol: ``async chat(messages)``.

II. Header handling (the part that breaks in practice)

1. ``auth_style`` decides where the key goes: ``bearer`` (``Authorization:
   Bearer ...``), ``x-api-key`` (Anthropic/Opencode-style gateways), or ``none``
   when the proxy injects credentials itself.
2. ``extra_headers`` is applied last and can therefore override anything,
   including the auth header. This is the escape hatch for gateways that need
   ``HTTP-Referer``, ``X-Title``, ``anthropic-version``, an organisation id, or
   a bespoke signature header.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any

from .errors import LlmError

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

AUTH_STYLE_BEARER = "bearer"
AUTH_STYLE_X_API_KEY = "x-api-key"
AUTH_STYLE_NONE = "none"

AUTH_STYLES: tuple[str, ...] = (AUTH_STYLE_BEARER, AUTH_STYLE_X_API_KEY, AUTH_STYLE_NONE)

DEFAULT_CHAT_PATH = "/chat/completions"


@dataclass(slots=True)
class LlmChatResult:
    """One completion returned by the LLM stage."""

    text: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""


def normalize_chat_url(base_url: str) -> str:
    """Return the absolute ``/chat/completions`` URL for ``base_url``.

    Users paste three shapes in practice, and all three are accepted:

    (1) ``https://host`` or ``https://host/v1``          -> append the path
    (2) ``https://host/v1/``                             -> append the path
    (3) ``https://host/v1/chat/completions``             -> already complete
    """
    candidate = (base_url or "").strip().rstrip("/")
    if not candidate:
        raise LlmError(
            "missing LLM base url",
            hint="未配置大模型接口地址（Base URL），请在插件配置的「LLM 预处理」中填写。",
        )
    if candidate.endswith(DEFAULT_CHAT_PATH):
        return candidate
    return f"{candidate}{DEFAULT_CHAT_PATH}"


class OpenAiCompatibleLlmClient:
    """Minimal ``/chat/completions`` client with explicit header control."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = "",
        auth_style: str = AUTH_STYLE_BEARER,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 20.0,
        max_retries: int = 1,
        temperature: float | None = 0.3,
        max_tokens: int = 800,
        session: Any | None = None,
        sleep: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.chat_url = normalize_chat_url(base_url)
        self.api_key = api_key
        self.model = model
        self.auth_style = auth_style if auth_style in AUTH_STYLES else AUTH_STYLE_BEARER
        self.extra_headers = {str(k): str(v) for k, v in (extra_headers or {}).items()}
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.temperature = temperature
        self.max_tokens = max(1, max_tokens)
        self._session = session
        self._owns_session = session is None
        self._sleep = sleep or asyncio.sleep
        self._logger = logger
        self._lock = asyncio.Lock()

    # ----------------------------------------------------------- plumbing ---

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self._logger is None:
            return
        method = getattr(self._logger, level, None)
        if callable(method):
            method(message, *args)

    def build_headers(self) -> dict[str, str]:
        """Return the exact headers sent with every request.

        Ordering matters: the auth header is written first so that a user-supplied
        ``extra_headers`` entry of the same name wins.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            if self.auth_style == AUTH_STYLE_BEARER:
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif self.auth_style == AUTH_STYLE_X_API_KEY:
                headers["x-api-key"] = self.api_key
        headers.update(self.extra_headers)
        return headers

    async def _ensure_session(self) -> Any:
        if self._session is not None and not getattr(self._session, "closed", False):
            return self._session
        async with self._lock:
            if self._session is not None and not getattr(self._session, "closed", False):
                return self._session
            try:
                import aiohttp
            except ImportError as exc:  # pragma: no cover - aiohttp ships with AstrBot
                raise LlmError(
                    "aiohttp is not installed",
                    hint="缺少 aiohttp 依赖，无法调用大模型接口。",
                ) from exc
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
            self._owns_session = True
            return self._session

    async def aclose(self) -> None:
        session, self._session = self._session, None
        if session is not None and self._owns_session:
            close = getattr(session, "close", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result

    # -------------------------------------------------------------- chat ---

    def build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": False,
            "max_tokens": self.max_tokens,
        }
        if self.model:
            payload["model"] = self.model
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    async def chat(self, messages: list[dict[str, Any]]) -> LlmChatResult:
        """Send ``messages`` and return the assistant reply."""
        if not messages:
            raise LlmError("empty messages", hint="内部错误：LLM 请求消息为空。")

        session = await self._ensure_session()
        payload = self.build_payload(messages)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(
                    self.chat_url,
                    json=payload,
                    headers=self.build_headers(),
                ) as response:
                    body = await response.text()
                    status = response.status
                    if status in RETRYABLE_STATUS or status >= 500:
                        last_error = LlmError(
                            f"LLM HTTP {status}",
                            hint=f"大模型接口暂时不可用（HTTP {status}）。",
                        )
                        if attempt < self.max_retries:
                            await self._sleep(self._backoff(attempt))
                            continue
                        raise last_error
                    if status != 200:
                        raise self._api_error(status, body)
                    return self._parse(body)
            except LlmError:
                raise
            except TimeoutError as exc:
                last_error = LlmError(
                    f"LLM timeout after {self.timeout}s",
                    hint=f"调用大模型超时（{self.timeout:.0f} 秒），可调低上下文条数或放宽超时。",
                )
                if attempt < self.max_retries:
                    await self._sleep(self._backoff(attempt))
                    continue
                raise last_error from exc
            except Exception as exc:
                if not _is_transport_error(exc):
                    raise
                last_error = LlmError(
                    f"LLM transport error: {exc!r}",
                    hint="无法连接大模型接口，请检查 Base URL 与网络。",
                )
                if attempt < self.max_retries:
                    await self._sleep(self._backoff(attempt))
                    continue
                raise last_error from exc

        raise last_error or LlmError("LLM request failed")  # pragma: no cover

    # ------------------------------------------------------------ helpers ---

    def _backoff(self, attempt: int) -> float:
        return min(1.5 * (2**attempt), 15.0) * (0.5 + random.random() / 2)

    def _api_error(self, status: int, body: str) -> LlmError:
        message = ""
        code = ""
        try:
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                error = parsed.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or "")
                    code = str(error.get("code") or error.get("type") or "")
                else:
                    message = str(parsed.get("message") or "")
                    code = str(parsed.get("code") or "")
        except json.JSONDecodeError:
            message = body[:200]

        if status in (401, 403) or code in {"invalid_api_key", "authentication_error"}:
            hint = (
                "大模型接口鉴权失败。请检查 API Key，以及是否需要额外的请求头"
                "（例如网关要求的 Authorization / x-api-key / anthropic-version）。"
            )
        elif status == 404:
            hint = (
                "大模型接口返回 404。请检查 Base URL 是否正确"
                "（插件会自动补 /chat/completions），以及服务商是否需要 /v1 路径。"
            )
        elif status == 429:
            hint = "大模型接口限流，请稍后重试或减少上下文。"
        else:
            hint = f"调用大模型失败（HTTP {status}）：{message or code or '未知错误'}"

        return LlmError(
            f"LLM HTTP {status} code={code or 'n/a'} message={message}",
            hint=hint,
        )

    @staticmethod
    def _parse(body: str) -> LlmChatResult:
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise LlmError(
                "LLM response is not JSON",
                hint="大模型接口返回了无法解析的响应（可能不是 OpenAI 兼容格式）。",
            ) from exc

        if not isinstance(parsed, dict):
            raise LlmError(
                "unexpected LLM response shape",
                hint="大模型接口返回了非预期的响应结构。",
            )

        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmError(
                "LLM response has no choices",
                hint="大模型没有返回任何内容，请检查模型名是否正确。",
            )

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, list):
            # Some gateways return content parts instead of a plain string.
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        text = str(content or "").strip()

        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        return LlmChatResult(
            text=text,
            model=str(parsed.get("model") or ""),
            usage=usage,
            finish_reason=str(first.get("finish_reason") or ""),
        )


def _is_transport_error(exc: Exception) -> bool:
    """Return True for network-ish exceptions without importing aiohttp."""
    if isinstance(exc, TimeoutError | OSError):
        return True
    module = type(exc).__module__ or ""
    return module.startswith("aiohttp") and "Error" in type(exc).__name__
