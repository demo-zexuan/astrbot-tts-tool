"""AstrBot wake-word text-to-speech plugin.

I. What it does

1. Listens for messages that start with a configured wake word (for example
   ``小助手 今天天气不错``) and replies with synthesized speech.
2. Optionally runs an LLM "director" first, which rewrites the message into a
   spoken script (free-style instruction following, fine-grained tags and
   conversation-aware delivery) before any TTS request is made.
3. Exposes explicit commands for the same pipeline: ``/tts``, ``/tts_voice``,
   ``/tts_llm``, ``/tts_status`` and ``/tts_help``.
4. Uses configurable DashScope Qwen-TTS models (default ``qwen3-tts-flash``) and
   sends the resulting audio to the current session.

I-bis. How the LLM stage reaches a model

1. ``llm_mode = astrbot_provider`` (default) calls a provider configured in
   AstrBot through ``Provider.text_chat``. AstrBot owns the API base, the key and
   the provider's ``custom_headers``, so a gateway that needs extra request
   headers (Opencode, proxies, ...) works as soon as it is configured in AstrBot.
2. ``llm_mode = direct`` calls an OpenAI-compatible ``/chat/completions``
   endpoint from this plugin, with an explicit auth style plus arbitrary extra
   headers (``llm_extra_headers``) for gateways that are not registered in
   AstrBot.

II. How it is wired

1. ``core/`` holds all logic and never imports ``astrbot``, so it is unit
   testable on its own.
2. This module is the only AstrBot-facing layer: it translates
   :class:`AstrMessageEvent` objects into core calls and core outcomes into
   message chains.
3. Plugin state lives in ``data/plugin_data/<plugin_dir>/`` as required by the
   AstrBot plugin storage convention.

III. AstrBot API references

1. Plugin skeleton and commands:
   https://docs.astrbot.app/dev/star/guides/simple.html
2. Message chains, ``Record``:
   https://docs.astrbot.app/dev/star/guides/send-message.html
3. ``_conf_schema.json``: https://docs.astrbot.app/dev/star/guides/plugin-config.html
4. Plugin KV/file storage convention:
   https://docs.astrbot.app/dev/star/guides/storage.html
5. Qwen-TTS HTTP API:
   https://help.aliyun.com/zh/model-studio/qwen-tts-api
6. Reusing a configured LLM (``Provider.text_chat`` with ``system_prompt``,
   ``contexts`` and ``model``):
   https://docs.astrbot.app/dev/star/guides/ai.html
7. Provider ``custom_headers`` (how non-standard gateway headers get applied):
   ``astrbot/core/provider/sources/openai_source.py``

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Record
from astrbot.api.star import Context, Star

from .core.audio_store import AudioStore
from .core.config import DEFAULT_MODEL, PluginSettings, mask_secret
from .core.director import TtsCapabilities
from .core.errors import LlmError, TtsError
from .core.llm_client import LlmChatResult, OpenAiCompatibleLlmClient
from .core.service import SynthesisOutcome, TtsService
from .core.text_utils import extract_command_argument, match_wake_word

#: Directory name AstrBot uses for ``data/plugin_data/<name>/``. It is the
#: plugin folder name, which is exactly the parent of this file.
PLUGIN_DIR_NAME = Path(__file__).resolve().parent.name

SESSION_VOICE_FILE = "session_voices.json"
PRUNE_INTERVAL_SECONDS = 1800.0
MAX_SESSION_VOICE_ENTRIES = 500

#: Upper bound for a single history message fed to the director. Chat logs can
#: contain very long messages, and the director only needs the gist.
CONTEXT_MESSAGE_MAX_CHARS = 400


def resolve_plugin_data_dir() -> Path:
    """Return ``data/plugin_data/<plugin_dir>/`` for this plugin.

    I. Resolution order

    1. ``astrbot.core.utils.astrbot_path.get_astrbot_data_path()`` when the
       helper is importable (AstrBot >= 4.5).
    2. ``$ASTRBOT_ROOT/data`` otherwise.
    3. ``./data`` as the last resort, which matches AstrBot's own fallback.
    """
    try:  # pragma: no cover - exercised inside AstrBot only
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path

        base = Path(get_astrbot_data_path())
    except Exception:
        root = os.environ.get("ASTRBOT_ROOT") or os.getcwd()
        base = Path(root) / "data"
    return base / "plugin_data" / PLUGIN_DIR_NAME


class AstrBotProviderLlmClient:
    """Run the director stage through AstrBot's own provider system.

    I. Why this is the default

    1. AstrBot already owns the API base, the credentials, the retry policy and —
       the part that is easy to get wrong — the provider's ``custom_headers``.
       Those headers are attached to the HTTP client inside the provider source
       (``openai_source.py`` sets ``default_headers=self.custom_headers``), so a
       gateway such as Opencode is reachable as soon as it is configured in
       AstrBot. Reimplementing the request here would silently drop them.
    2. ``Provider.text_chat`` is stateless with respect to chat history: it only
       uses the ``contexts`` it is given. The director therefore cannot pollute
       the session's conversation.

    II. Signature tolerance

    Providers have slightly different ``text_chat`` signatures across versions, so
    the accepted keyword arguments are introspected instead of assumed.
    """

    def __init__(
        self,
        *,
        plugin: Any,
        provider_id: str = "",
        model: str = "",
        logger: Any | None = None,
    ) -> None:
        self._plugin = plugin
        self.provider_id = provider_id
        self.model = model
        self._logger = logger
        self._cached_provider: Any | None = None
        self._cached_provider_id: str = ""

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self._logger is None:
            return
        method = getattr(self._logger, level, None)
        if callable(method):
            method(message, *args)

    def resolve_provider(self) -> Any | None:
        """Return the configured provider, falling back to the session default."""
        context = getattr(self._plugin, "context", None)
        if context is None:
            return None

        if self.provider_id:
            if self._cached_provider is not None and self._cached_provider_id == self.provider_id:
                return self._cached_provider
            getter = getattr(context, "get_provider_by_id", None)
            if callable(getter):
                try:
                    provider = getter(self.provider_id)
                except Exception as exc:
                    self._log(
                        "warning", "Cannot resolve LLM provider %s: %s", self.provider_id, exc
                    )
                    provider = None
                if provider is not None:
                    self._cached_provider = provider
                    self._cached_provider_id = self.provider_id
                    return provider
            self._log(
                "warning",
                "Configured LLM provider %r not found; trying the session default.",
                self.provider_id,
            )

        getter = getattr(context, "get_using_provider", None)
        if callable(getter):
            for args in ((), (None,)):
                try:
                    return getter(*args)
                except TypeError:
                    continue
                except Exception as exc:
                    self._log("debug", "get_using_provider failed: %s", exc)
                    return None
        return None

    async def chat(self, messages: list[dict[str, Any]]) -> LlmChatResult:
        """Send the director prompt through the resolved provider."""
        provider = self.resolve_provider()
        if provider is None:
            raise LlmError(
                "no LLM provider available",
                hint=(
                    "没有找到可用的大模型 Provider。请在 AstrBot 的「服务提供商」里配置一个"
                    "对话模型，或在插件配置的「LLM 预处理」中指定 Provider；"
                    "也可以把「接入方式」改为直连并填写 Base URL 与 API Key。"
                ),
            )

        system_prompt = ""
        turns: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
            if role == "system" and not system_prompt:
                system_prompt = content
                continue
            turns.append({"role": role, "content": content})
        if not turns:
            raise LlmError("no user message for the LLM", hint="内部错误：缺少用户消息。")

        prompt = turns[-1]["content"]
        # The director embeds conversation history inside its prompt on purpose,
        # so there is normally nothing to forward as structured chat turns. Any
        # extra non-system message would be forwarded, which keeps this adapter
        # correct for callers that do build a multi-turn prompt.
        contexts = turns[:-1] or None

        text_chat = getattr(provider, "text_chat", None)
        if not callable(text_chat):
            raise LlmError(
                "provider does not support text_chat",
                hint="所选 Provider 不是对话模型（缺少 text_chat 方法），请改选一个对话模型。",
            )

        kwargs: dict[str, Any] = {"prompt": prompt}
        try:
            parameters = inspect.signature(text_chat).parameters
            accepted = set(parameters)
            # A provider that declares **kwargs accepts everything; AstrBot's own
            # base class exposes these explicitly, subclasses often do not.
            accepts_var_kwargs = any(
                param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
            )
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            accepted = {"prompt", "contexts", "system_prompt", "model"}
            accepts_var_kwargs = True
        if accepts_var_kwargs:
            accepted |= {"contexts", "system_prompt", "model"}
        if "contexts" in accepted and contexts:
            kwargs["contexts"] = contexts
        if "system_prompt" in accepted and system_prompt:
            kwargs["system_prompt"] = system_prompt
        if "model" in accepted and self.model:
            kwargs["model"] = self.model

        try:
            response = await text_chat(**kwargs)
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(
                f"provider text_chat failed: {exc!r}",
                hint=f"调用大模型失败：{exc}",
            ) from exc

        text = getattr(response, "completion_text", None)
        if text is None:
            text = getattr(response, "content", None)
        if text is None:
            text = str(response or "")
        usage = getattr(response, "usage", None)
        return LlmChatResult(
            text=str(text or "").strip(),
            model=self.model or str(getattr(provider, "get_model", lambda: "")() or ""),
            usage=usage if isinstance(usage, dict) else {},
        )

    async def aclose(self) -> None:
        """Nothing to release: the provider's session belongs to AstrBot."""
        return None


class WakeWordTtsPlugin(Star):
    """Entry point registered with AstrBot."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config: dict[str, Any] = config if config is not None else {}

        self._data_dir = resolve_plugin_data_dir()
        self._voice_file = self._data_dir / SESSION_VOICE_FILE
        self._session_voices: dict[str, str] = {}
        self._last_trigger: dict[str, float] = {}
        self._wake_prefixes: tuple[str, ...] | None = None
        self._prune_task: asyncio.Task[None] | None = None
        self._llm_client: Any | None = None
        self._llm_fingerprint: tuple[Any, ...] | None = None
        self._retired_clients: list[asyncio.Task[None]] = []

        settings = self.settings()
        self._service = TtsService(
            settings,
            store=AudioStore(
                self._data_dir / "audio",
                ttl_hours=settings.cache_ttl_hours,
                max_files=settings.max_cache_files,
                logger=logger,
            ),
            logger=logger,
        )

    # ------------------------------------------------------------ lifecycle ---

    async def initialize(self) -> None:
        """Prepare storage and start the periodic cache pruning task."""
        settings = self.settings()
        self._service.store.ttl_hours = settings.cache_ttl_hours
        self._service.store.max_files = settings.max_cache_files
        try:
            self._service.store.ensure_ready()
        except OSError as exc:
            logger.error("Wake-word TTS: cannot create audio cache dir: %s", exc)

        self._load_session_voices()
        self._prune_once()
        self._prune_task = asyncio.create_task(self._prune_loop())
        logger.info(
            "Wake-word TTS ready: model=%s voice=%s wake_words=%s llm=%s",
            settings.model,
            settings.voice,
            ", ".join(settings.all_wake_words) or "(none)",
            self._describe_llm(settings),
        )

    async def terminate(self) -> None:
        """Stop background work and flush persisted session voices."""
        if self._prune_task is not None:
            self._prune_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._prune_task
            self._prune_task = None

        self._save_session_voices()
        with suppress(Exception):
            await self._service.aclose()
        for task in self._retired_clients:
            task.cancel()
        self._retired_clients.clear()
        logger.info("Wake-word TTS terminated.")

    # ------------------------------------------------------------- settings ---

    def settings(self) -> PluginSettings:
        """Parse the current config and keep the service in sync.

        Re-parsing on every call means a WebUI config change takes effect on the
        next message without a manual plugin reload.
        """
        settings = PluginSettings.from_dict(self.config)
        if getattr(self, "_service", None) is not None:
            self._service.update_settings(settings)
            self._service.store.ttl_hours = settings.cache_ttl_hours
            self._service.store.max_files = settings.max_cache_files
            self._service.llm_client = self._llm_client_for(settings)
        return settings

    # ------------------------------------------------------------ LLM stage ---

    def _describe_llm(self, settings: PluginSettings) -> str:
        """One-line description of the director stage for logs and status."""
        if not settings.llm_enable:
            return "off"
        if settings.llm_uses_provider():
            provider = settings.llm_provider_id or "(当前会话默认)"
            model = settings.llm_model or "(provider 默认)"
            return f"astrbot_provider provider={provider} model={model}"
        return f"direct model={settings.llm_model or '(未设置)'} url={settings.llm_base_url}"

    def _llm_client_for(self, settings: PluginSettings) -> Any | None:
        """Build (and cache) the LLM client for the current configuration.

        Provider mode returns a thin adapter over AstrBot's own provider system,
        which is what makes a gateway's ``custom_headers`` apply. Direct mode
        returns an OpenAI-compatible HTTP client owned by this plugin.
        """
        if not settings.llm_enable:
            return None

        fingerprint = settings.llm_fingerprint()
        if self._llm_client is not None and self._llm_fingerprint == fingerprint:
            return self._llm_client

        previous = self._llm_client
        if settings.llm_uses_provider():
            self._llm_client = AstrBotProviderLlmClient(
                plugin=self,
                provider_id=settings.llm_provider_id,
                model=settings.llm_model,
                logger=logger,
            )
        else:
            try:
                settings.llm_requires_credentials()
            except TtsError as exc:
                logger.warning("Wake-word TTS: LLM stage disabled: %s", exc.message)
                self._llm_client = None
                self._llm_fingerprint = fingerprint
                return None
            self._llm_client = OpenAiCompatibleLlmClient(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                auth_style=settings.llm_auth_style,
                extra_headers=settings.llm_headers(),
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                logger=logger,
            )

        self._llm_fingerprint = fingerprint
        # Replacing the client: release the previous HTTP session if it owns one.
        if previous is not None and previous is not self._llm_client:
            # Keep a reference so the close task is not garbage collected.
            task = asyncio.ensure_future(self._close_quietly(previous))
            self._retired_clients = [t for t in self._retired_clients if not t.done()]
            self._retired_clients.append(task)
        return self._llm_client

    @staticmethod
    async def _close_quietly(client: Any) -> None:
        close = getattr(client, "aclose", None)
        if callable(close):
            with suppress(Exception):
                await close()

    async def _conversation_history(
        self,
        event: AstrMessageEvent,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Read the most recent turns of this session for context awareness.

        Everything is best-effort: a missing conversation, an old AstrBot build
        without ``conversation_manager``, or unparsable history simply yields an
        empty context rather than failing the reply.
        """
        if limit <= 0:
            return []

        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            return []

        try:
            umo = str(event.unified_msg_origin or "")
            if not umo:
                return []
            conversation_id = await manager.get_curr_conversation_id(umo)
            if not conversation_id:
                return []
            conversation = await manager.get_conversation(umo, conversation_id)
            if conversation is None:
                return []
            raw = getattr(conversation, "history", "") or "[]"
            history = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exc:
            logger.debug("Wake-word TTS: conversation history unavailable: %s", exc)
            return []

        if not isinstance(history, list):
            return []

        turns: list[dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            content = item.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            text = str(content or "").strip()
            if not text:
                continue
            turns.append({"role": role, "content": text[:CONTEXT_MESSAGE_MAX_CHARS]})
        return turns[-limit:]

    def wake_prefixes(self) -> tuple[str, ...]:
        """AstrBot's configured wake prefixes (``["/"]`` by default)."""
        if self._wake_prefixes is not None:
            return self._wake_prefixes
        prefixes: tuple[str, ...] = ("/",)
        try:
            config = self.context.get_config()
            raw = config.get("wake_prefix") if hasattr(config, "get") else None
            if isinstance(raw, str):
                prefixes = (raw,) if raw else ("/",)
            elif isinstance(raw, list | tuple):
                parsed = tuple(str(item) for item in raw if item)
                if parsed:
                    prefixes = parsed
        except Exception as exc:
            logger.debug("Wake-word TTS: cannot read wake_prefix: %s", exc)
        self._wake_prefixes = prefixes
        return prefixes

    # --------------------------------------------------------- session voice ---

    def _load_session_voices(self) -> None:
        try:
            if not self._voice_file.is_file():
                return
            raw = json.loads(self._voice_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._session_voices = {
                    str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v
                }
        except (OSError, ValueError) as exc:
            logger.warning("Wake-word TTS: cannot read session voices: %s", exc)

    def _save_session_voices(self) -> None:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(list(self._session_voices.items())[-MAX_SESSION_VOICE_ENTRIES:])
            self._voice_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Wake-word TTS: cannot persist session voices: %s", exc)

    def _session_key(self, event: AstrMessageEvent) -> str:
        return str(getattr(event, "unified_msg_origin", "") or event.get_session_id())

    def _session_voice(self, event: AstrMessageEvent, settings: PluginSettings) -> str:
        if not settings.allow_session_voice_override:
            return ""
        return self._session_voices.get(self._session_key(event), "")

    # ---------------------------------------------------------------- cache ---

    def _prune_once(self) -> None:
        try:
            self._service.store.prune_sync()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Wake-word TTS: cache prune failed: %s", exc)

    async def _prune_loop(self) -> None:
        """Background housekeeping; cancelled by :meth:`terminate`."""
        try:
            while True:
                await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
                await self._service.store.prune()
        except asyncio.CancelledError:  # pragma: no cover - normal shutdown
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Wake-word TTS: cache prune loop stopped: %s", exc)

    # -------------------------------------------------------------- guards ---

    def _looks_like_command(self, event: AstrMessageEvent) -> bool:
        """Return True when the *original* message used an AstrBot wake prefix.

        AstrBot strips the wake prefix from ``event.message_str`` before handlers
        run, but ``message_obj.message_str`` keeps the original text.
        """
        original = ""
        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            original = str(getattr(message_obj, "message_str", "") or "")
        if not original:
            original = str(event.message_str or "")
        stripped = original.lstrip()
        if not stripped:
            return False
        if stripped.startswith("/"):
            return True
        prefixes = tuple(p for p in self.wake_prefixes() if p and p != "/")
        return any(stripped.startswith(prefix) for prefix in prefixes)

    def _passes_cooldown(self, event: AstrMessageEvent, settings: PluginSettings) -> bool:
        if settings.cooldown_seconds <= 0:
            return True
        key = self._session_key(event)
        now = time.monotonic()
        last = self._last_trigger.get(key, 0.0)
        if now - last < settings.cooldown_seconds:
            return False
        self._last_trigger[key] = now
        if len(self._last_trigger) > MAX_SESSION_VOICE_ENTRIES:
            cutoff = now - max(settings.cooldown_seconds * 10, 3600.0)
            self._last_trigger = {k: v for k, v in self._last_trigger.items() if v >= cutoff}
        return True

    # -------------------------------------------------------------- reply ---

    @staticmethod
    def _record_for(path: Path) -> Any:
        """Build a ``Record`` component pointing at ``path``.

        ``Record.fromFileSystem`` keeps the plain path in ``path`` and stores a
        ``file://`` URI in ``file``, which is what current AstrBot versions
        resolve best. Older builds without the helper get the documented
        ``Record(file=..., url=...)`` form instead.
        """
        from_file_system = getattr(Record, "fromFileSystem", None)
        if callable(from_file_system):
            return from_file_system(path=str(path))
        return Record(file=str(path), url=str(path))

    def _reply_chain(self, outcome: SynthesisOutcome) -> list[Any]:
        return [self._record_for(path) for path in outcome.paths]

    def _quoted_text(self, event: AstrMessageEvent) -> str:
        """Return the quoted message text when the user replied to something."""
        try:
            messages = event.get_messages()
        except Exception:
            return ""
        for component in messages or []:
            if type(component).__name__ != "Reply":
                continue
            text = str(getattr(component, "message_str", "") or "").strip()
            if text:
                return text
        return ""

    async def _synthesize_and_reply(
        self,
        event: AstrMessageEvent,
        raw_text: str,
        settings: PluginSettings,
        *,
        source: str,
    ) -> AsyncGenerator[Any, None]:
        """Shared pipeline for the wake word and the ``/tts`` command."""
        if not settings.enabled:
            return

        session_voice = self._session_voice(event, settings)
        if session_voice:
            settings = settings.with_voice(session_voice)

        if not self._passes_cooldown(event, settings):
            logger.debug("Wake-word TTS: cooldown active, ignoring trigger.")
            return

        try:
            # ``settings`` carries this message's wake-word profile and session
            # voice, so it is passed per request instead of mutating the service.
            history = await self._conversation_history(
                event,
                settings.llm_context_messages,
            )
            outcome = await self._service.synthesize(
                raw_text,
                settings=settings,
                history=history,
            )
        except TtsError as exc:
            logger.warning("Wake-word TTS failed: %s", exc.message)
            if settings.reply_on_error:
                yield event.plain_result(f"🔇 {exc.user_message()}")
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Wake-word TTS unexpected error: %s", exc, exc_info=True)
            if settings.reply_on_error:
                yield event.plain_result("🔇 语音合成时发生未知错误，请查看 AstrBot 日志。")
            return

        if not outcome.ok:
            if settings.reply_on_error:
                yield event.plain_result("🔇 没有生成任何音频。")
            return

        logger.info(
            "Wake-word TTS replied via %s: model=%s voice=%s segments=%s bytes=%s llm=%s",
            source,
            outcome.model,
            outcome.voice,
            len(outcome.paths),
            outcome.total_bytes,
            "yes" if outcome.director is not None else "no",
        )

        if settings.echo_source_text and outcome.segments:
            yield event.plain_result(outcome.segments[0])
        yield event.chain_result(self._reply_chain(outcome))
        if outcome.truncated:
            yield event.plain_result("（文本过长，只合成了前面的内容，可在配置中调整字符上限）")

        if source == "wake" and settings.stop_event_on_wake:
            # Stop the message from also reaching the LLM pipeline.
            event.stop_event()
            should_call_llm = getattr(event, "should_call_llm", None)
            if callable(should_call_llm):
                should_call_llm(False)

    # ------------------------------------------------------------- handlers ---

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """Wake-word listener: fires on any message starting with a wake word."""
        settings = self.settings()
        if not settings.enabled or not settings.all_wake_words:
            return
        if settings.ignore_slash_commands and self._looks_like_command(event):
            return

        raw_text = str(event.message_str or "")
        # I. Match the wake word
        # 1. Longest wake word wins, so more specific profiles take precedence.
        wake_word, remainder = match_wake_word(
            raw_text,
            settings.all_wake_words,
            require_separator=settings.wake_require_separator,
            case_sensitive=settings.wake_case_sensitive,
        )
        if not wake_word:
            return

        # II. Layer the per-wake-word profile over the global settings
        profile = settings.profile_for(wake_word)
        if profile is not None:
            settings = profile.apply(settings)

        # III. Decide the payload
        payload = remainder if settings.strip_wake_word else raw_text
        if not payload.strip():
            # A bare wake word is a ping, not a synthesis request.
            yield event.plain_result(
                f"我在。把想听的内容写在「{wake_word}」后面即可，例如：{wake_word} 你好呀。",
            )
            if settings.stop_event_on_wake:
                event.stop_event()
                should_call_llm = getattr(event, "should_call_llm", None)
                if callable(should_call_llm):
                    should_call_llm(False)
            return

        async for result in self._synthesize_and_reply(
            event,
            payload,
            settings,
            source="wake",
        ):
            yield result

    @filter.command("tts", alias={"tts_say", "语音", "合成"})
    async def tts_command(
        self,
        event: AstrMessageEvent,
        text: str = "",
    ) -> AsyncGenerator[Any, None]:
        """把文本合成为语音并发送：/tts 要说的内容"""
        settings = self.settings()
        if not settings.enabled:
            yield event.plain_result("语音合成插件当前已停用。")
            return
        if not settings.enable_command:
            yield event.plain_result("语音合成指令已在插件配置中关闭。")
            return

        payload = extract_command_argument(str(event.message_str or "")) or (text or "")
        if not payload.strip():
            quoted = self._quoted_text(event)
            if quoted:
                payload = quoted
        if not payload.strip():
            yield event.plain_result(self._usage_text(settings))
            return

        async for result in self._synthesize_and_reply(
            event,
            payload,
            settings,
            source="command",
        ):
            yield result

    @filter.command("tts_voice", alias={"音色"})
    async def tts_voice_command(
        self,
        event: AstrMessageEvent,
        voice: str = "",
    ) -> AsyncGenerator[Any, None]:
        """查看或设置当前会话的音色：/tts_voice [音色名 | reset]"""
        settings = self.settings()
        if not settings.enabled:
            yield event.plain_result("语音合成插件当前已停用。")
            return
        if not settings.allow_session_voice_override:
            yield event.plain_result(
                "当前配置未开启「按会话切换音色」。如需使用，请在插件配置中开启"
                "「允许 /tts_voice 覆盖音色」。",
            )
            return
        if settings.admin_only_voice_switch and not event.is_admin():
            yield event.plain_result("只有管理员可以切换音色。")
            return

        argument = extract_command_argument(str(event.message_str or "")) or (voice or "")
        argument = argument.strip()
        key = self._session_key(event)
        current = self._session_voices.get(key, "")

        if not argument or argument.lower() in {"show", "status", "查看"}:
            effective = current or settings.voice
            yield event.plain_result(
                f"当前会话音色：{effective}"
                f"{'（会话覆盖）' if current else '（沿用全局配置）'}\n"
                f"默认音色：{settings.voice}\n"
                "设置方式：/tts_voice Cherry ；恢复默认：/tts_voice reset",
            )
            return

        if argument.lower() in {"reset", "clear", "default", "默认", "清除", "重置"}:
            self._session_voices.pop(key, None)
            self._save_session_voices()
            yield event.plain_result(f"已恢复使用全局音色：{settings.voice}")
            return

        self._session_voices[key] = argument
        if len(self._session_voices) > MAX_SESSION_VOICE_ENTRIES:
            self._session_voices = dict(
                list(self._session_voices.items())[-MAX_SESSION_VOICE_ENTRIES:]
            )
        self._save_session_voices()
        yield event.plain_result(
            f"已把当前会话音色设置为：{argument}\n发送一条语音试试，或使用 /tts_voice reset 还原。",
        )

    @filter.command("tts_llm", alias={"tts_direct", "tts_script"})
    async def tts_llm_command(
        self,
        event: AstrMessageEvent,
        text: str = "",
    ) -> AsyncGenerator[Any, None]:
        """只运行 LLM 预处理并显示结果（不合成语音）：/tts_llm 要朗读的内容"""
        settings = self.settings()
        if not settings.enabled:
            yield event.plain_result("语音合成插件当前已停用。")
            return
        if not settings.llm_enable:
            yield event.plain_result(
                "LLM 预处理当前是关闭状态。可在插件配置的「LLM 预处理」中开启后重试。",
            )
            return

        payload = extract_command_argument(str(event.message_str or "")) or (text or "")
        if not payload.strip():
            quoted = self._quoted_text(event)
            payload = quoted
        if not payload.strip():
            yield event.plain_result("用法：/tts_llm 想朗读的内容（不会发送语音）")
            return

        history = await self._conversation_history(event, settings.llm_context_messages)
        capabilities = TtsCapabilities.from_settings(settings)
        try:
            result = await self._service.direct(
                payload,
                settings=settings,
                history=history,
            )
        except TtsError as exc:
            yield event.plain_result(f"🧠 LLM 预处理失败：{exc.user_message()}")
            return

        if result.text == payload and not result.instructions:
            yield event.plain_result(
                "🧠 LLM 预处理没有改变文本，也没有产出语气指令。\n"
                "请检查「LLM 预处理」是否已启用并选中了可用的 Provider。",
            )
            return

        lines = [
            "【LLM 预处理结果】（仅供调试，未发送语音）",
            f"使用模型：{result.model or settings.llm_model or '（provider 默认）'}",
            f"原文（{len(payload)} 字）：{payload}",
            f"改写（{len(result.text)} 字）：{result.text}",
        ]
        if capabilities.supports_instructions:
            lines.append(f"语气指令：{result.instructions or '（无）'}")
        else:
            lines.append(
                f"语气指令：{result.instructions or '（无）'}"
                f"（当前模型 {settings.model} 不支持 instructions，已忽略）",
            )
        if capabilities.tags_enabled:
            lines.append(f"使用标签：{'、'.join(result.tags) or '（无）'}")
        if result.stripped_tags:
            lines.append(f"已移除的非法标签：{'、'.join(result.stripped_tags)}")
        if result.reason:
            lines.append(f"模型说明：{result.reason}")
        yield event.plain_result("\n".join(lines))

    @filter.command("tts_status", alias={"tts_info"})
    async def tts_status_command(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[Any, None]:
        """查看唤醒词 TTS 插件的当前状态。"""
        settings = self.settings()
        session_voice = self._session_voice(event, settings)
        cache_files = self._service.store.count()

        lines = [
            "【唤醒词 TTS 状态】",
            f"插件状态：{'启用' if settings.enabled else '停用'}",
            f"唤醒词：{'、'.join(settings.all_wake_words) or '（未配置）'}",
            f"模型：{settings.model}",
            f"音色：{session_voice or settings.voice}{'（会话覆盖）' if session_voice else ''}",
            f"语种：{settings.language_type}",
            f"地域：{settings.region}",
            f"接口地址：{settings.synthesis_route()}",
            f"API Key：{mask_secret(settings.api_key)}",
            f"单段最大字符：{settings.max_chars}"
            f"（{'自动分段' if settings.split_long_text else '超出直接截断'}，"
            f"最多 {settings.max_segments} 段）",
            f"并发上限：{settings.max_concurrent}，冷却：{settings.cooldown_seconds:g} 秒",
            f"音频缓存：{cache_files} 个文件，"
            f"保留 {settings.cache_ttl_hours:g} 小时 / 最多 {settings.max_cache_files} 个",
            f"缓存目录：{self._service.store.directory}",
            "—— LLM 预处理（文本导演） ——",
            f"状态：{self._describe_llm(settings)}",
            f"改写正文：{'允许' if settings.llm_rewrite else '不允许（仅产出语气指令）'}"
            f"，应用语气指令：{'是' if settings.llm_apply_instructions else '否'}",
            f"细粒度标签：{settings.llm_tag_style}",
            f"上下文条数：{settings.llm_context_messages}"
            f"，失败兜底：{'开启' if settings.llm_fail_open else '关闭'}",
        ]
        if not settings.llm_uses_provider():
            lines.append(f"大模型地址：{settings.llm_base_url}")
            lines.append(f"大模型 API Key：{mask_secret(settings.llm_api_key)}")
            if settings.llm_extra_headers:
                names = "、".join(name for name, _ in settings.llm_extra_headers)
                lines.append(f"自定义请求头：{names}")
        if settings.profiles:
            profile_lines = [
                f"  · {p.wake_word} → {p.model or settings.model} / {p.voice or settings.voice}"
                for p in settings.profiles
            ]
            lines.append("专属唤醒词配置：")
            lines.extend(profile_lines)
        yield event.plain_result("\n".join(lines))

    @filter.command("tts_help", alias={"tts_usage"})
    async def tts_help_command(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[Any, None]:
        """查看唤醒词 TTS 插件的使用说明。"""
        yield event.plain_result(self._usage_text(self.settings()))

    # ---------------------------------------------------------------- texts ---

    def _usage_text(self, settings: PluginSettings) -> str:
        wake_words = "、".join(settings.all_wake_words) or "（未配置，请先在插件配置中填写）"
        sample = settings.all_wake_words[0] if settings.all_wake_words else "唤醒词"
        return (
            "【唤醒词 TTS 使用说明】\n"
            f"唤醒词：{wake_words}\n"
            f"用法一：直接说「{sample} 想听的内容」\n"
            "用法二：/tts 想听的内容（也可以 /tts 后引用一条消息）\n"
            "/tts_voice [音色名]  查看或切换当前会话音色（reset 还原）\n"
            "/tts_llm 想朗读的内容  只运行 LLM 预处理并显示结果（不发送语音）\n"
            "/tts_status          查看当前模型、音色、LLM 预处理与缓存状态\n"
            "/tts_help            显示本说明\n"
            f"当前模型：{settings.model}（推荐 {DEFAULT_MODEL}），音色：{settings.voice}\n"
            "提示：语音只在 OneBot v11(QQ)、Telegram、企业微信等支持语音的消息平台可用。"
        )
