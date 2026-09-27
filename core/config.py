"""Configuration parsing for the wake-word TTS plugin.

I. Why a dedicated parser

1. ``_conf_schema.json`` guarantees the *shape* of a fresh config file, but users
   hand-edit ``data/config/<plugin>_config.json`` and older configs may miss keys
   added by newer versions.
2. Therefore every value goes through a coercing reader instead of being used
   directly, so a malformed value degrades to its default instead of crashing a
   message handler.
3. The parser is intentionally free of ``astrbot`` imports so it can be tested
   standalone.

II. Layout of the resulting objects

1. ``TtsProfile`` – one wake word bound to its own model/voice overrides.
2. ``PluginSettings`` – the flattened global settings plus the profile list.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import TtsConfigError

# ---------------------------------------------------------------- constants ---

REGION_BEIJING = "beijing"
REGION_SINGAPORE = "singapore"
REGION_QWENAI = "qwenai"
REGION_CUSTOM = "custom"

DEFAULT_BASE_URLS: dict[str, str] = {
    REGION_BEIJING: "https://dashscope.aliyuncs.com/api/v1",
    REGION_SINGAPORE: "https://dashscope-intl.aliyuncs.com/api/v1",
    REGION_QWENAI: "https://maas.qianwenaiapi.com/api/v1",
}

#: The DashScope route for non-realtime Qwen-TTS synthesis. It is also used by
#: :func:`normalize_base_url` to recognise URLs users copy from the console.
DASHSCOPE_SYNTHESIS_ROUTE = "/services/aigc/multimodal-generation/generation"

SUPPORTED_LANGUAGE_TYPES: tuple[str, ...] = (
    "Auto",
    "Chinese",
    "English",
    "German",
    "Italian",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Korean",
    "French",
    "Russian",
)

#: Model IDs verified to answer on the HTTP non-realtime synthesis route.
#: The Qwen-Audio-TTS family (``qwen-audio-3.x-tts-*``) is deliberately absent:
#: those models are served over the WebSocket realtime API only, and calling
#: them here returns ``InvalidParameter: url error``.
HTTP_TTS_MODELS: tuple[str, ...] = (
    "qwen3-tts-flash",
    "qwen3-tts-instruct-flash",
    "qwen-tts",
    "qwen-tts-latest",
)

DEFAULT_MODEL = "qwen3-tts-flash"
DEFAULT_VOICE = "Cherry"

# -------------------------------------------------------------- LLM stage ---

LLM_MODE_PROVIDER = "astrbot_provider"
LLM_MODE_DIRECT = "direct"

LLM_MODES: tuple[str, ...] = (LLM_MODE_PROVIDER, LLM_MODE_DIRECT)

#: Default endpoint for direct mode. The field is free text, so any
#: OpenAI-compatible gateway (Opencode, a self-hosted proxy, ...) works too.
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com/v1"

#: DeepSeek ids offered as hints. The field stays free text so an Opencode or
#: any other OpenAI-compatible model id can be typed in.
DEEPSEEK_MODELS: tuple[str, ...] = (
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)

AUTH_STYLE_BEARER = "bearer"
AUTH_STYLE_X_API_KEY = "x-api-key"
AUTH_STYLE_NONE = "none"

LLM_AUTH_STYLES: tuple[str, ...] = (
    AUTH_STYLE_BEARER,
    AUTH_STYLE_X_API_KEY,
    AUTH_STYLE_NONE,
)

#: Inline tag vocabularies the director may ask the LLM to use. They live in the
#: config module because they are configuration values; the prompt text for each
#: vocabulary lives in :mod:`core.director`.
TAG_STYLE_NONE = "none"
TAG_STYLE_QWEN_AUDIO = "qwen-audio"
TAG_STYLE_SSML = "ssml"
TAG_STYLE_CUSTOM = "custom"

TAG_STYLES: tuple[str, ...] = (
    TAG_STYLE_NONE,
    TAG_STYLE_QWEN_AUDIO,
    TAG_STYLE_SSML,
    TAG_STYLE_CUSTOM,
)

# ------------------------------------------------------------ coercion utils ---


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce JSON-ish scalars to ``bool`` without raising."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "y"}:
            return True
        if lowered in {"0", "false", "no", "off", "n", ""}:
            return False
    return default


def _as_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _as_float(value: Any, default: float, *, minimum: float | None = None) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float):
        return str(value)
    return default


def _as_str_list(value: Any) -> list[str]:
    """Accept a real list, a comma/newline separated string, or a scalar."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace("，", ",").replace("\n", ",").split(",")
        return [p.strip() for p in parts if p.strip()]
    if isinstance(value, dict):
        # ``template_list``/``object`` shaped mapping: keep any string values.
        return [s for s in (_as_str(v) for v in value.values()) if s]
    if isinstance(value, list | tuple | set):
        items: list[str] = []
        for entry in value:
            text = _as_str(entry)
            if text:
                items.append(text)
        return items
    text = _as_str(value)
    return [text] if text else []


def dedupe_preserving_order(values: list[str]) -> list[str]:
    """Case-insensitively deduplicate while keeping the first spelling."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def mask_secret(secret: str) -> str:
    """Mask an API key for display: keep a short head and tail only."""
    if not secret:
        return "（未配置）"
    if len(secret) <= 10:
        return f"{secret[:2]}****"
    return f"{secret[:6]}****{secret[-4:]}"


def _as_headers(value: Any) -> tuple[tuple[str, str], ...]:
    """Parse extra HTTP headers from the three shapes users actually type.

    (1) a real mapping (``object``/``dict`` schema type)
    (2) a JSON object string
    (3) one ``Key: Value`` per line, which is what a textarea invites
    """
    pairs: list[tuple[str, str]] = []

    if isinstance(value, dict):
        for key, item in value.items():
            name = _as_str(key)
            text = _as_str(item)
            if name:
                pairs.append((name, text))
        return tuple(pairs)

    if not isinstance(value, str) or not value.strip():
        return ()

    raw = value.strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            for key, item in parsed.items():
                name = _as_str(key)
                if name:
                    pairs.append((name, _as_str(item)))
            return tuple(pairs)

    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        name, _, text = line.partition(":")
        name = name.strip().strip('"').strip("'")
        text = text.strip().strip(",").strip('"').strip("'")
        if name:
            pairs.append((name, text))
    return tuple(pairs)


def _parse_llm_mode(value: Any) -> str:
    mode = _as_str(value).lower()
    return mode if mode in LLM_MODES else LLM_MODE_PROVIDER


def _parse_auth_style(value: Any) -> str:
    style = _as_str(value).lower()
    return style if style in LLM_AUTH_STYLES else AUTH_STYLE_BEARER


def _parse_tag_style(value: Any) -> str:
    style = _as_str(value).lower()
    return style if style in TAG_STYLES else TAG_STYLE_NONE


# -------------------------------------------------------------- base url ---


def normalize_base_url(raw: str, region: str) -> str:
    """Return a usable ``.../api/v1`` base URL for the DashScope HTTP API.

    Users copy wildly different strings from the console (host only, host with
    ``/api/v1``, or the full synthesis route). All of them are accepted here and
    reduced to the ``/api/v1`` root that the client appends routes to.
    """
    candidate = (raw or "").strip().rstrip("/")
    if not candidate:
        candidate = DEFAULT_BASE_URLS.get(region, DEFAULT_BASE_URLS[REGION_BEIJING])

    if candidate.endswith(DASHSCOPE_SYNTHESIS_ROUTE):
        candidate = candidate[: -len(DASHSCOPE_SYNTHESIS_ROUTE)]
    candidate = candidate.rstrip("/")

    if not candidate.endswith("/api/v1"):
        candidate = f"{candidate}/v1" if candidate.endswith("/api") else f"{candidate}/api/v1"
    return candidate


# ---------------------------------------------------------------- profiles ---


@dataclass(frozen=True, slots=True)
class TtsProfile:
    """A wake word that overrides part of the global TTS settings."""

    wake_word: str
    model: str = ""
    voice: str = ""
    language_type: str = ""
    instructions: str = ""
    optimize_instructions: bool | None = None

    def apply(self, base: PluginSettings) -> PluginSettings:
        """Return ``base`` with this profile's non-empty overrides applied."""
        changes: dict[str, Any] = {}
        if self.model:
            changes["model"] = self.model
        if self.voice:
            changes["voice"] = self.voice
        if self.language_type:
            changes["language_type"] = self.language_type
        if self.instructions:
            changes["instructions"] = self.instructions
        if self.optimize_instructions is not None:
            changes["optimize_instructions"] = self.optimize_instructions
        return replace(base, **changes) if changes else base


def _parse_profiles(raw: Any) -> tuple[TtsProfile, ...]:
    """Parse the ``wake_profiles`` config value.

    Three shapes occur in the wild:

    (1) ``template_list`` output – a list of dicts carrying ``__template_key``.
    (2) nested ``object`` config – the same keys without ``__template_key``.
    (3) a plain list of wake-word strings, which bind no overrides at all.
    """
    if raw is None:
        return ()
    entries: list[Any]
    if isinstance(raw, dict):
        # ``object``-style schema stores one profile directly.
        entries = [raw]
    elif isinstance(raw, list | tuple):
        entries = list(raw)
    else:
        return ()

    profiles: list[TtsProfile] = []
    for entry in entries:
        if isinstance(entry, str):
            wake_word = entry.strip()
            if wake_word:
                profiles.append(TtsProfile(wake_word=wake_word))
            continue
        if not isinstance(entry, dict):
            continue
        wake_word = _as_str(entry.get("wake_word"))
        if not wake_word:
            continue
        optimize_raw = entry.get("optimize_instructions")
        language_type = _as_str(entry.get("language_type"))
        # The WebUI renders the profile language as a select, so "inherit" is the
        # explicit "no override" choice.
        inherits = language_type.casefold() in {"inherit", "none", "default"}
        if inherits or (language_type and language_type not in SUPPORTED_LANGUAGE_TYPES):
            language_type = ""
        profiles.append(
            TtsProfile(
                wake_word=wake_word,
                model=_as_str(entry.get("model")),
                voice=_as_str(entry.get("voice")),
                language_type=language_type,
                instructions=_as_str(entry.get("instructions")),
                optimize_instructions=(
                    None if optimize_raw in (None, "") else _as_bool(optimize_raw, False)
                ),
            ),
        )
    return tuple(profiles)


# --------------------------------------------------------------- settings ---


@dataclass(frozen=True, slots=True)
class PluginSettings:
    """Immutable snapshot of the plugin configuration."""

    # -- behaviour ---------------------------------------------------------
    enabled: bool = True
    enable_command: bool = True
    ignore_slash_commands: bool = True
    stop_event_on_wake: bool = True
    echo_source_text: bool = False
    cooldown_seconds: float = 0.0
    max_concurrent: int = 3
    reply_on_error: bool = True
    allow_session_voice_override: bool = True
    admin_only_voice_switch: bool = False

    # -- wake word ---------------------------------------------------------
    wake_words: tuple[str, ...] = ()
    wake_require_separator: bool = False
    wake_case_sensitive: bool = False
    strip_wake_word: bool = True

    # -- DashScope ---------------------------------------------------------
    region: str = REGION_BEIJING
    base_url: str = DEFAULT_BASE_URLS[REGION_BEIJING]
    api_key: str = ""
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    language_type: str = "Auto"
    instructions: str = ""
    optimize_instructions: bool = False
    timeout_seconds: float = 60.0
    max_retries: int = 2

    # -- text handling -----------------------------------------------------
    max_chars: int = 300
    split_long_text: bool = True
    max_segments: int = 3
    clean_markdown: bool = True
    clean_url: bool = True
    clean_emoji: bool = False

    # -- storage -----------------------------------------------------------
    cache_ttl_hours: float = 6.0
    max_cache_files: int = 200

    # -- LLM director (pre-synthesis script preparation) --------------------
    llm_enable: bool = True
    llm_mode: str = LLM_MODE_PROVIDER
    llm_provider_id: str = ""
    llm_model: str = ""
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: str = ""
    llm_auth_style: str = AUTH_STYLE_BEARER
    llm_extra_headers: tuple[tuple[str, str], ...] = ()
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 1
    llm_temperature: float = 0.3
    llm_max_tokens: int = 800
    llm_context_messages: int = 6
    llm_rewrite: bool = True
    llm_apply_instructions: bool = True
    llm_max_input_chars: int = 1200
    llm_fail_open: bool = True
    llm_system_prompt: str = ""
    llm_tag_style: str = TAG_STYLE_NONE
    llm_custom_tags: str = ""

    # -- profiles ----------------------------------------------------------
    profiles: tuple[TtsProfile, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------ derived ---

    @property
    def all_wake_words(self) -> tuple[str, ...]:
        """Global wake words plus every profile wake word, longest first.

        Sorting by descending length makes ``小助手你好`` win over ``小助手`` so
        the more specific wake word gets to pick its own profile.
        """
        words = list(self.wake_words)
        words.extend(p.wake_word for p in self.profiles)
        unique = dedupe_preserving_order(words)
        return tuple(sorted(unique, key=len, reverse=True))

    def profile_for(self, wake_word: str) -> TtsProfile | None:
        """Return the profile bound to ``wake_word`` (case-insensitive)."""
        key = wake_word.casefold()
        for profile in self.profiles:
            if profile.wake_word.casefold() == key:
                return profile
        return None

    def synthesis_route(self) -> str:
        """Full URL of the Qwen-TTS synthesis endpoint."""
        return f"{self.base_url.rstrip('/')}{DASHSCOPE_SYNTHESIS_ROUTE}"

    # ------------------------------------------------------- LLM helpers ---

    @property
    def llm_enabled(self) -> bool:
        """True when the director stage should run for this request."""
        return self.llm_enable

    def llm_headers(self) -> dict[str, str]:
        """Extra HTTP headers for direct mode as a plain dict."""
        return dict(self.llm_extra_headers)

    def llm_uses_provider(self) -> bool:
        """True when the call goes through AstrBot's provider system."""
        return self.llm_mode == LLM_MODE_PROVIDER

    def llm_fingerprint(self) -> tuple[Any, ...]:
        """Identity of the LLM transport, used to rebuild the client on change."""
        return (
            self.llm_mode,
            self.llm_provider_id,
            self.llm_model,
            self.llm_base_url,
            self.llm_api_key,
            self.llm_auth_style,
            self.llm_extra_headers,
            self.llm_timeout_seconds,
            self.llm_max_retries,
            self.llm_temperature,
            self.llm_max_tokens,
        )

    def llm_requires_credentials(self) -> bool:
        """Validate direct mode before a request is attempted."""
        if self.llm_uses_provider():
            return True
        if not self.llm_base_url:
            raise TtsConfigError(
                "missing LLM base url",
                hint="LLM 预处理使用直连模式，请先填写大模型的 Base URL。",
            )
        if self.llm_auth_style != AUTH_STYLE_NONE and not self.llm_api_key:
            raise TtsConfigError(
                "missing LLM api key",
                hint=(
                    "LLM 预处理使用直连模式且鉴权方式不为 none，请填写 API Key；"
                    "如果网关自带鉴权，可把鉴权方式改为 none 并补充自定义请求头。"
                ),
            )
        return True

    def validate_for_synthesis(self) -> None:
        """Raise :class:`TtsConfigError` when a request cannot be attempted."""
        if not self.api_key:
            raise TtsConfigError(
                "missing api_key",
                hint=(
                    "还没有配置 DashScope API Key。请到 AstrBot 的「插件」页面打开本插件配置，"
                    "在「DashScope API Key」中填入百炼（Model Studio）API Key，"
                    "获取地址：https://bailian.console.aliyun.com/?tab=model#/api-key"
                ),
            )
        if not self.model:
            raise TtsConfigError(
                "missing model",
                hint=(
                    "没有配置 TTS 模型名称，请在插件配置中填写 model"
                    "（如 qwen3-tts-flash、qwen-tts）。"
                ),
            )
        if not self.voice:
            raise TtsConfigError(
                "missing voice",
                hint="没有配置音色（voice），请在插件配置中填写，例如 Cherry。",
            )

    # ----------------------------------------------------------- factory ---

    @classmethod
    def from_dict(
        cls,
        raw: Any,
        *,
        fallback_voice: str = DEFAULT_VOICE,
    ) -> PluginSettings:
        """Build settings from the raw ``AstrBotConfig`` mapping.

        ``fallback_voice`` lets a per-session voice override be layered on top
        without re-parsing the whole config.
        """
        if not isinstance(raw, dict):
            raw = {}

        region = _as_str(raw.get("region"), REGION_BEIJING).lower()
        if region not in {
            REGION_BEIJING,
            REGION_SINGAPORE,
            REGION_QWENAI,
            REGION_CUSTOM,
        }:
            region = REGION_BEIJING

        base_url = normalize_base_url(_as_str(raw.get("base_url")), region)

        language_type = _as_str(raw.get("language_type"), "Auto") or "Auto"
        if language_type not in SUPPORTED_LANGUAGE_TYPES:
            language_type = "Auto"

        voice = _as_str(raw.get("voice"), fallback_voice) or fallback_voice
        if not voice:
            voice = DEFAULT_VOICE

        wake_words = dedupe_preserving_order(_as_str_list(raw.get("wake_words")))

        return cls(
            enabled=_as_bool(raw.get("enabled"), True),
            enable_command=_as_bool(raw.get("enable_command"), True),
            ignore_slash_commands=_as_bool(raw.get("ignore_slash_commands"), True),
            stop_event_on_wake=_as_bool(raw.get("stop_event_on_wake"), True),
            echo_source_text=_as_bool(raw.get("echo_source_text"), False),
            cooldown_seconds=_as_float(raw.get("cooldown_seconds"), 0.0, minimum=0.0),
            max_concurrent=max(1, _as_int(raw.get("max_concurrent"), 3, minimum=1)),
            reply_on_error=_as_bool(raw.get("reply_on_error"), True),
            allow_session_voice_override=_as_bool(
                raw.get("allow_session_voice_override"),
                True,
            ),
            admin_only_voice_switch=_as_bool(
                raw.get("admin_only_voice_switch"),
                False,
            ),
            wake_words=tuple(wake_words),
            wake_require_separator=_as_bool(raw.get("wake_require_separator"), False),
            wake_case_sensitive=_as_bool(raw.get("wake_case_sensitive"), False),
            strip_wake_word=_as_bool(raw.get("strip_wake_word"), True),
            region=region,
            base_url=base_url,
            api_key=_as_str(raw.get("api_key")),
            model=_as_str(raw.get("model"), DEFAULT_MODEL) or DEFAULT_MODEL,
            voice=voice,
            language_type=language_type,
            instructions=_as_str(raw.get("instructions")),
            optimize_instructions=_as_bool(raw.get("optimize_instructions"), False),
            timeout_seconds=_as_float(raw.get("timeout_seconds"), 60.0, minimum=5.0),
            max_retries=max(0, _as_int(raw.get("max_retries"), 2, minimum=0)),
            max_chars=max(20, _as_int(raw.get("max_chars"), 300, minimum=20)),
            split_long_text=_as_bool(raw.get("split_long_text"), True),
            max_segments=max(1, _as_int(raw.get("max_segments"), 3, minimum=1)),
            clean_markdown=_as_bool(raw.get("clean_markdown"), True),
            clean_url=_as_bool(raw.get("clean_url"), True),
            clean_emoji=_as_bool(raw.get("clean_emoji"), False),
            cache_ttl_hours=_as_float(raw.get("cache_ttl_hours"), 6.0, minimum=0.0),
            max_cache_files=max(1, _as_int(raw.get("max_cache_files"), 200, minimum=1)),
            profiles=_parse_profiles(raw.get("wake_profiles")),
            llm_enable=_as_bool(raw.get("llm_enable"), True),
            llm_mode=_parse_llm_mode(raw.get("llm_mode")),
            llm_provider_id=_as_str(raw.get("llm_provider_id")),
            llm_model=_as_str(raw.get("llm_model")),
            llm_base_url=_as_str(raw.get("llm_base_url"), DEFAULT_LLM_BASE_URL)
            or DEFAULT_LLM_BASE_URL,
            llm_api_key=_as_str(raw.get("llm_api_key")),
            llm_auth_style=_parse_auth_style(raw.get("llm_auth_style")),
            llm_extra_headers=_as_headers(raw.get("llm_extra_headers")),
            llm_timeout_seconds=_as_float(
                raw.get("llm_timeout_seconds"),
                20.0,
                minimum=3.0,
            ),
            llm_max_retries=max(0, _as_int(raw.get("llm_max_retries"), 1, minimum=0)),
            llm_temperature=_as_float(raw.get("llm_temperature"), 0.3, minimum=0.0),
            llm_max_tokens=max(64, _as_int(raw.get("llm_max_tokens"), 800, minimum=64)),
            llm_context_messages=max(
                0,
                _as_int(raw.get("llm_context_messages"), 6, minimum=0),
            ),
            llm_rewrite=_as_bool(raw.get("llm_rewrite"), True),
            llm_apply_instructions=_as_bool(raw.get("llm_apply_instructions"), True),
            llm_max_input_chars=max(
                50,
                _as_int(raw.get("llm_max_input_chars"), 1200, minimum=50),
            ),
            llm_fail_open=_as_bool(raw.get("llm_fail_open"), True),
            llm_system_prompt=_as_str(raw.get("llm_system_prompt")),
            llm_tag_style=_parse_tag_style(raw.get("llm_tag_style")),
            llm_custom_tags=_as_str(raw.get("llm_custom_tags")),
        )

    def with_voice(self, voice: str) -> PluginSettings:
        """Return a copy using ``voice`` as the effective voice."""
        return replace(self, voice=voice) if voice else self
