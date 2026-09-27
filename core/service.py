"""Synthesis orchestration: settings + HTTP client + audio store.

The service is the only place that knows the full pipeline, which keeps
``main.py`` limited to translating AstrBot events into calls and back.

I. Pipeline

1. Optionally hand the raw text to the LLM "director" (:mod:`core.director`),
   which rewrites it into a spoken script and may add a delivery instruction.
2. Sanitize the resulting text.
3. Reject empty payloads early with a user-facing hint.
4. Split long text into segments that fit the model's per-request limit.
5. Synthesize each segment, store it, and return the local paths in order.

II. Concurrency

A semaphore bounds simultaneous upstream requests so a busy group cannot exhaust
the DashScope quota. Failed segments abort the request; already stored files stay
on disk and are removed by the normal cache pruning.

III. Per-request settings

Everything that can differ between two messages (wake-word profile, session
voice, director output) is passed in as an argument rather than written back to
``self``, so concurrent messages can never observe each other's settings.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio_store import AudioStore
from .config import PluginSettings
from .dashscope_client import DashScopeTtsClient
from .director import (
    DirectorResult,
    TtsCapabilities,
    build_messages,
    parse_director_response,
)
from .errors import LlmError, TtsError, TtsTextError
from .text_utils import sanitize_text, split_for_tts


@dataclass(slots=True)
class SynthesisOutcome:
    """Everything a handler needs to reply with the generated audio."""

    paths: list[Path] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    model: str = ""
    voice: str = ""
    truncated: bool = False
    total_bytes: int = 0
    request_ids: list[str] = field(default_factory=list)
    director: DirectorResult | None = None
    """The LLM stage output, when it ran; ``None`` means the stage was skipped."""

    @property
    def ok(self) -> bool:
        return bool(self.paths)

    @property
    def spoke_text(self) -> str:
        """The text that was actually synthesized."""
        return " ".join(self.segments)

    @property
    def instructions(self) -> str:
        return self.director.instructions if self.director else ""


class TtsService:
    """Turn text into local audio files using a DashScope Qwen-TTS model."""

    def __init__(
        self,
        settings: PluginSettings,
        *,
        client: DashScopeTtsClient | None = None,
        store: AudioStore | None = None,
        llm_client: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm_client = llm_client
        self._logger = logger
        self._client = client
        # An injected client (tests, or a caller-provided transport) keeps its
        # own lifecycle until the transport settings actually change.
        self._client_fingerprint = (
            self._transport_fingerprint(settings) if client is not None else None
        )
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_size = 0

    # ----------------------------------------------------------- plumbing ---

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self._logger is None:
            return
        log_method = getattr(self._logger, level, None)
        if callable(log_method):
            log_method(message, *args)

    def _transport_fingerprint(self, settings: PluginSettings) -> tuple[Any, ...]:
        return (
            settings.base_url,
            settings.api_key,
            settings.timeout_seconds,
            settings.max_retries,
        )

    @property
    def client(self) -> DashScopeTtsClient:
        """The upstream client, rebuilt when transport settings changed."""
        fingerprint = self._transport_fingerprint(self.settings)
        if self._client is None or self._client_fingerprint != fingerprint:
            self._client = DashScopeTtsClient(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key,
                timeout=self.settings.timeout_seconds,
                max_retries=self.settings.max_retries,
                logger=self._logger,
            )
            self._client_fingerprint = fingerprint
        return self._client

    def _get_semaphore(self) -> asyncio.Semaphore:
        size = self.settings.max_concurrent
        if self._semaphore is None or self._semaphore_size != size:
            self._semaphore = asyncio.Semaphore(size)
            self._semaphore_size = size
        return self._semaphore

    async def aclose(self) -> None:
        """Release the upstream HTTP sessions."""
        if self._client is not None:
            await self._client.aclose()
        close_llm = getattr(self.llm_client, "aclose", None)
        if callable(close_llm):
            await close_llm()

    def update_settings(self, settings: PluginSettings) -> None:
        """Swap in freshly parsed settings (called on every config reload)."""
        self.settings = settings

    # ------------------------------------------------------------ director ---

    async def direct(
        self,
        raw_text: str,
        *,
        settings: PluginSettings | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> DirectorResult:
        """Run the LLM stage and return the script to speak.

        Returns a pass-through result when the stage is disabled, so callers can
        use this method unconditionally. When the stage runs but fails, the
        behaviour is governed by ``llm_fail_open``.
        """
        effective = settings or self.settings
        if not effective.llm_enable or self.llm_client is None:
            return DirectorResult(text=raw_text, model="")

        capabilities = TtsCapabilities.from_settings(effective)
        prompt_text = raw_text[: effective.llm_max_input_chars]
        if len(raw_text) > len(prompt_text):
            self._log(
                "info",
                "Director input truncated from %s to %s chars",
                len(raw_text),
                len(prompt_text),
            )

        messages = build_messages(
            text=prompt_text,
            capabilities=capabilities,
            history=history,
            system_prompt=effective.llm_system_prompt,
            allow_rewrite=effective.llm_rewrite,
        )

        try:
            reply = await self.llm_client.chat(messages)
            reply_text = reply.text if hasattr(reply, "text") else str(reply)
            reply_model = getattr(reply, "model", "") or effective.llm_model
        except LlmError as exc:
            if not effective.llm_fail_open:
                raise
            self._log(
                "warning",
                "Director failed, falling back to the original text: %s",
                exc.message,
            )
            return DirectorResult(text=raw_text, used_fallback=True, raw="")

        result = parse_director_response(
            reply_text,
            original_text=raw_text,
            capabilities=capabilities,
            model=reply_model,
        )
        if result.used_fallback and not effective.llm_fail_open:
            raise LlmError(
                "director returned no usable script",
                hint="LLM 预处理没有返回可用文本，且已关闭失败兜底。",
            )

        self._log(
            "debug",
            "Director[%s] %s chars -> %s chars, instructions=%r, tags=%s, stripped=%s",
            result.model or "?",
            len(raw_text),
            len(result.text),
            result.instructions,
            list(result.tags),
            list(result.stripped_tags),
        )
        return result

    # ---------------------------------------------------------- synthesis ---

    def prepare_segments(
        self,
        raw_text: str,
        *,
        settings: PluginSettings | None = None,
    ) -> tuple[list[str], bool]:
        """Sanitize and segment raw text. Raises :class:`TtsTextError` if empty."""
        effective = settings or self.settings
        cleaned = sanitize_text(
            raw_text,
            clean_markdown=effective.clean_markdown,
            clean_url=effective.clean_url,
            clean_emoji=effective.clean_emoji,
        )
        if not cleaned:
            raise TtsTextError(
                "empty text after sanitizing",
                hint=("没有可合成的文本。请把内容写在唤醒词后面，例如「小助手 今天天气不错」。"),
            )
        return split_for_tts(
            cleaned,
            max_chars=effective.max_chars,
            max_segments=effective.max_segments,
            split_long_text=effective.split_long_text,
        )

    async def synthesize(
        self,
        raw_text: str,
        *,
        settings: PluginSettings | None = None,
        voice_override: str = "",
        model_override: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> SynthesisOutcome:
        """Synthesize ``raw_text`` and return the stored audio files.

        ``settings`` lets a caller supply a per-request variant (a wake-word
        profile, a session voice) without mutating the shared service state, so
        concurrent messages can never leak settings into each other. ``history``
        feeds the director's context-awareness.
        """
        effective = settings or self.settings
        if voice_override:
            effective = effective.with_voice(voice_override)
        model = model_override or effective.model
        effective.validate_for_synthesis()

        if self.store is None:
            raise TtsError("audio store is not configured", hint="插件内部错误：音频存储未初始化。")

        # I. Optional LLM stage: turn the chat message into a spoken script
        director = await self.direct(raw_text, settings=effective, history=history)

        # II. The director's delivery instruction wins when allowed, otherwise
        #     the configured instruction is used unchanged.
        instructions = effective.instructions
        if effective.llm_apply_instructions and director.instructions:
            instructions = director.instructions

        # III. Sanitize and segment whatever the director produced
        segments, truncated = self.prepare_segments(director.text, settings=effective)
        if not segments:
            raise TtsTextError("no segments", hint="没有可合成的文本。")

        outcome = SynthesisOutcome(
            segments=segments,
            model=model,
            voice=effective.voice,
            truncated=truncated,
            director=director,
        )

        semaphore = self._get_semaphore()
        for index, segment in enumerate(segments):
            async with semaphore:
                self._log(
                    "debug",
                    "Synthesizing segment %s/%s (%s chars) with %s/%s",
                    index + 1,
                    len(segments),
                    len(segment),
                    model,
                    effective.voice,
                )
                result = await self.client.synthesize(
                    text=segment,
                    model=model,
                    voice=effective.voice,
                    language_type=effective.language_type,
                    instructions=instructions,
                    optimize_instructions=effective.optimize_instructions,
                )
            path = await self.store.save(result.audio, prefix="tts")
            outcome.paths.append(path)
            outcome.total_bytes += len(result.audio)
            if result.request_id:
                outcome.request_ids.append(result.request_id)

        return outcome
