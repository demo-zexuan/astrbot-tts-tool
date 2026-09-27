"""Tests for the synthesis orchestration service.

The HTTP client is replaced by a recording fake, which keeps these tests focused
on segmentation, storage and error propagation.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from core.audio_store import AudioStore
from core.config import PluginSettings
from core.dashscope_client import SynthesisResult
from core.errors import TtsTextError
from core.service import SynthesisOutcome, TtsService
from tests import mock_dashscope

#: Three 13-character sentences. ``max_chars`` is clamped to 20 by the config
#: parser, so the body must be long enough for splitting to be observable.
LONG_TEXT = "这是一句比较长的测试句子。" * 3


@dataclass
class FakeClient:
    """Records every call and returns a canned WAV payload."""

    audio: bytes = field(default_factory=mock_dashscope.make_wav)
    fail_on: set[int] = field(default_factory=set)
    calls: list[dict[str, object]] = field(default_factory=list)
    delay: float = 0.0
    concurrent: int = 0
    max_concurrent_seen: int = 0
    closed: bool = False

    async def synthesize(self, **kwargs: object) -> SynthesisResult:
        index = len(self.calls)
        self.calls.append(kwargs)
        if self.delay:
            self.concurrent += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self.concurrent)
            try:
                await asyncio.sleep(self.delay)
            finally:
                self.concurrent -= 1
        if index in self.fail_on:
            from core.errors import TtsApiError

            raise TtsApiError("boom", hint="合成失败", status_code=500)
        return SynthesisResult(
            audio=self.audio,
            request_id=f"req-{index}",
            source="url",
        )

    async def aclose(self) -> None:
        self.closed = True


def build_service(
    tmp_path: Path,
    client: FakeClient,
    **config: object,
) -> TtsService:
    settings = PluginSettings.from_dict(
        {"api_key": "sk-test", "model": "test-model", "voice": "Cherry", **config},
    )
    return TtsService(
        settings,
        client=client,  # type: ignore[arg-type]
        store=AudioStore(tmp_path / "audio"),
    )


class TestPrepareSegments:
    def test_markdown_is_cleaned_before_synthesis(self, tmp_path: Path) -> None:
        service = build_service(tmp_path, FakeClient())
        segments, _ = service.prepare_segments("**你好** https://example.com")
        assert segments == ["你好"]

    def test_empty_text_raises_a_friendly_error(self, tmp_path: Path) -> None:
        service = build_service(tmp_path, FakeClient())
        with pytest.raises(TtsTextError) as excinfo:
            service.prepare_segments("   \n  ")
        assert "没有可合成的文本" in excinfo.value.user_message()

    def test_long_text_is_split(self, tmp_path: Path) -> None:
        service = build_service(tmp_path, FakeClient(), max_chars=20, max_segments=4)
        segments, _truncated = service.prepare_segments("句子。" * 12)
        assert len(segments) > 1
        assert all(len(segment) <= 20 for segment in segments)


class TestSynthesize:
    async def test_single_segment_produces_one_file(self, tmp_path: Path) -> None:
        client = FakeClient()
        service = build_service(tmp_path, client)
        outcome = await service.synthesize("你好呀")

        assert isinstance(outcome, SynthesisOutcome)
        assert outcome.ok
        assert len(outcome.paths) == 1
        assert outcome.paths[0].read_bytes() == client.audio
        assert outcome.request_ids == ["req-0"]
        assert outcome.total_bytes == len(client.audio)
        assert outcome.model == "test-model"
        assert outcome.voice == "Cherry"

    async def test_each_segment_is_requested_separately(self, tmp_path: Path) -> None:
        client = FakeClient()
        service = build_service(tmp_path, client, max_chars=20, max_segments=5)
        outcome = await service.synthesize(LONG_TEXT)

        assert len(outcome.paths) == len(client.calls)
        assert len(client.calls) > 1
        for call in client.calls:
            assert call["model"] == "test-model"
            assert call["voice"] == "Cherry"

    async def test_voice_override_is_used(self, tmp_path: Path) -> None:
        client = FakeClient()
        service = build_service(tmp_path, client)
        outcome = await service.synthesize("你好", voice_override="Ethan")

        assert outcome.voice == "Ethan"
        assert client.calls[0]["voice"] == "Ethan"

    async def test_model_override_is_used(self, tmp_path: Path) -> None:
        client = FakeClient()
        service = build_service(tmp_path, client)
        outcome = await service.synthesize("你好", model_override="qwen-tts")
        assert outcome.model == "qwen-tts"
        assert client.calls[0]["model"] == "qwen-tts"

    async def test_missing_api_key_is_reported_before_any_request(self, tmp_path: Path) -> None:
        client = FakeClient()
        settings = PluginSettings.from_dict({"model": "m", "voice": "Cherry"})
        service = TtsService(
            settings,
            client=client,  # type: ignore[arg-type]
            store=AudioStore(tmp_path / "audio"),
        )
        from core.errors import TtsConfigError

        with pytest.raises(TtsConfigError):
            await service.synthesize("你好")
        assert client.calls == []

    async def test_a_failing_segment_aborts_the_request(self, tmp_path: Path) -> None:
        client = FakeClient(fail_on={1})
        service = build_service(tmp_path, client, max_chars=20, max_segments=5)
        from core.errors import TtsApiError

        with pytest.raises(TtsApiError):
            await service.synthesize(LONG_TEXT)
        assert len(client.calls) == 2

    async def test_truncation_flag_is_propagated(self, tmp_path: Path) -> None:
        client = FakeClient()
        service = build_service(tmp_path, client, max_chars=20, max_segments=1)
        outcome = await service.synthesize(LONG_TEXT)
        assert outcome.truncated is True

    async def test_settings_can_be_swapped_between_calls(self, tmp_path: Path) -> None:
        client = FakeClient()
        service = build_service(tmp_path, client)
        service.update_settings(
            PluginSettings.from_dict({"api_key": "sk-test", "model": "second", "voice": "Ethan"}),
        )
        outcome = await service.synthesize("你好")
        assert outcome.model == "second"
        assert outcome.voice == "Ethan"


class TestConcurrency:
    async def test_segments_within_one_message_are_sequential(self, tmp_path: Path) -> None:
        # Segments of a single reply must stay ordered, so they never overlap
        # even when the concurrency budget would allow it.
        client = FakeClient(delay=0.02)
        service = build_service(
            tmp_path,
            client,
            max_chars=20,
            max_segments=8,
            max_concurrent=4,
        )
        outcome = await service.synthesize(LONG_TEXT)
        assert len(outcome.paths) > 1
        assert client.max_concurrent_seen == 1

    async def test_the_semaphore_bounds_parallel_messages(self, tmp_path: Path) -> None:
        client = FakeClient(delay=0.05)
        service = build_service(tmp_path, client, max_concurrent=1)

        await asyncio.gather(
            service.synthesize("第一条消息"),
            service.synthesize("第二条消息"),
            service.synthesize("第三条消息"),
        )

        assert client.max_concurrent_seen == 1

    async def test_parallel_messages_can_overlap_when_allowed(self, tmp_path: Path) -> None:
        client = FakeClient(delay=0.05)
        service = build_service(tmp_path, client, max_concurrent=3)

        await asyncio.gather(
            service.synthesize("第一条消息"),
            service.synthesize("第二条消息"),
            service.synthesize("第三条消息"),
        )

        assert client.max_concurrent_seen > 1


class TestClientLifecycle:
    async def test_the_client_is_lazily_created_from_settings(self, tmp_path: Path) -> None:
        service = build_service(tmp_path, FakeClient())
        # An injected client is kept as-is.
        assert isinstance(service.client, FakeClient)

    async def test_a_new_client_replaces_the_injected_one_when_transport_changes(
        self,
        tmp_path: Path,
    ) -> None:
        service = build_service(tmp_path, FakeClient())
        service.update_settings(
            PluginSettings.from_dict(
                {"api_key": "sk-other", "model": "m", "voice": "Cherry"},
            ),
        )
        from core.dashscope_client import DashScopeTtsClient

        assert isinstance(service.client, DashScopeTtsClient)
        await service.aclose()

    async def test_aclose_is_safe_when_no_client_was_built(self, tmp_path: Path) -> None:
        settings = PluginSettings.from_dict({"api_key": "sk-test", "base_url": "http://x/api/v1"})
        service = TtsService(settings, store=AudioStore(tmp_path / "audio"))
        await service.aclose()
