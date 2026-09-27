"""Opt-in tests that call the real DashScope Qwen-TTS API.

These are skipped unless ``DASHSCOPE_API_KEY`` is present, so the default test
run stays offline and deterministic. They exist to verify three things a mock
cannot: the configured model name really exists, the response really is a
playable WAV file, and the whole plugin pipeline works against production.

Usage::

    DASHSCOPE_API_KEY=sk-... pytest tests/test_live_dashscope.py -v -s

Optional environment variables:

``DASHSCOPE_REGION``      ``beijing`` (default) or ``singapore``
``DASHSCOPE_TTS_MODEL``   defaults to ``qwen-audio-3.1-tts-flash``
``DASHSCOPE_TTS_VOICE``   defaults to ``Cherry``

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import os
import struct
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from core.config import DEFAULT_MODEL, DEFAULT_VOICE, PluginSettings, normalize_base_url
from core.dashscope_client import DashScopeTtsClient
from core.wav import normalize_wav_sizes, wav_duration_seconds
from tests.conftest import records_of

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
REGION = os.environ.get("DASHSCOPE_REGION", "beijing").lower()
MODEL = os.environ.get("DASHSCOPE_TTS_MODEL", DEFAULT_MODEL)
VOICE = os.environ.get("DASHSCOPE_TTS_VOICE", DEFAULT_VOICE)

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="DASHSCOPE_API_KEY is not set; live API tests are opt-in",
)


def describe_wav(data: bytes) -> str:
    with wave.open(BytesIO(data), "rb") as handle:
        return (
            f"{handle.getnchannels()}ch {handle.getframerate()}Hz "
            f"{handle.getsampwidth() * 8}bit {handle.getnframes()} frames "
            f"({wav_duration_seconds(data):.2f}s)"
        )


async def test_live_model_produces_a_playable_wav() -> None:
    """The default model name must resolve and return real audio."""
    client = DashScopeTtsClient(
        base_url=normalize_base_url("", REGION),
        api_key=API_KEY,
        timeout=90.0,
        max_retries=1,
    )
    try:
        result = await client.synthesize(
            text="你好，这是一条语音合成测试。",
            model=MODEL,
            voice=VOICE,
            language_type="Chinese",
        )
    finally:
        await client.aclose()

    assert result.audio.startswith(b"RIFF"), "response was not a WAV container"
    assert len(result.audio) > 4096, f"audio suspiciously small: {len(result.audio)} bytes"
    assert result.source in {"url", "base64"}
    assert result.request_id, "no request id returned"

    # The raw reply carries a streaming placeholder header, so the duration is
    # only meaningful once normalized.
    normalized = normalize_wav_sizes(result.audio)
    duration = wav_duration_seconds(normalized)
    assert 0.2 < duration < 30.0, f"implausible duration {duration}s"

    print(
        f"\n[live] model={MODEL} voice={VOICE} region={REGION} "
        f"source={result.source} bytes={len(result.audio)}\n"
        f"[live] raw header claims {struct.unpack_from('<I', result.audio, 40)[0]} data bytes, "
        f"normalized -> {describe_wav(normalized)}",
    )


async def test_live_plugin_pipeline_writes_and_sends_audio(
    harness: Any,
    tmp_path: Path,
) -> None:
    """Wake word -> real API -> cached wav -> Record in the reply chain."""
    handle = harness(
        {
            "enabled": True,
            "wake_words": ["小助手"],
            "api_key": API_KEY,
            "region": REGION,
            "model": MODEL,
            "voice": VOICE,
            "language_type": "Chinese",
        },
    )
    await handle.plugin.initialize()

    event = handle.new_event("小助手 你好，插件联调测试成功。")
    results = await handle.drive(handle.plugin.on_message, event)

    records = records_of(results)
    assert records, "the plugin did not send any voice message"
    paths = [Path(record.path) for record in records]
    for path in paths:
        assert path.exists()
        assert path.parent == handle.audio_dir
        stored = path.read_bytes()
        assert stored.startswith(b"RIFF")
        # The store must have fixed the placeholder header before handing the
        # file to AstrBot, otherwise players see a multi-hour duration.
        assert struct.unpack_from("<I", stored, 40)[0] == len(stored) - 44
        assert 0.2 < wav_duration_seconds(stored) < 30.0
    print(f"\n[live] wrote {[p.name for p in paths]} ({describe_wav(paths[0].read_bytes())})")
    assert event.is_stopped() is True


async def test_live_status_command_reports_the_model(
    harness: Any,
) -> None:
    """``/tts_status`` must render without touching the network."""
    handle = harness(
        {
            "api_key": API_KEY,
            "region": REGION,
            "model": MODEL,
            "voice": VOICE,
        },
    )
    await handle.plugin.initialize()

    results = await handle.drive(
        handle.plugin.tts_status_command,
        handle.new_event("tts_status"),
    )
    output = "\n".join(
        str(getattr(component, "text", ""))
        for result in results
        for component in getattr(result, "chain", [])
    )

    assert MODEL in output
    assert API_KEY not in output


# --------------------------------------------------------------------------
# Live verification of the LLM director stage.
#
# The key authorised for these tests also drives DashScope's OpenAI-compatible
# endpoint, so the director can be verified against a real model instead of a
# mock. Override with LIVE_LLM_MODEL / LIVE_LLM_BASE_URL if you prefer another
# OpenAI-compatible provider (DeepSeek, Opencode, ...).
# --------------------------------------------------------------------------

LIVE_LLM_MODEL = os.environ.get("LIVE_LLM_MODEL", "qwen-plus")
LIVE_LLM_BASE_URL = os.environ.get(
    "LIVE_LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


async def test_live_director_call_follows_the_json_contract() -> None:
    """A real model must return a usable script, not a chat answer."""
    from core.config import TAG_STYLE_NONE
    from core.director import TtsCapabilities, build_messages, parse_director_response
    from core.llm_client import OpenAiCompatibleLlmClient

    settings = PluginSettings.from_dict(
        {
            "api_key": API_KEY,
            "model": MODEL,
            "voice": VOICE,
            "llm_tag_style": TAG_STYLE_NONE,
        },
    )
    capabilities = TtsCapabilities.from_settings(settings)
    messages = build_messages(
        text="小助手 温柔一点念：宝宝该睡觉了",
        capabilities=capabilities,
        history=[
            {"role": "user", "content": "已经很晚了"},
            {"role": "assistant", "content": "该休息啦"},
        ],
    )

    client = OpenAiCompatibleLlmClient(
        base_url=LIVE_LLM_BASE_URL,
        api_key=API_KEY,
        model=LIVE_LLM_MODEL,
        timeout=60.0,
        max_retries=1,
        temperature=0.3,
    )
    try:
        reply = await client.chat(messages)
    finally:
        await client.aclose()

    assert reply.text, "the model returned an empty completion"
    result = parse_director_response(
        reply.text,
        original_text="温柔一点念：宝宝该睡觉了",
        capabilities=capabilities,
        model=reply.model,
    )

    assert result.text.strip(), "the director produced no speakable text"
    assert result.text != "温柔一点念：宝宝该睡觉了", "the director did not touch the text"
    assert "<|" not in result.text, "tags leaked into a model that cannot read them"
    assert 0 < len(result.text) < 400, f"implausible script length: {len(result.text)}"
    assert result.model == LIVE_LLM_MODEL
    print(
        f"\n[live] director model={LIVE_LLM_MODEL} raw={reply.text[:160]!r}\n"
        f"[live] script={result.text!r} instructions={result.instructions!r}",
    )


async def test_live_pipeline_with_llm_and_tts(harness: Any) -> None:
    """Full chain: wake word -> real LLM director -> real TTS -> audio file."""
    handle = harness(
        {
            "enabled": True,
            "wake_words": ["小助手"],
            "api_key": API_KEY,
            "region": REGION,
            "model": MODEL,
            "voice": VOICE,
            "language_type": "Chinese",
            "llm_enable": True,
            "llm_mode": "direct",
            "llm_base_url": LIVE_LLM_BASE_URL,
            "llm_api_key": API_KEY,
            "llm_model": LIVE_LLM_MODEL,
            "llm_timeout_seconds": 60,
            "llm_fail_open": False,
        },
    )
    await handle.plugin.initialize()

    event = handle.new_event("小助手 用温柔一点的语气念：宝宝该睡觉了")
    results = await handle.drive(handle.plugin.on_message, event)

    records = records_of(results)
    assert records, "the plugin did not send any voice message"
    path = Path(records[0].path)
    stored = path.read_bytes()
    assert stored.startswith(b"RIFF")
    assert 0.2 < wav_duration_seconds(stored) < 60.0
    print(
        f"\n[live] llm+tts pipeline wrote {path.name} ({describe_wav(stored)})",
    )
