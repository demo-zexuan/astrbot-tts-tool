"""Tests for the DashScope Qwen-TTS HTTP client.

Every test drives the real ``aiohttp`` transport against the fake endpoint in
``tests/mock_dashscope.py``, so request building, retry handling and error
mapping are covered end to end.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import base64

import pytest

from core.dashscope_client import DashScopeTtsClient
from core.errors import TtsApiError, TtsConfigError, TtsNetworkError
from tests import mock_dashscope


def build_client(server: mock_dashscope.MockDashScope, **overrides: object) -> DashScopeTtsClient:
    kwargs: dict[str, object] = {
        "base_url": server.base_url,
        "api_key": "sk-test",
        "timeout": 10.0,
        "max_retries": 2,
        "backoff": 0.01,
    }
    kwargs.update(overrides)
    return DashScopeTtsClient(**kwargs)  # type: ignore[arg-type]


class TestRequest:
    async def test_request_body_matches_the_documented_shape(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        client = build_client(dashscope)
        try:
            result = await client.synthesize(
                text="你好呀",
                model="qwen-audio-3.1-tts-flash",
                voice="Cherry",
                language_type="Chinese",
            )
        finally:
            await client.aclose()

        assert result.audio.startswith(b"RIFF")
        assert result.source == "url"
        assert result.request_id == "req-1"

        sent = dashscope.state.requests[0]
        assert sent == {
            "model": "qwen-audio-3.1-tts-flash",
            "input": {"text": "你好呀", "voice": "Cherry", "language_type": "Chinese"},
        }

    async def test_bearer_token_is_sent(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        client = build_client(dashscope, api_key="sk-secret")
        try:
            await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()
        assert dashscope.state.headers[0]["Authorization"] == "Bearer sk-secret"
        assert dashscope.state.headers[0]["Content-Type"] == "application/json"

    async def test_instructions_are_only_sent_when_present(self) -> None:
        without = DashScopeTtsClient.build_payload(text="t", model="m", voice="Cherry")
        assert "instructions" not in without["input"]

        with_instructions = DashScopeTtsClient.build_payload(
            text="t",
            model="m",
            voice="Cherry",
            instructions="语速快一点",
            optimize_instructions=True,
        )
        assert with_instructions["input"]["instructions"] == "语速快一点"
        assert with_instructions["input"]["optimize_instructions"] is True

    async def test_optimize_is_dropped_without_instructions(self) -> None:
        payload = DashScopeTtsClient.build_payload(
            text="t",
            model="m",
            voice="Cherry",
            optimize_instructions=True,
        )
        assert "optimize_instructions" not in payload["input"]

    async def test_synthesis_url_is_derived_from_base_url(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        client = build_client(dashscope)
        try:
            assert client.synthesis_url == dashscope.synthesis_url
        finally:
            await client.aclose()


class TestResponseHandling:
    async def test_inline_base64_is_used_when_no_url_is_returned(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.mode = "base64"
        client = build_client(dashscope)
        try:
            result = await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()

        assert result.source == "base64"
        assert result.audio == dashscope.state.audio
        assert result.audio.startswith(b"RIFF")

    async def test_a_failing_download_falls_back_to_inline_audio(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.mode = "base64"
        dashscope.state.download_status = 404
        # Force the URL branch by hand so the fallback is actually exercised.
        client = build_client(dashscope)
        original_extract = client._extract_audio

        async def with_url_first(data: dict) -> object:
            output = data.setdefault("output", {})
            audio = output.setdefault("audio", {})
            audio["url"] = f"{dashscope.base_url}/audio/missing.wav"
            return await original_extract(data)

        client._extract_audio = with_url_first  # type: ignore[method-assign]
        try:
            result = await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()

        assert result.source == "base64"
        assert result.audio == dashscope.state.audio

    async def test_missing_audio_is_reported(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.error_payload = {
            "status_code": 200,
            "request_id": "req-x",
            "code": "",
            "message": "",
            "output": {"audio": {"url": "", "data": ""}},
        }
        client = build_client(dashscope)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()
        assert "空音频" in excinfo.value.user_message()

    async def test_non_json_body_is_reported(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.raw_body = "<html>gateway error</html>"
        client = build_client(dashscope)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()
        assert "无法解析" in excinfo.value.user_message()


class TestErrorsAndRetries:
    async def test_rate_limit_is_retried_and_then_succeeds(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.rate_limit_times = 2
        client = build_client(dashscope, max_retries=3)
        try:
            result = await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()

        assert result.audio.startswith(b"RIFF")
        assert len(dashscope.state.requests) == 3

    async def test_rate_limit_gives_up_after_the_configured_attempts(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.rate_limit_times = 10
        client = build_client(dashscope, max_retries=1)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()

        assert excinfo.value.status_code == 429
        assert len(dashscope.state.requests) == 2

    async def test_invalid_api_key_is_not_retried(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.synthesis_status = 401
        client = build_client(dashscope, max_retries=3)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()

        assert len(dashscope.state.requests) == 1
        assert "鉴权失败" in excinfo.value.user_message()

    async def test_model_error_hint_mentions_the_model(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.error_payload = {
            "status_code": 400,
            "code": "Model.Invalid",
            "message": "model not found",
            "request_id": "req-err",
        }
        client = build_client(dashscope)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(text="hi", model="nope", voice="Cherry")
        finally:
            await client.aclose()
        assert (
            "模型不可用" in excinfo.value.user_message() or "不可用" in excinfo.value.user_message()
        )
        # The hint must name a model that actually works on this route.
        assert "qwen3-tts-flash" in excinfo.value.user_message()

    async def test_a_model_that_is_not_served_here_explains_the_route_problem(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        # Verified against the real API: an HTTP call to a WebSocket-only model
        # (for example qwen-audio-3.1-tts-flash) answers
        # {"code": "InvalidParameter", "message": "url error, please check url！"}.
        dashscope.state.error_payload = {
            "status_code": 400,
            "code": "InvalidParameter",
            "message": "url error, please check url！",
            "request_id": "req-url",
        }
        client = build_client(dashscope)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(
                    text="hi",
                    model="qwen-audio-3.1-tts-flash",
                    voice="Cherry",
                )
        finally:
            await client.aclose()

        hint = excinfo.value.user_message()
        assert "接口地址上不可用" in hint
        assert "qwen3-tts-flash" in hint
        assert "qwen-audio-3.1-tts-flash" in hint

    async def test_model_not_exist_uses_the_same_route_hint(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.error_payload = {
            "status_code": 400,
            "code": "InvalidParameter",
            "message": "Model not exist.",
        }
        client = build_client(dashscope)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(text="hi", model="nope", voice="Cherry")
        finally:
            await client.aclose()
        assert "Model not exist." in excinfo.value.user_message()

    async def test_instructions_rejection_names_the_instruct_family(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.error_payload = {
            "status_code": 400,
            "code": "InvalidParameter",
            "message": "instructions is not supported by this model",
        }
        client = build_client(dashscope)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(
                    text="hi",
                    model="qwen3-tts-flash",
                    voice="Cherry",
                    instructions="慢一点",
                )
        finally:
            await client.aclose()
        assert "qwen3-tts-instruct-flash" in excinfo.value.user_message()

    async def test_voice_error_hint_mentions_the_voice(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.error_payload = {
            "status_code": 400,
            "code": "BadRequest",
            "message": "voice is not supported",
        }
        client = build_client(dashscope)
        try:
            with pytest.raises(TtsApiError) as excinfo:
                await client.synthesize(text="hi", model="m", voice="Nope")
        finally:
            await client.aclose()
        assert "音色" in excinfo.value.user_message()

    async def test_transport_failure_is_wrapped_as_a_network_error(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        # Point at a port that is guaranteed to be closed.
        client = DashScopeTtsClient(
            base_url="http://127.0.0.1:1/api/v1",
            api_key="sk-test",
            timeout=2.0,
            max_retries=1,
            backoff=0.01,
        )
        try:
            with pytest.raises(TtsNetworkError):
                await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()

    async def test_missing_api_key_fails_fast(self) -> None:
        client = DashScopeTtsClient(base_url="http://127.0.0.1:1/api/v1", api_key="")
        try:
            with pytest.raises(TtsConfigError):
                await client.synthesize(text="hi", model="m", voice="Cherry")
        finally:
            await client.aclose()

    async def test_blank_text_fails_fast(self) -> None:
        client = DashScopeTtsClient(base_url="http://127.0.0.1:1/api/v1", api_key="sk-x")
        try:
            with pytest.raises(TtsConfigError):
                await client.synthesize(text="   ", model="m", voice="Cherry")
        finally:
            await client.aclose()


class TestSessionLifecycle:
    async def test_an_injected_session_is_not_closed(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            client = DashScopeTtsClient(
                base_url=dashscope.base_url,
                api_key="sk-test",
                timeout=10.0,
                backoff=0.01,
                session=session,
            )
            result = await client.synthesize(text="hi", model="m", voice="Cherry")
            await client.aclose()
            assert not session.closed
            assert result.audio.startswith(b"RIFF")

    async def test_owned_session_is_reused_across_calls(
        self,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        client = build_client(dashscope)
        try:
            first = await client.synthesize(text="a", model="m", voice="Cherry")
            session = client._session
            second = await client.synthesize(text="b", model="m", voice="Cherry")
            assert client._session is session
        finally:
            await client.aclose()

        assert first.audio and second.audio
        assert len(dashscope.state.requests) == 2


class TestBase64Decoding:
    def test_data_uri_prefix_is_stripped(self) -> None:
        from core.dashscope_client import _decode_base64

        payload = base64.b64encode(b"abc").decode()
        assert _decode_base64(f"data:audio/wav;base64,{payload}") == b"abc"
        assert _decode_base64(f"base64://{payload}") == b"abc"

    def test_invalid_base64_raises_a_friendly_error(self) -> None:
        from core.dashscope_client import _decode_base64

        with pytest.raises(TtsApiError):
            _decode_base64("!!!! not base64 at all !!!!")
