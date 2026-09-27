"""Tests for the OpenAI-compatible LLM client used by direct mode.

Every test drives the real ``aiohttp`` transport against the fake endpoint in
``tests/mock_llm.py``, so the request body and — most importantly — the request
headers are verified as they go over the wire.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import pytest

from core.errors import LlmError
from core.llm_client import (
    AUTH_STYLE_BEARER,
    AUTH_STYLE_NONE,
    AUTH_STYLE_X_API_KEY,
    OpenAiCompatibleLlmClient,
    normalize_chat_url,
)
from tests import mock_llm

MESSAGES = [
    {"role": "system", "content": "你是导演"},
    {"role": "user", "content": "你好"},
]


def build_client(server: mock_llm.MockLlm, **overrides: object) -> OpenAiCompatibleLlmClient:
    kwargs: dict[str, object] = {
        "base_url": server.base_url,
        "api_key": "sk-test",
        "model": "deepseek-chat",
        "timeout": 10.0,
        "max_retries": 1,
    }
    kwargs.update(overrides)
    return OpenAiCompatibleLlmClient(**kwargs)  # type: ignore[arg-type]


class TestUrlNormalization:
    @pytest.mark.parametrize(
        ("base", "expected_suffix"),
        [
            ("https://api.deepseek.com/v1", "/v1/chat/completions"),
            ("https://api.deepseek.com/v1/", "/v1/chat/completions"),
            ("https://api.deepseek.com", "/chat/completions"),
            (
                "https://api.deepseek.com/v1/chat/completions",
                "/v1/chat/completions",
            ),
            ("https://gateway.example.com/openai/v1", "/openai/v1/chat/completions"),
        ],
    )
    def test_normalization(self, base: str, expected_suffix: str) -> None:
        assert normalize_chat_url(base).endswith(expected_suffix)

    def test_empty_base_url_is_rejected(self) -> None:
        with pytest.raises(LlmError) as excinfo:
            normalize_chat_url("   ")
        assert "Base URL" in excinfo.value.user_message()

    async def test_a_bare_host_still_reaches_the_endpoint(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        client = build_client(llm_server, base_url=llm_server.bare_base_url)
        try:
            result = await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert result.text


class TestHeaders:
    async def test_bearer_style_puts_the_key_in_authorization(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        client = build_client(llm_server, auth_style=AUTH_STYLE_BEARER, api_key="sk-abc")
        try:
            await client.chat(MESSAGES)
        finally:
            await client.aclose()

        headers = llm_server.state.headers[0]
        assert headers["Authorization"] == "Bearer sk-abc"
        assert headers["Content-Type"] == "application/json"
        assert "x-api-key" not in headers

    async def test_x_api_key_style(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        client = build_client(llm_server, auth_style=AUTH_STYLE_X_API_KEY, api_key="sk-abc")
        try:
            await client.chat(MESSAGES)
        finally:
            await client.aclose()

        headers = llm_server.state.headers[0]
        assert headers["x-api-key"] == "sk-abc"
        assert "Authorization" not in headers

    async def test_none_style_sends_no_credentials(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        client = build_client(llm_server, auth_style=AUTH_STYLE_NONE, api_key="")
        try:
            await client.chat(MESSAGES)
        finally:
            await client.aclose()

        headers = llm_server.state.headers[0]
        assert "Authorization" not in headers
        assert "x-api-key" not in headers

    async def test_extra_headers_reach_the_gateway(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        # This is the Opencode-style case: a gateway that needs its own headers.
        client = build_client(
            llm_server,
            extra_headers={
                "anthropic-version": "2023-06-01",
                "X-Title": "astrbot-tts-wakeword",
                "HTTP-Referer": "https://github.com/demo-zexuan/astrbot-tts-tool",
            },
        )
        try:
            await client.chat(MESSAGES)
        finally:
            await client.aclose()

        headers = llm_server.state.headers[0]
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["X-Title"] == "astrbot-tts-wakeword"
        assert headers["HTTP-Referer"] == "https://github.com/demo-zexuan/astrbot-tts-tool"

    async def test_extra_headers_override_the_generated_auth_header(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        client = build_client(
            llm_server,
            api_key="sk-generated",
            extra_headers={"Authorization": "Token sk-manual"},
        )
        try:
            await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert llm_server.state.headers[0]["Authorization"] == "Token sk-manual"

    async def test_header_values_are_stringified(self) -> None:
        client = OpenAiCompatibleLlmClient(
            base_url="https://example.com/v1",
            api_key="sk-x",
            extra_headers={"X-Retry": 3, "X-Flag": True},  # type: ignore[dict-item]
        )
        headers = client.build_headers()
        assert headers["X-Retry"] == "3"
        assert headers["X-Flag"] == "True"

    async def test_no_api_key_means_no_auth_header_even_in_bearer_mode(self) -> None:
        client = OpenAiCompatibleLlmClient(
            base_url="https://example.com/v1",
            api_key="",
            auth_style=AUTH_STYLE_BEARER,
        )
        assert "Authorization" not in client.build_headers()


class TestRequestAndResponse:
    async def test_payload_shape(self, llm_server: mock_llm.MockLlm) -> None:
        client = build_client(
            llm_server, model="deepseek-v4-flash", max_tokens=256, temperature=0.2
        )
        try:
            await client.chat(MESSAGES)
        finally:
            await client.aclose()

        sent = llm_server.state.requests[0]
        assert sent["model"] == "deepseek-v4-flash"
        assert sent["messages"] == MESSAGES
        assert sent["max_tokens"] == 256
        assert sent["temperature"] == 0.2
        assert sent["stream"] is False

    async def test_temperature_can_be_disabled(self, llm_server: mock_llm.MockLlm) -> None:
        client = build_client(llm_server, temperature=None)
        try:
            await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert "temperature" not in llm_server.state.requests[0]

    async def test_model_can_be_omitted(self, llm_server: mock_llm.MockLlm) -> None:
        client = build_client(llm_server, model="")
        try:
            await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert "model" not in llm_server.state.requests[0]

    async def test_response_text_and_metadata(self, llm_server: mock_llm.MockLlm) -> None:
        llm_server.state.reply = '{"text": "你好呀"}'
        client = build_client(llm_server)
        try:
            result = await client.chat(MESSAGES)
        finally:
            await client.aclose()

        assert result.text == '{"text": "你好呀"}'
        assert result.model == "fake-model"
        assert result.finish_reason == "stop"
        assert result.usage["total_tokens"] == 30

    async def test_content_parts_are_joined(self, llm_server: mock_llm.MockLlm) -> None:
        # Some gateways answer with content parts instead of a plain string.
        llm_server.state.raw_body = (
            '{"choices": [{"message": {"role": "assistant", '
            '"content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}, '
            '"finish_reason": "stop"}]}'
        )
        client = build_client(llm_server)
        try:
            result = await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert result.text == "part1part2"

    async def test_missing_choices_is_an_error(self, llm_server: mock_llm.MockLlm) -> None:
        llm_server.state.raw_body = '{"id": "x", "choices": []}'
        client = build_client(llm_server)
        try:
            with pytest.raises(LlmError) as excinfo:
                await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert "没有返回任何内容" in excinfo.value.user_message()

    async def test_non_json_body_is_an_error(self, llm_server: mock_llm.MockLlm) -> None:
        llm_server.state.raw_body = "<html>502 Bad Gateway</html>"
        client = build_client(llm_server)
        try:
            with pytest.raises(LlmError) as excinfo:
                await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert "无法解析" in excinfo.value.user_message()

    async def test_empty_messages_is_rejected(self, llm_server: mock_llm.MockLlm) -> None:
        client = build_client(llm_server)
        try:
            with pytest.raises(LlmError):
                await client.chat([])
        finally:
            await client.aclose()


class TestErrorsAndRetries:
    async def test_rate_limit_is_retried_then_succeeds(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        llm_server.state.rate_limit_times = 1
        client = build_client(llm_server, max_retries=2)
        try:
            result = await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert result.text
        assert len(llm_server.state.requests) == 2

    async def test_rate_limit_gives_up(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        llm_server.state.rate_limit_times = 5
        client = build_client(llm_server, max_retries=0)
        try:
            with pytest.raises(LlmError) as excinfo:
                await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert len(llm_server.state.requests) == 1
        assert "429" in excinfo.value.hint or "限流" in excinfo.value.hint

    async def test_authentication_failure_mentions_headers(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        llm_server.state.error_status = 401
        client = build_client(llm_server, max_retries=2)
        try:
            with pytest.raises(LlmError) as excinfo:
                await client.chat(MESSAGES)
        finally:
            await client.aclose()

        assert len(llm_server.state.requests) == 1
        hint = excinfo.value.user_message()
        assert "鉴权失败" in hint
        # The hint must point at the header knobs, which is the actual fix.
        assert "x-api-key" in hint and "anthropic-version" in hint

    async def test_not_found_explains_the_path(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        llm_server.state.error_status = 404
        llm_server.state.error_body = {"error": {"message": "not found"}}
        client = build_client(llm_server)
        try:
            with pytest.raises(LlmError) as excinfo:
                await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert "Base URL" in excinfo.value.user_message()

    async def test_transport_failure_is_wrapped(self) -> None:
        client = OpenAiCompatibleLlmClient(
            base_url="http://127.0.0.1:1/v1",
            api_key="sk-test",
            timeout=2.0,
            max_retries=0,
        )
        try:
            with pytest.raises(LlmError) as excinfo:
                await client.chat(MESSAGES)
        finally:
            await client.aclose()
        assert "无法连接" in excinfo.value.user_message()


class TestSessionLifecycle:
    async def test_an_injected_session_is_not_closed(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            client = OpenAiCompatibleLlmClient(
                base_url=llm_server.base_url,
                api_key="sk-test",
                session=session,
            )
            await client.chat(MESSAGES)
            await client.aclose()
            assert not session.closed

    async def test_the_owned_session_is_reused(
        self,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        client = build_client(llm_server)
        try:
            await client.chat(MESSAGES)
            session = client._session
            await client.chat(MESSAGES)
            assert client._session is session
        finally:
            await client.aclose()
        assert len(llm_server.state.requests) == 2
