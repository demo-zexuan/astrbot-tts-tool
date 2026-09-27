"""End-to-end tests for the LLM director stage inside the plugin pipeline.

These drive the real ``main.py`` handlers (through the AstrBot stub) against the
fake DashScope endpoint, so they cover the whole chain:

    wake word -> director (provider or direct) -> TTS HTTP -> audio -> Record

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

from typing import Any

import pytest

from tests import astrbot_stub, mock_dashscope, mock_llm
from tests.conftest import records_of, texts_of

SCRIPT = (
    '{"text": "晚安呀，做个好梦～", "instructions": "轻声细语，语速稍慢",'
    ' "tags": [], "reason": "加了语气词并放慢节奏"}'
)


def base_config(server: mock_dashscope.MockDashScope, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "enabled": True,
        "wake_words": ["小助手"],
        "api_key": "sk-tts",
        "model": "qwen3-tts-flash",
        "voice": "Cherry",
        "base_url": server.base_url,
        # The director is on by default in production; tests keep it explicit.
        "llm_enable": True,
        "llm_mode": "astrbot_provider",
    }
    config.update(overrides)
    return config


async def started(harness: Any, config: dict[str, Any]) -> Any:
    handle = harness(config)
    await handle.plugin.initialize()
    return handle


def use_provider(handle: Any, provider: astrbot_stub.FakeProvider, provider_id: str = "") -> None:
    """Expose ``provider`` to the plugin, optionally under a specific id."""
    if provider_id:
        handle.runtime.providers[provider_id] = provider
    else:
        handle.runtime.using_provider = provider


class TestProviderMode:
    async def test_the_rewritten_script_is_what_gets_synthesized(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)

        results = await handle.drive(
            handle.plugin.on_message,
            handle.new_event("小助手 念一句晚安"),
        )

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "晚安呀，做个好梦～"
        assert len(provider.calls) == 1

    async def test_the_provider_receives_system_prompt_and_prompt(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 念一句晚安"))

        call = provider.calls[0]
        assert "文本导演" in call["system_prompt"]
        assert "【需要朗读的原文】" in call["prompt"]
        assert "念一句晚安" in call["prompt"]
        assert "model" not in call  # empty llm_model means "provider default"

    async def test_a_configured_provider_id_is_used(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            base_config(dashscope, llm_provider_id="deepseek/deepseek-flash"),
        )
        chosen = astrbot_stub.FakeProvider(name="chosen", reply=SCRIPT)
        fallback = astrbot_stub.FakeProvider(name="fallback", reply='{"text": "错误来源"}')
        handle.runtime.providers["deepseek/deepseek-flash"] = chosen
        handle.runtime.using_provider = fallback

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        assert len(chosen.calls) == 1
        assert fallback.calls == []

    async def test_an_unknown_provider_id_falls_back_to_the_session_default(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_provider_id="missing/id"))
        fallback = astrbot_stub.FakeProvider(reply=SCRIPT)
        handle.runtime.using_provider = fallback

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        assert len(fallback.calls) == 1

    async def test_the_model_override_is_forwarded(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_model="deepseek-v4-pro"))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        assert provider.calls[0]["model"] == "deepseek-v4-pro"

    async def test_a_minimal_provider_signature_still_works(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        """A provider exposing only ``prompt`` must not break the stage."""
        handle = await started(harness, base_config(dashscope))

        class MinimalProvider:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def get_model(self) -> str:
                return "minimal"

            async def text_chat(self, prompt: str = "", **_: Any) -> Any:
                self.calls.append({"prompt": prompt})
                return astrbot_stub.FakeLlmResponse(SCRIPT)

        provider = MinimalProvider()
        handle.runtime.using_provider = provider

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 晚安"))

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "晚安呀，做个好梦～"
        assert len(provider.calls) == 1

    async def test_disabled_llm_never_calls_the_provider(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_enable=False))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results)
        assert provider.calls == []
        assert dashscope.state.requests[0]["input"]["text"] == "你好呀"


class TestContextAwareness:
    async def test_recent_history_is_passed_to_the_provider(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        import json

        handle = await started(harness, base_config(dashscope))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)
        handle.runtime.conversation_manager.history = json.dumps(
            [
                {"role": "user", "content": "宝宝要睡觉了"},
                {"role": "assistant", "content": "好的，我来讲个故事"},
                {"role": "system", "content": "这条应当被忽略"},
            ],
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 念一句晚安"))

        # History is embedded in the prompt as a transcript rather than passed as
        # chat turns: the director must rewrite, not continue the conversation.
        prompt = provider.calls[0]["prompt"]
        assert "宝宝要睡觉了" in prompt
        assert "好的，我来讲个故事" in prompt
        assert "这条应当被忽略" not in prompt

    async def test_the_history_limit_is_applied(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        import json

        handle = await started(harness, base_config(dashscope, llm_context_messages=2))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)
        handle.runtime.conversation_manager.history = json.dumps(
            [{"role": "user", "content": f"第{i}条"} for i in range(6)],
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        prompt = provider.calls[0]["prompt"]
        assert "第5条" in prompt
        assert "第4条" in prompt
        assert "第3条" not in prompt

    async def test_zero_context_disables_history(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        import json

        handle = await started(harness, base_config(dashscope, llm_context_messages=0))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)
        handle.runtime.conversation_manager.history = json.dumps(
            [{"role": "user", "content": "不该出现的历史内容"}],
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        prompt = provider.calls[0]["prompt"]
        assert "（无历史消息）" in prompt
        assert "不该出现的历史内容" not in prompt

    async def test_a_broken_history_never_breaks_the_reply(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)
        handle.runtime.conversation_manager.history = "这不是 JSON"

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        assert records_of(results)
        assert len(provider.calls) == 1

    async def test_a_missing_conversation_manager_is_tolerated(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)
        # Simulate a build without conversation support.
        del handle.plugin.context.conversation_manager

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        assert records_of(results)
        assert "（无历史消息）" in provider.calls[0]["prompt"]


class TestFailureHandling:
    async def test_no_provider_falls_back_to_the_original_text(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        # Default configuration in production: the director is on, but a fresh
        # install may have no chat provider configured yet.
        handle = await started(harness, base_config(dashscope))

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "你好呀"

    async def test_no_provider_with_fail_open_disabled_reports_an_error(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_fail_open=False))

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results) == []
        assert dashscope.state.requests == []
        assert any("Provider" in text for text in texts_of(results))

    async def test_a_provider_exception_falls_back(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        provider = astrbot_stub.FakeProvider()
        provider.raise_error = RuntimeError("upstream exploded")
        use_provider(handle, provider)

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "你好呀"

    async def test_a_provider_exception_with_fail_open_disabled_reports_an_error(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_fail_open=False))
        provider = astrbot_stub.FakeProvider()
        provider.raise_error = RuntimeError("upstream exploded")
        use_provider(handle, provider)

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results) == []
        assert any("调用大模型失败" in text for text in texts_of(results))

    async def test_an_unusable_reply_falls_back_to_the_original_text(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        use_provider(handle, astrbot_stub.FakeProvider(reply="   "))

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "你好呀"

    async def test_an_unusable_reply_with_fail_open_disabled_reports_an_error(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_fail_open=False))
        use_provider(handle, astrbot_stub.FakeProvider(reply=""))

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results) == []
        assert any("没有返回可用文本" in text for text in texts_of(results))


class TestInstructionsAndTags:
    async def test_director_instructions_are_sent_for_instruct_models(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            base_config(dashscope, model="qwen3-tts-instruct-flash"),
        )
        use_provider(handle, astrbot_stub.FakeProvider(reply=SCRIPT))

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 念一句晚安"))

        sent = dashscope.state.requests[0]["input"]
        assert sent["instructions"] == "轻声细语，语速稍慢"

    async def test_director_instructions_are_dropped_for_plain_models(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, model="qwen3-tts-flash"))
        use_provider(handle, astrbot_stub.FakeProvider(reply=SCRIPT))

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 念一句晚安"))

        assert "instructions" not in dashscope.state.requests[0]["input"]

    async def test_apply_instructions_can_be_disabled(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            base_config(
                dashscope,
                model="qwen3-tts-instruct-flash",
                instructions="固定语气",
                llm_apply_instructions=False,
            ),
        )
        use_provider(handle, astrbot_stub.FakeProvider(reply=SCRIPT))

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 念一句晚安"))

        assert dashscope.state.requests[0]["input"]["instructions"] == "固定语气"

    async def test_configured_instructions_are_used_when_the_director_has_none(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            base_config(
                dashscope,
                model="qwen3-tts-instruct-flash",
                instructions="固定语气",
            ),
        )
        use_provider(handle, astrbot_stub.FakeProvider(reply='{"text": "晚安"}'))

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 晚安"))

        assert dashscope.state.requests[0]["input"]["instructions"] == "固定语气"

    async def test_tags_are_stripped_before_synthesis_by_default(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        use_provider(
            handle,
            astrbot_stub.FakeProvider(reply='{"text": "<|EMO|>开心 你好<|PAUSE|>呀"}'),
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        spoken = dashscope.state.requests[0]["input"]["text"]
        assert "<|" not in spoken
        assert "开心" in spoken and "你好" in spoken

    async def test_tags_survive_when_the_style_is_enabled(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_tag_style="qwen-audio"))
        use_provider(
            handle,
            astrbot_stub.FakeProvider(reply='{"text": "<|EMO|>开心 你好"}'),
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        assert dashscope.state.requests[0]["input"]["text"].startswith("<|EMO|>")

    async def test_rewrite_can_be_forbidden(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_rewrite=False))
        provider = astrbot_stub.FakeProvider(reply='{"text": "被改写了"}')
        use_provider(handle, provider)

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert "不得改动原文字句" in provider.calls[0]["system_prompt"]
        # The director is instructed not to rewrite, but nothing stops a model
        # from ignoring it; the configuration is what is asserted here.

    async def test_director_input_is_truncated(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_max_input_chars=50))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 " + "字" * 300))

        assert "字" * 51 not in provider.calls[0]["prompt"]


class TestDirectMode:
    async def test_direct_mode_reaches_the_configured_endpoint(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        llm_server.state.reply = SCRIPT
        handle = await started(
            harness,
            base_config(
                dashscope,
                llm_mode="direct",
                llm_base_url=llm_server.base_url,
                llm_api_key="sk-llm",
                llm_model="deepseek-chat",
            ),
        )

        results = await handle.drive(
            handle.plugin.on_message, handle.new_event("小助手 念一句晚安")
        )

        assert records_of(results)
        assert llm_server.state.requests[0]["model"] == "deepseek-chat"
        assert dashscope.state.requests[0]["input"]["text"] == "晚安呀，做个好梦～"

    async def test_direct_mode_sends_the_configured_extra_headers(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        """The Opencode case: a gateway that requires its own headers."""
        handle = await started(
            harness,
            base_config(
                dashscope,
                llm_mode="direct",
                llm_base_url=llm_server.base_url,
                llm_api_key="sk-llm",
                llm_auth_style="x-api-key",
                llm_extra_headers="anthropic-version: 2023-06-01\nX-Title: astrbot-tts-wakeword",
            ),
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        headers = llm_server.state.headers[0]
        assert headers["x-api-key"] == "sk-llm"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["X-Title"] == "astrbot-tts-wakeword"

    async def test_direct_mode_without_a_key_disables_the_stage(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        # A missing credential must not break TTS: the stage is skipped and the
        # original text is synthesized.
        handle = await started(
            harness,
            base_config(dashscope, llm_mode="direct", llm_api_key=""),
        )

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "你好呀"

    async def test_direct_mode_auth_style_none_needs_no_key(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
        llm_server: mock_llm.MockLlm,
    ) -> None:
        llm_server.state.reply = SCRIPT
        handle = await started(
            harness,
            base_config(
                dashscope,
                llm_mode="direct",
                llm_base_url=llm_server.base_url,
                llm_api_key="",
                llm_auth_style="none",
            ),
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 晚安"))

        headers = llm_server.state.headers[0]
        assert "Authorization" not in headers
        assert dashscope.state.requests[0]["input"]["text"] == "晚安呀，做个好梦～"

    async def test_direct_mode_failure_falls_back(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            base_config(
                dashscope,
                llm_mode="direct",
                llm_base_url="http://127.0.0.1:1/v1",
                llm_api_key="sk-llm",
                llm_max_retries=0,
                llm_timeout_seconds=3,
            ),
        )

        results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好呀"))

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "你好呀"


class TestDebugAndStatus:
    async def test_tts_llm_shows_the_script_without_synthesizing(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            base_config(dashscope, model="qwen3-tts-instruct-flash"),
        )
        use_provider(handle, astrbot_stub.FakeProvider(reply=SCRIPT))

        results = await handle.drive(
            handle.plugin.tts_llm_command,
            handle.new_event("tts_llm 念一句晚安"),
        )

        output = "\n".join(texts_of(results))
        assert "晚安呀，做个好梦～" in output
        assert "轻声细语" in output
        assert "加了语气词" in output
        assert dashscope.state.requests == []

    async def test_tts_llm_reports_a_disabled_stage(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_enable=False))

        results = await handle.drive(
            handle.plugin.tts_llm_command,
            handle.new_event("tts_llm 你好"),
        )

        assert any("关闭" in text for text in texts_of(results))

    async def test_tts_llm_uses_the_quoted_message(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        provider = astrbot_stub.FakeProvider(reply=SCRIPT)
        use_provider(handle, provider)

        await handle.drive(
            handle.plugin.tts_llm_command,
            handle.new_event("tts_llm", messages=[astrbot_stub.Reply(message_str="引用内容")]),
        )

        assert "引用内容" in provider.calls[0]["prompt"]

    async def test_tts_llm_reports_a_failure(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope, llm_fail_open=False))

        results = await handle.drive(
            handle.plugin.tts_llm_command,
            handle.new_event("tts_llm 你好"),
        )

        assert any("失败" in text for text in texts_of(results))

    async def test_status_describes_the_llm_stage(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            base_config(dashscope, llm_tag_style="qwen-audio", llm_context_messages=4),
        )

        results = await handle.drive(
            handle.plugin.tts_status_command,
            handle.new_event("tts_status"),
        )
        output = "\n".join(texts_of(results))

        assert "LLM 预处理" in output
        assert "astrbot_provider" in output
        assert "qwen-audio" in output
        assert "上下文条数：4" in output

    async def test_status_masks_the_direct_mode_key(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            base_config(
                dashscope,
                llm_mode="direct",
                llm_base_url="https://api.deepseek.com/v1",
                llm_api_key="sk-llm-super-secret",
                llm_extra_headers="X-Title: demo",
            ),
        )

        results = await handle.drive(
            handle.plugin.tts_status_command,
            handle.new_event("tts_status"),
        )
        output = "\n".join(texts_of(results))

        assert "sk-llm-super-secret" not in output
        assert "sk-llm****cret" in output
        assert "X-Title" in output

    async def test_help_mentions_the_llm_command(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))

        results = await handle.drive(
            handle.plugin.tts_help_command,
            handle.new_event("tts_help"),
        )

        assert any("/tts_llm" in text for text in texts_of(results))

    def test_the_command_is_registered(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = harness(base_config(dashscope))
        registered = dict(handle.filters.registry.commands)
        assert "tts_llm" in registered
        assert {"tts_direct", "tts_script"} <= registered["tts_llm"]


class TestConfigReload:
    async def test_changing_the_provider_takes_effect_without_a_reload(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, base_config(dashscope))
        first = astrbot_stub.FakeProvider(reply='{"text": "第一条"}')
        use_provider(handle, first)

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        # Switching the model in the WebUI changes the LLM fingerprint, so the
        # client is rebuilt on the next message.
        handle.plugin.config["llm_model"] = "deepseek-v4-pro"
        second = astrbot_stub.FakeProvider(reply='{"text": "第二条"}')
        use_provider(handle, second)

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        assert dashscope.state.requests[0]["input"]["text"] == "第一条"
        assert dashscope.state.requests[1]["input"]["text"] == "第二条"
        assert second.calls[0]["model"] == "deepseek-v4-pro"


@pytest.mark.parametrize("mode", ["astrbot_provider", "direct"])
async def test_the_wake_word_pipeline_works_in_both_modes(
    harness: Any,
    dashscope: mock_dashscope.MockDashScope,
    llm_server: mock_llm.MockLlm,
    mode: str,
) -> None:
    config = base_config(dashscope, llm_mode=mode)
    if mode == "direct":
        llm_server.state.reply = SCRIPT
        config.update(
            {
                "llm_base_url": llm_server.base_url,
                "llm_api_key": "sk-llm",
            },
        )
    handle = await started(harness, config)
    if mode == "astrbot_provider":
        use_provider(handle, astrbot_stub.FakeProvider(reply=SCRIPT))

    results = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 念一句晚安"))

    assert records_of(results)
    assert dashscope.state.requests[0]["input"]["text"] == "晚安呀，做个好梦～"
