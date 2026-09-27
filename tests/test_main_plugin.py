"""End-to-end tests for the plugin entry point.

``main.py`` is imported exactly the way AstrBot imports it
(``data.plugins.<plugin_dir>.main``, relative imports included) against the stub
package in ``tests/astrbot_stub.py``. Every test drives a handler with a fake
event and inspects the resulting message chain, while the HTTP layer talks to the
local fake DashScope endpoint.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests import astrbot_stub, mock_dashscope
from tests.conftest import PluginHarness, records_of, texts_of

BASE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "wake_words": ["小助手"],
    "api_key": "sk-test",
    "model": "qwen-audio-3.1-tts-flash",
    "voice": "Cherry",
}


def make_config(server: mock_dashscope.MockDashScope, **overrides: Any) -> dict[str, Any]:
    config = {**BASE_CONFIG, "base_url": server.base_url}
    config.update(overrides)
    return config


async def started(harness: Any, config: dict[str, Any], **context: Any) -> PluginHarness:
    """Build a plugin and run its AstrBot ``initialize`` hook."""
    handle = harness(config, **context)
    await handle.plugin.initialize()
    return handle


def record_paths(results: list[Any]) -> list[Path]:
    return [Path(record.path) for record in records_of(results)]


class TestWakeWord:
    async def test_wake_word_replies_with_a_voice_message(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event("小助手 你好呀")

        results = await handle.drive(handle.plugin.on_message, event)

        records = records_of(results)
        assert len(records) == 1
        paths = record_paths(results)
        assert paths[0].parent == handle.audio_dir
        assert paths[0].read_bytes().startswith(b"RIFF")
        assert records[0].file.startswith("file://")
        assert dashscope.state.requests[0]["input"]["text"] == "你好呀"

    async def test_the_event_is_stopped_so_the_llm_does_not_also_reply(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event("小助手 你好呀")

        await handle.drive(handle.plugin.on_message, event)

        assert event.is_stopped() is True
        assert event.call_llm is False

    async def test_stopping_can_be_disabled(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, stop_event_on_wake=False))
        event = handle.new_event("小助手 你好呀")

        results = await handle.drive(handle.plugin.on_message, event)

        assert records_of(results)
        assert event.is_stopped() is False

    async def test_unrelated_messages_are_ignored(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event("今天天气怎么样")

        results = await handle.drive(handle.plugin.on_message, event)

        assert results == []
        assert dashscope.state.requests == []

    async def test_no_wake_words_configured_means_no_listening(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, wake_words=[]))
        event = handle.new_event("小助手 你好呀")

        assert await handle.drive(handle.plugin.on_message, event) == []

    async def test_bare_wake_word_answers_with_a_prompt(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event("小助手")

        results = await handle.drive(handle.plugin.on_message, event)

        assert records_of(results) == []
        assert any("我在" in text for text in texts_of(results))
        assert dashscope.state.requests == []
        assert event.is_stopped() is True

    async def test_disabled_plugin_ignores_everything(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, enabled=False))
        event = handle.new_event("小助手 你好呀")

        assert await handle.drive(handle.plugin.on_message, event) == []
        assert dashscope.state.requests == []

    async def test_separator_requirement_prevents_partial_word_matches(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(dashscope, wake_require_separator=True),
        )

        blocked = handle.new_event("小助手们真棒")
        assert await handle.drive(handle.plugin.on_message, blocked) == []

        allowed = handle.new_event("小助手，你好")
        assert records_of(await handle.drive(handle.plugin.on_message, allowed))

    async def test_stripping_the_wake_word_can_be_disabled(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, strip_wake_word=False))
        event = handle.new_event("小助手 你好呀")

        await handle.drive(handle.plugin.on_message, event)

        assert "小助手" in dashscope.state.requests[0]["input"]["text"]

    async def test_platform_commands_are_not_hijacked(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        # AstrBot strips the wake prefix, so the handler sees "help" while the
        # original message keeps the "/" that marks it as a command.
        handle = await started(harness, make_config(dashscope, wake_words=["help"]))
        message_obj = astrbot_stub.AstrBotMessage(message_str="/help")
        event = handle.new_event("help", message_obj=message_obj)

        assert await handle.drive(handle.plugin.on_message, event) == []
        assert dashscope.state.requests == []

    async def test_cooldown_suppresses_a_second_trigger(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, cooldown_seconds=60))

        first = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 一"))
        second = await handle.drive(handle.plugin.on_message, handle.new_event("小助手 二"))

        assert records_of(first)
        assert second == []
        assert len(dashscope.state.requests) == 1

    async def test_cooldown_is_per_session(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, cooldown_seconds=60))

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 一"))
        other = await handle.drive(
            handle.plugin.on_message,
            handle.new_event("小助手 二", unified_msg_origin="test:FriendMessage:999"),
        )

        assert records_of(other)
        assert len(dashscope.state.requests) == 2


class TestWarmProfiles:
    async def test_a_profile_binds_its_own_voice(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(
                dashscope,
                wake_profiles=[{"wake_word": "小美", "voice": "Ethan"}],
            ),
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小美 你好"))

        assert dashscope.state.requests[0]["input"]["voice"] == "Ethan"

    async def test_the_global_voice_still_applies_to_other_wake_words(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(
                dashscope,
                wake_profiles=[{"wake_word": "小美", "voice": "Ethan"}],
            ),
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        assert dashscope.state.requests[0]["input"]["voice"] == "Cherry"

    async def test_the_longest_wake_word_wins(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(
                dashscope,
                wake_words=["小助手"],
                wake_profiles=[{"wake_word": "小助手小美", "voice": "Serena"}],
            ),
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手小美 你好"))

        assert dashscope.state.requests[0]["input"]["voice"] == "Serena"
        assert dashscope.state.requests[0]["input"]["text"] == "你好"

    async def test_a_profile_can_override_the_model(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(
                dashscope,
                wake_profiles=[{"wake_word": "小美", "model": "qwen3-tts-flash"}],
            ),
        )

        await handle.drive(handle.plugin.on_message, handle.new_event("小美 你好"))

        assert dashscope.state.requests[0]["model"] == "qwen3-tts-flash"


class TestTtsCommand:
    async def test_command_synthesizes_the_free_form_argument(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event("tts 你好 世界")

        results = await handle.drive(handle.plugin.tts_command, event)

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "你好 世界"

    async def test_command_without_arguments_shows_usage(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event("tts")

        results = await handle.drive(handle.plugin.tts_command, event)

        assert records_of(results) == []
        assert any("使用说明" in text for text in texts_of(results))
        assert dashscope.state.requests == []

    async def test_command_falls_back_to_the_quoted_message(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event(
            "tts",
            messages=[astrbot_stub.Reply(message_str="被引用的内容")],
        )

        results = await handle.drive(handle.plugin.tts_command, event)

        assert records_of(results)
        assert dashscope.state.requests[0]["input"]["text"] == "被引用的内容"

    async def test_command_can_be_disabled(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, enable_command=False))
        event = handle.new_event("tts 你好")

        results = await handle.drive(handle.plugin.tts_command, event)

        assert records_of(results) == []
        assert any("关闭" in text for text in texts_of(results))

    async def test_empty_text_after_cleaning_reports_a_hint(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event("tts https://example.com")

        results = await handle.drive(handle.plugin.tts_command, event)

        assert records_of(results) == []
        assert any("没有可合成的文本" in text for text in texts_of(results))


class TestLongText:
    async def test_long_text_produces_multiple_records(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(dashscope, max_chars=20, max_segments=4),
        )
        event = handle.new_event("小助手 " + "这是一句测试。" * 4)

        results = await handle.drive(handle.plugin.on_message, event)

        assert len(records_of(results)) > 1
        assert len(dashscope.state.requests) == len(records_of(results))

    async def test_exceeding_the_segment_limit_announces_truncation(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(dashscope, max_chars=8, max_segments=1),
        )
        event = handle.new_event("小助手 " + "这是一句测试。" * 4)

        results = await handle.drive(handle.plugin.on_message, event)

        assert len(records_of(results)) == 1
        assert any("文本过长" in text for text in texts_of(results))

    async def test_echo_source_text_prefixes_the_audio(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, echo_source_text=True))
        event = handle.new_event("小助手 你好呀")

        results = await handle.drive(handle.plugin.on_message, event)

        assert "你好呀" in texts_of(results)
        assert records_of(results)


class TestErrors:
    async def test_a_rejected_api_key_replies_with_a_hint(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.synthesis_status = 401
        handle = await started(harness, make_config(dashscope))
        event = handle.new_event("小助手 你好呀")

        results = await handle.drive(handle.plugin.on_message, event)

        assert records_of(results) == []
        assert any("鉴权失败" in text for text in texts_of(results))
        # Errors must not swallow the message: the LLM may still answer.
        assert event.is_stopped() is False

    async def test_error_replies_can_be_silenced(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        dashscope.state.synthesis_status = 401
        handle = await started(harness, make_config(dashscope, reply_on_error=False))
        event = handle.new_event("小助手 你好呀")

        assert await handle.drive(handle.plugin.on_message, event) == []

    async def test_a_missing_api_key_is_explained(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, api_key=""))
        event = handle.new_event("小助手 你好呀")

        results = await handle.drive(handle.plugin.on_message, event)

        assert any("API Key" in text for text in texts_of(results))


class TestSessionVoice:
    async def test_setting_and_using_a_session_voice(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))

        set_results = await handle.drive(
            handle.plugin.tts_voice_command,
            handle.new_event("tts_voice Serena"),
        )
        assert any("Serena" in text for text in texts_of(set_results))

        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))
        assert dashscope.state.requests[-1]["input"]["voice"] == "Serena"

    async def test_showing_the_current_voice(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))

        results = await handle.drive(
            handle.plugin.tts_voice_command,
            handle.new_event("tts_voice"),
        )

        assert any("Cherry" in text for text in texts_of(results))

    async def test_resetting_the_session_voice(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        await handle.drive(handle.plugin.tts_voice_command, handle.new_event("tts_voice Serena"))

        reset = await handle.drive(
            handle.plugin.tts_voice_command,
            handle.new_event("tts_voice reset"),
        )

        assert any("Cherry" in text for text in texts_of(reset))
        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))
        assert dashscope.state.requests[-1]["input"]["voice"] == "Cherry"

    async def test_the_session_voice_survives_a_plugin_reload(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        config = make_config(dashscope)
        handle = await started(harness, config)
        await handle.drive(handle.plugin.tts_voice_command, handle.new_event("tts_voice Serena"))

        # A second instance loads the persisted session voice from disk.
        reloaded = await started(harness, config)
        await reloaded.drive(reloaded.plugin.on_message, reloaded.new_event("小助手 你好"))

        assert dashscope.state.requests[-1]["input"]["voice"] == "Serena"

    async def test_the_session_voice_file_is_valid_json(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        await handle.drive(handle.plugin.tts_voice_command, handle.new_event("tts_voice Serena"))

        stored = handle.data_dir / "plugin_data" / handle.module.PLUGIN_DIR_NAME
        payload = json.loads((stored / "session_voices.json").read_text(encoding="utf-8"))
        assert list(payload.values()) == ["Serena"]

    async def test_the_override_can_be_disabled(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(dashscope, allow_session_voice_override=False),
        )

        results = await handle.drive(
            handle.plugin.tts_voice_command,
            handle.new_event("tts_voice Serena"),
        )

        assert any("未开启" in text for text in texts_of(results))

    async def test_admin_only_mode_rejects_regular_users(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope, admin_only_voice_switch=True))

        denied = await handle.drive(
            handle.plugin.tts_voice_command,
            handle.new_event("tts_voice Serena", admin=False),
        )
        assert any("管理员" in text for text in texts_of(denied))

        allowed = await handle.drive(
            handle.plugin.tts_voice_command,
            handle.new_event("tts_voice Serena", admin=True),
        )
        assert any("Serena" in text for text in texts_of(allowed))


class TestStatusAndHelp:
    async def test_status_reports_the_effective_configuration_without_leaking_the_key(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(dashscope, api_key="sk-super-secret-value"),
        )

        results = await handle.drive(
            handle.plugin.tts_status_command,
            handle.new_event("tts_status"),
        )
        output = "\n".join(texts_of(results))

        assert "qwen-audio-3.1-tts-flash" in output
        assert "Cherry" in output
        assert "小助手" in output
        assert "sk-super-secret-value" not in output
        assert "sk-sup****alue" in output

    async def test_status_lists_wake_profiles(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(
            harness,
            make_config(dashscope, wake_profiles=[{"wake_word": "小美", "voice": "Ethan"}]),
        )

        results = await handle.drive(
            handle.plugin.tts_status_command,
            handle.new_event("tts_status"),
        )
        output = "\n".join(texts_of(results))

        assert "小美" in output and "Ethan" in output

    async def test_status_counts_cached_audio(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        results = await handle.drive(
            handle.plugin.tts_status_command,
            handle.new_event("tts_status"),
        )

        assert any("音频缓存：1 个文件" in text for text in texts_of(results))

    async def test_help_lists_the_commands(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))

        results = await handle.drive(
            handle.plugin.tts_help_command,
            handle.new_event("tts_help"),
        )
        output = "\n".join(texts_of(results))

        for command in ("/tts ", "/tts_voice", "/tts_status", "/tts_help"):
            assert command in output


class TestRegistration:
    def test_all_commands_are_registered_with_their_aliases(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = harness(make_config(dashscope))
        registered = dict(handle.filters.registry.commands)

        assert "tts" in registered
        assert {"tts_say", "语音", "合成"} <= registered["tts"]
        assert "tts_voice" in registered
        assert "tts_status" in registered
        assert "tts_help" in registered

    def test_the_wake_listener_is_registered_for_all_message_types(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = harness(make_config(dashscope))

        assert handle.filters.registry.event_message_types == [astrbot_stub.EventMessageType.ALL]

    def test_the_plugin_class_is_discoverable_by_astrbot(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        # AstrBot falls back to class names ending in "Plugin" or named "Main".
        handle = harness(make_config(dashscope))
        assert handle.plugin.__class__.__name__.lower().endswith("plugin")


class TestStorageLayout:
    async def test_audio_is_cached_under_plugin_data(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        await handle.drive(handle.plugin.on_message, handle.new_event("小助手 你好"))

        expected_root = handle.data_dir / "plugin_data" / handle.module.PLUGIN_DIR_NAME
        assert handle.audio_dir.is_dir()
        assert handle.audio_dir.parent == expected_root

    async def test_terminate_is_idempotent(
        self,
        harness: Any,
        dashscope: mock_dashscope.MockDashScope,
    ) -> None:
        handle = await started(harness, make_config(dashscope))
        await handle.plugin.terminate()
        await handle.plugin.terminate()


def test_masking_short_secrets_is_safe() -> None:
    from core.config import mask_secret

    assert mask_secret("") == "（未配置）"
    assert mask_secret("short") == "sh****"
    assert mask_secret("sk-1234567890abcdef") == "sk-123****cdef"


@pytest.mark.parametrize("wake_word", ["小助手", "Hey Bot", "小Q"])
def test_wake_word_normalization_is_stable(wake_word: str) -> None:
    from core.text_utils import match_wake_word

    matched, rest = match_wake_word(f"{wake_word} 内容", [wake_word])
    assert matched == wake_word
    assert rest == "内容"
