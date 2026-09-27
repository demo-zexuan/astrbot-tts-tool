"""Tests for the LLM director stage: capabilities, prompting and parsing.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import json

import pytest

from core.config import (
    TAG_STYLE_CUSTOM,
    TAG_STYLE_NONE,
    TAG_STYLE_QWEN_AUDIO,
    TAG_STYLE_SSML,
    PluginSettings,
)
from core.director import (
    DEFAULT_SYSTEM_PROMPT,
    TtsCapabilities,
    build_messages,
    parse_director_response,
    strip_forbidden_tags,
)


def settings(**overrides: object) -> PluginSettings:
    return PluginSettings.from_dict({"api_key": "sk-test", **overrides})


def capabilities(**overrides: object) -> TtsCapabilities:
    return TtsCapabilities.from_settings(settings(**overrides))


class TestCapabilities:
    def test_instructions_are_only_offered_for_instruct_models(self) -> None:
        assert capabilities(model="qwen3-tts-instruct-flash").supports_instructions is True
        assert capabilities(model="qwen3-tts-flash").supports_instructions is False
        assert capabilities(model="qwen-tts").supports_instructions is False

    def test_tags_are_off_by_default(self) -> None:
        caps = capabilities()
        assert caps.tag_style == TAG_STYLE_NONE
        assert caps.tags_enabled is False

    def test_tags_can_be_enabled(self) -> None:
        assert capabilities(llm_tag_style=TAG_STYLE_QWEN_AUDIO).tags_enabled is True
        assert capabilities(llm_tag_style=TAG_STYLE_SSML).tags_enabled is True

    def test_description_forbids_tags_when_disabled(self) -> None:
        text = capabilities(model="qwen3-tts-flash").describe()
        assert "严格禁止" in text
        assert "<|EMO|>" in text  # named as a counter-example
        assert "不支持 instructions" in text

    def test_description_offers_tags_when_enabled(self) -> None:
        text = capabilities(llm_tag_style=TAG_STYLE_QWEN_AUDIO).describe()
        assert "<|EMO|>" in text
        assert "<|SPEED|>" in text
        assert "严格禁止" not in text

    def test_custom_tags_are_inlined(self) -> None:
        caps = capabilities(
            llm_tag_style=TAG_STYLE_CUSTOM,
            llm_custom_tags="只允许 <|PAUSE|>",
        )
        assert "只允许 <|PAUSE|>" in caps.describe()

    def test_empty_custom_tags_still_forbids_everything(self) -> None:
        caps = capabilities(llm_tag_style=TAG_STYLE_CUSTOM, llm_custom_tags="")
        assert "请不要输出任何标签" in caps.describe()

    def test_output_budget_is_reported(self) -> None:
        caps = capabilities(max_chars=100, max_segments=4)
        assert "约 400 字" in caps.describe()

    def test_voice_and_language_are_reported(self) -> None:
        caps = capabilities(voice="Ethan", language_type="Chinese")
        text = caps.describe()
        assert "Ethan" in text and "Chinese" in text


class TestPromptBuilding:
    def test_messages_use_system_then_user(self) -> None:
        messages = build_messages(text="你好", capabilities=capabilities())
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_user_message_carries_capabilities_context_and_text(self) -> None:
        messages = build_messages(
            text="小助手 温柔点念",
            capabilities=capabilities(model="qwen3-tts-instruct-flash"),
            history=[
                {"role": "user", "content": "宝宝要睡觉了"},
                {"role": "assistant", "content": "好的，晚安"},
            ],
        )
        user = messages[1]["content"]
        assert "【目标模型能力】" in user
        assert "【最近的对话上下文】" in user
        assert "【需要朗读的原文】" in user
        assert "宝宝要睡觉了" in user
        assert "小助手 温柔点念" in user
        # The context must be role-labelled so the model can tell speakers apart.
        assert "用户：" in user and "机器人：" in user

    def test_missing_history_is_stated_explicitly(self) -> None:
        messages = build_messages(text="你好", capabilities=capabilities(), history=None)
        assert "（无历史消息）" in messages[1]["content"]

    def test_history_entries_without_content_are_skipped(self) -> None:
        messages = build_messages(
            text="你好",
            capabilities=capabilities(),
            history=[{"role": "user", "content": ""}, {"role": "user", "content": "有内容"}],
        )
        user = messages[1]["content"]
        assert "有内容" in user

    def test_disabling_rewrite_adds_an_explicit_constraint(self) -> None:
        messages = build_messages(
            text="你好",
            capabilities=capabilities(),
            allow_rewrite=False,
        )
        assert "不得改动原文字句" in messages[0]["content"]

    def test_a_custom_system_prompt_replaces_the_default(self) -> None:
        messages = build_messages(
            text="你好",
            capabilities=capabilities(),
            system_prompt="你只说方言",
        )
        assert messages[0]["content"] == "你只说方言"

    def test_the_default_prompt_demands_json_only(self) -> None:
        assert "只输出一个 JSON 对象" in DEFAULT_SYSTEM_PROMPT
        assert "绝不改变" in DEFAULT_SYSTEM_PROMPT


class TestParsing:
    def test_plain_json_is_parsed(self) -> None:
        result = parse_director_response(
            '{"text": "晚安呀～", "instructions": "轻声", "tags": ["<|EMO|>温柔"], "reason": "加语气词"}',
            original_text="晚安",
            capabilities=capabilities(
                model="qwen3-tts-instruct-flash", llm_tag_style=TAG_STYLE_QWEN_AUDIO
            ),
        )
        assert result.text == "晚安呀～"
        assert result.instructions == "轻声"
        assert result.tags == ("<|EMO|>温柔",)
        assert result.reason == "加语气词"
        assert result.used_fallback is False

    def test_fenced_json_is_parsed(self) -> None:
        raw = '这是结果：\n```json\n{"text": "你好呀"}\n```\n希望有帮助'
        result = parse_director_response(raw, original_text="你好")
        assert result.text == "你好呀"

    def test_json_embedded_in_prose_is_parsed(self) -> None:
        raw = '好的，我改好了：{"text": "你好呀"} 就这样。'
        result = parse_director_response(raw, original_text="你好")
        assert result.text == "你好呀"

    def test_plain_prose_reply_is_used_as_the_script(self) -> None:
        result = parse_director_response("你好呀，很高兴见到你～", original_text="你好")
        assert result.text == "你好呀，很高兴见到你～"
        assert result.used_fallback is False

    def test_empty_reply_falls_back_to_the_original_text(self) -> None:
        result = parse_director_response("   ", original_text="原文")
        assert result.text == "原文"
        assert result.used_fallback is True

    def test_json_without_text_falls_back(self) -> None:
        result = parse_director_response('{"instructions": "快点"}', original_text="原文")
        assert result.text == "原文"
        assert result.used_fallback is True

    def test_instructions_are_dropped_for_non_instruct_models(self) -> None:
        result = parse_director_response(
            '{"text": "你好", "instructions": "温柔"}',
            original_text="你好",
            capabilities=capabilities(model="qwen3-tts-flash"),
        )
        assert result.instructions == ""

    def test_instructions_survive_for_instruct_models(self) -> None:
        result = parse_director_response(
            '{"text": "你好", "instructions": "温柔"}',
            original_text="你好",
            capabilities=capabilities(model="qwen3-tts-instruct-flash"),
        )
        assert result.instructions == "温柔"

    def test_tags_are_stripped_when_the_model_cannot_read_them(self) -> None:
        result = parse_director_response(
            '{"text": "<|EMO|>开心 你好<|PAUSE|>呀", "tags": ["<|EMO|>", "<|PAUSE|>"]}',
            original_text="你好",
            capabilities=capabilities(llm_tag_style=TAG_STYLE_NONE),
        )
        assert "<|" not in result.text
        assert result.stripped_tags == ("<|EMO|>", "<|PAUSE|>")
        assert result.tags == ()
        assert "开心" in result.text and "你好" in result.text

    def test_tags_are_kept_when_enabled(self) -> None:
        result = parse_director_response(
            '{"text": "<|EMO|>开心 你好"}',
            original_text="你好",
            capabilities=capabilities(llm_tag_style=TAG_STYLE_QWEN_AUDIO),
        )
        assert result.text.startswith("<|EMO|>")
        assert result.stripped_tags == ()

    def test_non_string_values_are_coerced(self) -> None:
        result = parse_director_response('{"text": 123}', original_text="原文")
        assert result.text == "123"

    def test_raw_reply_is_retained_for_debugging(self) -> None:
        raw = '{"text": "hi"}'
        assert parse_director_response(raw, original_text="x").raw == raw

    def test_model_name_is_carried_through(self) -> None:
        result = parse_director_response(
            '{"text": "hi"}',
            original_text="x",
            model="deepseek-chat",
        )
        assert result.model == "deepseek-chat"


class TestTagStripping:
    def test_qwen_style_tags_are_removed(self) -> None:
        cleaned, removed = strip_forbidden_tags("你好<|EMO|>开心")
        assert cleaned == "你好 开心"
        assert removed == ("<|EMO|>",)

    def test_bracket_tags_are_removed(self) -> None:
        cleaned, removed = strip_forbidden_tags("你好[pause]呀")
        assert "[" not in cleaned
        assert removed == ("[pause]",)

    def test_ssml_tags_are_removed(self) -> None:
        cleaned, removed = strip_forbidden_tags('你好<break time="300ms"/>呀')
        assert "<" not in cleaned
        assert removed == ('<break time="300ms"/>',)

    def test_plain_text_is_untouched(self) -> None:
        text = "你好，今天天气不错（真的）。"
        cleaned, removed = strip_forbidden_tags(text)
        assert cleaned == text
        assert removed == ()

    def test_math_like_text_survives(self) -> None:
        cleaned, _ = strip_forbidden_tags("a < b 且 b > c")
        assert "a < b" in cleaned

    def test_multiple_tags_are_collapsed(self) -> None:
        cleaned, removed = strip_forbidden_tags("a<|EMO|><|PAUSE|>b")
        assert len(removed) == 2
        assert "a" in cleaned and "b" in cleaned
        assert "<|" not in cleaned


class TestJsonExtractionEdgeCases:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"text": "hi"}',
            '{"text": "hi", "tags": []}',
            '前缀 {"text": "hi"} 后缀',
            '```json\n{"text": "hi"}\n```',
        ],
    )
    def test_all_supported_shapes(self, raw: str) -> None:
        assert parse_director_response(raw, original_text="x").text == "hi"

    def test_malformed_json_falls_back_to_prose_then_original(self) -> None:
        result = parse_director_response('{"text": "hi"', original_text="原文")
        # Broken JSON is not valid prose either (it starts with "{"), so the
        # original text is used and the fallback flag is set.
        assert result.text == "原文"
        assert result.used_fallback is True

    def test_json_list_is_not_treated_as_an_object(self) -> None:
        result = parse_director_response('["a", "b"]', original_text="原文")
        assert result.text == "原文"
        assert result.used_fallback is True

    def test_a_valid_json_round_trip_via_json_dumps(self) -> None:
        payload = {"text": "你好", "instructions": "", "tags": [], "reason": ""}
        result = parse_director_response(json.dumps(payload), original_text="x")
        assert result.text == "你好"
