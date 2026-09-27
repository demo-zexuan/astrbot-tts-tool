"""Unit tests for wake-word matching, sanitizing and segmentation.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

from core.text_utils import (
    extract_command_argument,
    match_wake_word,
    sanitize_text,
    split_for_tts,
    strip_wake_word,
)


class TestWakeMatching:
    def test_basic_prefix_match_returns_the_payload(self) -> None:
        assert match_wake_word("小助手 你好呀", ["小助手"]) == ("小助手", "你好呀")

    def test_no_match_returns_none_and_normalized_text(self) -> None:
        word, rest = match_wake_word("  你好呀", ["小助手"])
        assert word is None
        assert rest == "你好呀"

    def test_longest_wake_word_wins(self) -> None:
        word, rest = match_wake_word("小助手小美 你好", ["小助手", "小助手小美"])
        assert (word, rest) == ("小助手小美", "你好")

    def test_match_is_case_insensitive_by_default(self) -> None:
        # The configured spelling is returned, not the one the user typed.
        assert match_wake_word("HEYBOT hi", ["heybot"]) == ("heybot", "hi")
        assert match_wake_word("HeyBot hi", ["heybot"]) == ("heybot", "hi")

    def test_case_sensitive_mode(self) -> None:
        assert match_wake_word("HEYBOT hi", ["heybot"], case_sensitive=True)[0] is None
        assert match_wake_word("heybot hi", ["heybot"], case_sensitive=True)[0] == "heybot"

    def test_leading_invisible_characters_are_ignored(self) -> None:
        word, rest = match_wake_word("\u200b\u3000小助手 你好", ["小助手"])
        assert (word, rest) == ("小助手", "你好")

    def test_separator_requirement_blocks_word_continuations(self) -> None:
        assert match_wake_word("小助手们好", ["小助手"], require_separator=True)[0] is None
        assert match_wake_word("小助手，你好", ["小助手"], require_separator=True)[0] == "小助手"
        assert match_wake_word("小助手", ["小助手"], require_separator=True)[0] == "小助手"

    def test_separator_not_required_allows_direct_continuation(self) -> None:
        assert match_wake_word("小助手你好", ["小助手"]) == ("小助手", "你好")

    def test_empty_inputs_are_safe(self) -> None:
        assert match_wake_word("", ["小助手"]) == (None, "")
        assert match_wake_word("小助手 hi", []) == (None, "小助手 hi")

    def test_strip_wake_word_removes_separators(self) -> None:
        assert strip_wake_word("小助手，你好", "小助手") == "你好"
        assert strip_wake_word("别的", "小助手") == "别的"


class TestCommandArgument:
    def test_text_after_the_command_token_is_kept_verbatim(self) -> None:
        assert extract_command_argument("tts 你好 世界") == "你好 世界"

    def test_command_without_argument(self) -> None:
        assert extract_command_argument("tts") == ""
        assert extract_command_argument("") == ""

    def test_internal_spacing_is_preserved(self) -> None:
        assert extract_command_argument("tts   a  b ") == "a  b"


class TestSanitize:
    def test_cq_codes_and_zero_width_spaces_are_removed(self) -> None:
        assert sanitize_text("[CQ:at,qq=1]\u200b你好") == "你好"

    def test_markdown_is_unwrapped_but_content_survives(self) -> None:
        text = "# 标题\n**加粗**和`代码`以及[链接](https://example.com)"
        assert sanitize_text(text) == "标题 加粗和代码以及链接"

    def test_urls_are_removed_when_enabled(self) -> None:
        assert sanitize_text("看这个 https://example.com/a 挺好") == "看这个 挺好"
        assert "https://example.com/a" in sanitize_text(
            "看这个 https://example.com/a 挺好",
            clean_url=False,
        )

    def test_emoji_is_kept_unless_enabled(self) -> None:
        assert "😀" in sanitize_text("你好😀")
        assert sanitize_text("你好😀", clean_emoji=True) == "你好"

    def test_code_blocks_keep_their_body(self) -> None:
        assert sanitize_text("```python\nprint(1)\n```") == "print(1)"

    def test_whitespace_is_collapsed(self) -> None:
        assert sanitize_text("你好\n\n\n   世界") == "你好 世界"

    def test_empty_input(self) -> None:
        assert sanitize_text("") == ""
        assert sanitize_text("   \n  ") == ""


class TestSplitting:
    def test_short_text_is_a_single_segment(self) -> None:
        assert split_for_tts("你好呀", max_chars=50, max_segments=3) == (["你好呀"], False)

    def test_sentences_are_packed_within_the_limit(self) -> None:
        text = "第一句。第二句。第三句。"
        segments, truncated = split_for_tts(text, max_chars=8, max_segments=5)
        assert truncated is False
        assert all(len(segment) <= 8 for segment in segments)
        assert "".join(segments).replace(" ", "") == text

    def test_a_long_sentence_is_cut_at_a_soft_break(self) -> None:
        text = "这是一段很长的话，" + "后半部分也还是很长" * 3
        segments, _ = split_for_tts(text, max_chars=20, max_segments=10)
        assert len(segments) > 1
        assert all(len(segment) <= 20 for segment in segments)

    def test_a_long_sentence_without_breaks_is_hard_cut(self) -> None:
        text = "字" * 100
        segments, _ = split_for_tts(text, max_chars=30, max_segments=10)
        assert segments
        assert all(len(segment) <= 30 for segment in segments)

    def test_max_segments_marks_truncation(self) -> None:
        text = "。".join(f"第{i}句" for i in range(10)) + "。"
        segments, truncated = split_for_tts(text, max_chars=6, max_segments=2)
        assert truncated is True
        assert len(segments) == 2

    def test_splitting_disabled_truncates_in_place(self) -> None:
        segments, truncated = split_for_tts(
            "字" * 100,
            max_chars=10,
            max_segments=5,
            split_long_text=False,
        )
        assert segments == ["字" * 10]
        assert truncated is True

    def test_splitting_disabled_keeps_short_text_intact(self) -> None:
        assert split_for_tts("短", max_chars=10, max_segments=5, split_long_text=False) == (
            ["短"],
            False,
        )

    def test_empty_text_yields_nothing(self) -> None:
        assert split_for_tts("   ", max_chars=10, max_segments=3) == ([], False)
