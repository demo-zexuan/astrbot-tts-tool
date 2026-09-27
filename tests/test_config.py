"""Unit tests for the configuration parser.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import pytest

from core.config import (
    DEFAULT_BASE_URLS,
    DEFAULT_MODEL,
    HTTP_TTS_MODELS,
    PluginSettings,
    TtsProfile,
    normalize_base_url,
)
from core.errors import TtsConfigError


class TestCoercion:
    """Values that reach the plugin from hand-edited JSON stay usable."""

    def test_defaults_are_applied_when_config_is_empty(self) -> None:
        settings = PluginSettings.from_dict({})
        assert settings.enabled is True
        assert settings.model == DEFAULT_MODEL
        assert settings.voice == "Cherry"
        assert settings.base_url == DEFAULT_BASE_URLS["beijing"]
        assert settings.wake_words == ()
        assert settings.profiles == ()

    def test_stringly_typed_scalars_are_coerced(self) -> None:
        settings = PluginSettings.from_dict(
            {
                "enabled": "false",
                "max_chars": "450",
                "cooldown_seconds": "2.5",
                "max_retries": "7",
                "cache_ttl_hours": "12",
            },
        )
        assert settings.enabled is False
        assert settings.max_chars == 450
        assert settings.cooldown_seconds == 2.5
        assert settings.max_retries == 7
        assert settings.cache_ttl_hours == 12.0

    def test_nonsense_values_fall_back_to_defaults(self) -> None:
        settings = PluginSettings.from_dict(
            {
                "enabled": None,
                "max_chars": "abc",
                "timeout_seconds": [],
            },
        )
        assert settings.enabled is True
        assert settings.max_chars == 300
        assert settings.timeout_seconds == 60.0

    def test_a_bare_scalar_becomes_a_single_wake_word(self) -> None:
        # A scalar can never be iterated, so it is treated as one wake word
        # rather than dropped silently.
        assert PluginSettings.from_dict({"wake_words": 42}).wake_words == ("42",)
        assert PluginSettings.from_dict({"wake_words": "小助手"}).wake_words == ("小助手",)

    def test_numeric_floor_is_enforced(self) -> None:
        settings = PluginSettings.from_dict(
            {"max_chars": 1, "max_concurrent": 0, "max_segments": -5, "max_cache_files": 0},
        )
        assert settings.max_chars == 20
        assert settings.max_concurrent == 1
        assert settings.max_segments == 1
        assert settings.max_cache_files == 1

    def test_unknown_language_type_falls_back_to_auto(self) -> None:
        assert PluginSettings.from_dict({"language_type": "Klingon"}).language_type == "Auto"
        assert PluginSettings.from_dict({"language_type": "Japanese"}).language_type == "Japanese"

    def test_empty_voice_falls_back_to_the_fallback_argument(self) -> None:
        settings = PluginSettings.from_dict({"voice": ""}, fallback_voice="Ethan")
        assert settings.voice == "Ethan"


class TestWakeWordList:
    def test_comma_and_newline_separated_strings_are_accepted(self) -> None:
        settings = PluginSettings.from_dict({"wake_words": "小助手，小Q\nAstrBot"})
        assert settings.wake_words == ("小助手", "小Q", "AstrBot")

    def test_duplicates_are_removed_case_insensitively(self) -> None:
        settings = PluginSettings.from_dict({"wake_words": ["Bot", "bot", "小助手"]})
        assert settings.wake_words == ("Bot", "小助手")

    def test_all_wake_words_merges_profiles_longest_first(self) -> None:
        settings = PluginSettings.from_dict(
            {
                "wake_words": ["小助手"],
                "wake_profiles": [{"wake_word": "小助手小美", "voice": "Ethan"}],
            },
        )
        assert settings.all_wake_words == ("小助手小美", "小助手")

    def test_profile_lookup_is_case_insensitive(self) -> None:
        settings = PluginSettings.from_dict(
            {"wake_profiles": [{"wake_word": "Mini", "voice": "Ethan"}]},
        )
        profile = settings.profile_for("mini")
        assert profile is not None and profile.voice == "Ethan"
        assert settings.profile_for("nope") is None


class TestProfiles:
    def test_plain_string_entries_become_wake_words(self) -> None:
        settings = PluginSettings.from_dict({"wake_profiles": ["小美"]})
        assert settings.profiles == (TtsProfile(wake_word="小美"),)

    def test_object_style_config_is_accepted(self) -> None:
        settings = PluginSettings.from_dict(
            {"wake_profiles": {"wake_word": "小美", "model": "qwen-tts", "voice": "Serena"}},
        )
        profile = settings.profiles[0]
        assert profile.model == "qwen-tts"
        assert profile.voice == "Serena"

    def test_inherit_language_and_invalid_language_are_cleared(self) -> None:
        settings = PluginSettings.from_dict(
            {
                "wake_profiles": [
                    {"wake_word": "a", "language_type": "inherit"},
                    {"wake_word": "b", "language_type": "Nonsense"},
                ],
            },
        )
        assert settings.profiles[0].language_type == ""
        assert settings.profiles[1].language_type == ""

    def test_entries_without_a_wake_word_are_dropped(self) -> None:
        settings = PluginSettings.from_dict(
            {"wake_profiles": [{"voice": "Ethan"}, {"wake_word": "  "}]},
        )
        assert settings.profiles == ()

    def test_optimize_instructions_tristate(self) -> None:
        settings = PluginSettings.from_dict(
            {
                "wake_profiles": [
                    {"wake_word": "a"},
                    {"wake_word": "b", "optimize_instructions": True},
                    {"wake_word": "c", "optimize_instructions": False},
                ],
            },
        )
        assert settings.profiles[0].optimize_instructions is None
        assert settings.profiles[1].optimize_instructions is True
        assert settings.profiles[2].optimize_instructions is False

    def test_apply_only_overrides_non_empty_fields(self) -> None:
        base = PluginSettings.from_dict(
            {"model": "global-model", "voice": "Cherry", "language_type": "Chinese"}
        )
        profile = TtsProfile(wake_word="小美", voice="Ethan")
        merged = profile.apply(base)
        assert merged.voice == "Ethan"
        assert merged.model == "global-model"
        assert merged.language_type == "Chinese"
        assert merged.wake_words == base.wake_words

    def test_apply_without_overrides_returns_the_same_object(self) -> None:
        base = PluginSettings.from_dict({})
        assert TtsProfile(wake_word="x").apply(base) is base


class TestBaseUrl:
    @pytest.mark.parametrize(
        ("raw", "region", "expected"),
        [
            ("", "beijing", "https://dashscope.aliyuncs.com/api/v1"),
            ("", "singapore", "https://dashscope-intl.aliyuncs.com/api/v1"),
            ("", "qwenai", "https://maas.qianwenaiapi.com/api/v1"),
            ("", "custom", "https://dashscope.aliyuncs.com/api/v1"),
            ("https://example.com", "custom", "https://example.com/api/v1"),
            ("https://example.com/api", "custom", "https://example.com/api/v1"),
            ("https://example.com/api/v1", "custom", "https://example.com/api/v1"),
            (
                "https://example.com/api/v1/services/aigc/multimodal-generation/generation/",
                "custom",
                "https://example.com/api/v1",
            ),
        ],
    )
    def test_normalization(self, raw: str, region: str, expected: str) -> None:
        assert normalize_base_url(raw, region) == expected

    def test_an_unknown_region_falls_back_to_beijing(self) -> None:
        settings = PluginSettings.from_dict({"region": "mars"})
        assert settings.region == "beijing"
        assert settings.base_url == DEFAULT_BASE_URLS["beijing"]

    def test_the_default_model_is_served_over_http(self) -> None:
        # Guards against re-introducing a WebSocket-only model as the default.
        assert DEFAULT_MODEL in HTTP_TTS_MODELS
        assert PluginSettings.from_dict({}).model in HTTP_TTS_MODELS

    def test_synthesis_route_is_composed(self) -> None:
        settings = PluginSettings.from_dict({"base_url": "https://example.com/api/v1"})
        assert settings.synthesis_route().endswith(
            "/api/v1/services/aigc/multimodal-generation/generation",
        )


class TestValidation:
    def test_missing_api_key_is_reported_with_a_hint(self) -> None:
        settings = PluginSettings.from_dict({})
        with pytest.raises(TtsConfigError) as excinfo:
            settings.validate_for_synthesis()
        assert "API Key" in excinfo.value.user_message()

    def test_missing_model_is_reported(self) -> None:
        settings = PluginSettings.from_dict({"api_key": "sk-x", "model": " "})
        # An empty model falls back to the default, so force it explicitly.
        object.__setattr__(settings, "model", "")
        with pytest.raises(TtsConfigError):
            settings.validate_for_synthesis()

    def test_complete_config_passes(self) -> None:
        PluginSettings.from_dict({"api_key": "sk-x"}).validate_for_synthesis()

    def test_with_voice_copies_without_mutating(self) -> None:
        settings = PluginSettings.from_dict({})
        other = settings.with_voice("Ethan")
        assert other.voice == "Ethan"
        assert settings.voice == "Cherry"
        assert settings.with_voice("") is settings
