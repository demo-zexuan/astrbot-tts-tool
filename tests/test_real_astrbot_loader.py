"""Integration tests against a **real** installed AstrBot.

I. Why these exist

1. Every other test in this suite runs against ``tests/astrbot_stub.py`` or reads
   AstrBot's source as text. Neither proves that the plugin *imports and
   registers* under the real framework.
2. This module is skipped unless ``astrbot`` is importable, so it never affects
   the default offline run. Install it with::

       python -m venv .venv-astrbot && . .venv-astrbot/bin/activate
       pip install /path/to/AstrBot
       pytest tests/test_real_astrbot_loader.py -v

3. What it verifies, in decreasing order of "would actually break a user":
   - ``main.py`` imports under the real ``data.plugins.<dir>.main`` module name,
     so every ``from astrbot...`` import the plugin makes really exists;
   - the decorators register the commands and the wake-word listener in
     AstrBot's real handler registry;
   - ``_conf_schema.json`` is accepted by the real ``AstrBotConfig`` (an
     unsupported ``type`` raises there) and the generated defaults parse cleanly
     through ``PluginSettings.from_dict``;
   - ``metadata.yaml`` is parsed by AstrBot's own loader;
   - a WAV written by this plugin survives AstrBot's real media pipeline
     (``Record.convert_to_base64``), which is the exact path used when a voice
     message is sent to QQ/Telegram.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

from tests.conftest import PLUGIN_DIR_NAME, PLUGIN_MODULE, REPO_ROOT
from tests.test_wav import with_placeholder_sizes


def _real_astrbot_version() -> str:
    """Version of a *real* installed AstrBot, or ``""``.

    ``tests/astrbot_stub.py`` installs a fake ``astrbot`` package into
    ``sys.modules`` when the ``harness`` fixture runs. That stub has no
    ``__version__``, so it can be told apart from the real distribution — which
    matters because the stub would otherwise shadow the real modules and make
    these tests pass vacuously.
    """
    module = sys.modules.get("astrbot")
    if module is None:
        try:
            module = importlib.import_module("astrbot")
        except Exception:
            return ""
    return str(getattr(module, "__version__", "") or "")


REAL_ASTRBOT_VERSION = _real_astrbot_version()

pytestmark = pytest.mark.skipif(
    not REAL_ASTRBOT_VERSION,
    reason=(
        "a real AstrBot is not installed (or the in-session test stub has taken "
        "over sys.modules); run `pytest tests/test_real_astrbot_loader.py` on its "
        "own after `pip install /path/to/AstrBot`"
    ),
)


@pytest.fixture(scope="module")
def loaded_plugin() -> object:
    """Import ``main.py`` the way AstrBot's PluginManager does."""
    import tempfile

    if not _real_astrbot_version():  # pragma: no cover - guarded by pytestmark
        pytest.skip("the real astrbot package is not importable in this session")

    workdir = Path(tempfile.mkdtemp(prefix="astrbot-plugin-load-"))
    plugin_link = workdir / "data" / "plugins" / PLUGIN_DIR_NAME
    plugin_link.parent.mkdir(parents=True, exist_ok=True)
    plugin_link.symlink_to(REPO_ROOT, target_is_directory=True)

    for name in list(sys.modules):
        if name == "data" or name.startswith("data."):
            del sys.modules[name]

    sys.path.insert(0, str(workdir))
    try:
        return importlib.import_module(PLUGIN_MODULE)
    finally:
        sys.path.remove(str(workdir))


class TestRealImport:
    """The plugin must import against the installed AstrBot, not just the stub."""

    def test_module_imports_and_exposes_the_plugin_class(self, loaded_plugin: object) -> None:
        print(f"\n[integration] loaded with real AstrBot {REAL_ASTRBOT_VERSION}")
        plugin_cls = getattr(loaded_plugin, "WakeWordTtsPlugin", None)
        assert plugin_cls is not None
        assert isinstance(plugin_cls, type)

    def test_the_class_is_registered_in_star_map(self, loaded_plugin: object) -> None:
        from astrbot.core.star.star import star_map, star_registry

        metadata = star_map.get(PLUGIN_MODULE)
        assert metadata is not None, "Star.__init_subclass__ did not register the plugin"
        assert metadata.star_cls_type is not None
        assert any(m.module_path == PLUGIN_MODULE for m in star_registry)

    def test_astrbot_can_discover_the_class_by_name(self, loaded_plugin: object) -> None:
        """AstrBot falls back to classes ending in 'Plugin' or named 'Main'."""
        from astrbot.core.star.star_manager import PluginManager

        classes = PluginManager._get_classes(loaded_plugin)
        assert classes, "PluginManager would not find the plugin class"
        assert getattr(loaded_plugin, classes[0]).__name__.endswith("Plugin")

    def test_every_symbol_the_plugin_imports_exists(self) -> None:
        """Import the exact names main.py asks for, so a rename fails loudly."""
        from astrbot.api import AstrBotConfig, logger
        from astrbot.api.event import AstrMessageEvent, filter
        from astrbot.api.message_components import Record
        from astrbot.api.star import Context, Star

        # Referencing each name keeps the import itself the assertion.
        for imported in (AstrBotConfig, logger, AstrMessageEvent, filter, Record, Context, Star):
            assert imported is not None

        for name in ("command", "event_message_type", "EventMessageType"):
            assert hasattr(filter, name), f"filter.{name} is missing"
        assert hasattr(Record, "fromFileSystem")
        assert hasattr(AstrMessageEvent, "chain_result")
        assert hasattr(AstrMessageEvent, "stop_event")
        assert hasattr(AstrMessageEvent, "should_call_llm")
        assert filter.EventMessageType.ALL is not None


class TestRealRegistration:
    """Decorators must land in the real handler registry."""

    def _handlers(self, loaded_plugin: object) -> list[object]:
        from astrbot.core.star.star_handler import star_handlers_registry

        return [
            handler
            for handler in star_handlers_registry
            if getattr(handler, "handler_module_path", "") == PLUGIN_MODULE
        ]

    def test_all_commands_are_registered(self, loaded_plugin: object) -> None:
        names: set[str] = set()
        for handler in self._handlers(loaded_plugin):
            for event_filter in getattr(handler, "event_filters", []):
                for full_name in getattr(event_filter, "get_complete_command_names", lambda: [])():
                    names.add(full_name)
        assert {"tts", "tts_voice", "tts_llm", "tts_status", "tts_help"} <= names

    def test_command_aliases_are_registered(self, loaded_plugin: object) -> None:
        names: set[str] = set()
        for handler in self._handlers(loaded_plugin):
            for event_filter in getattr(handler, "event_filters", []):
                for full_name in getattr(event_filter, "get_complete_command_names", lambda: [])():
                    names.add(full_name)
        assert {"语音", "音色", "tts_say"} <= names

    def test_the_wake_word_listener_is_registered_for_all_messages(
        self,
        loaded_plugin: object,
    ) -> None:
        from astrbot.core.star.filter.event_message_type import EventMessageType

        listener_found = False
        for handler in self._handlers(loaded_plugin):
            if handler.handler_name != "on_message":
                continue
            for event_filter in getattr(handler, "event_filters", []):
                if getattr(event_filter, "event_message_type", None) is EventMessageType.ALL:
                    listener_found = True
        assert listener_found, "the wake-word listener is not registered for EventMessageType.ALL"

    def test_handler_signatures_survive_the_real_parser(
        self,
        loaded_plugin: object,
    ) -> None:
        """Command parameters must be parseable by AstrBot's CommandFilter."""
        for handler in self._handlers(loaded_plugin):
            for event_filter in getattr(handler, "event_filters", []):
                if hasattr(event_filter, "init_handler_md") and hasattr(
                    event_filter,
                    "handler_params",
                ):
                    event_filter.init_handler_md(handler)
                    # No exception and every declared parameter resolved.
                    assert isinstance(event_filter.handler_params, dict)


class TestRealConfigSchema:
    """``_conf_schema.json`` must be accepted by the real AstrBotConfig."""

    def test_schema_builds_a_config_file_without_error(self, tmp_path: Path) -> None:
        from astrbot.core.config.astrbot_config import AstrBotConfig

        schema = json.loads((REPO_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        config_path = tmp_path / f"{PLUGIN_DIR_NAME}_config.json"

        # AstrBotConfig raises TypeError for an unsupported schema type, so this
        # doubles as a schema-type check against the installed version.
        config = AstrBotConfig(config_path=str(config_path), schema=schema)

        assert config_path.is_file()
        assert set(config.keys()) == set(schema.keys())

    def test_generated_defaults_parse_through_the_plugin_parser(self, tmp_path: Path) -> None:
        from astrbot.core.config.astrbot_config import AstrBotConfig

        from core.config import HTTP_TTS_MODELS, PluginSettings

        schema = json.loads((REPO_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        config = AstrBotConfig(config_path=str(tmp_path / "conf.json"), schema=schema)

        settings = PluginSettings.from_dict(dict(config))
        # Defaults must be immediately usable, apart from the missing API key.
        assert settings.model in HTTP_TTS_MODELS
        assert settings.voice
        assert settings.wake_words, "the default config must ship a wake word"
        assert settings.llm_enable is True
        assert settings.llm_mode == "astrbot_provider"
        assert settings.llm_tag_style == "none"

    def test_the_config_can_be_saved_and_reloaded(self, tmp_path: Path) -> None:
        from astrbot.core.config.astrbot_config import AstrBotConfig

        from core.config import PluginSettings

        schema = json.loads((REPO_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        config_path = tmp_path / "conf.json"
        config = AstrBotConfig(config_path=str(config_path), schema=schema)

        config["wake_words"] = ["小助手", "小美"]
        config["llm_model"] = "deepseek-v4-pro"
        config.save_config()

        reloaded = AstrBotConfig(config_path=str(config_path), schema=schema)
        settings = PluginSettings.from_dict(dict(reloaded))
        assert settings.all_wake_words == ("小助手", "小美")
        assert settings.llm_model == "deepseek-v4-pro"


class TestRealMetadata:
    def test_astrbot_parses_metadata_yaml(self, loaded_plugin: object) -> None:
        from astrbot.core.star.star_manager import PluginManager

        # Both helpers are staticmethods, so no PluginManager instance (and no
        # database, provider manager or platform manager) is required.
        metadata = PluginManager._load_plugin_metadata(plugin_path=str(REPO_ROOT))

        assert metadata is not None
        assert metadata.name == "astrbot_plugin_tts_wakeword"
        assert metadata.version.startswith("v")
        assert metadata.author
        assert metadata.repo.startswith("https://github.com/")
        assert metadata.support_platforms
        assert metadata.astrbot_version


class TestRealMediaPipeline:
    """The WAV this plugin writes must pass AstrBot's own audio pipeline."""

    async def test_record_from_filesystem_converts_to_base64_wav(
        self,
        tmp_path: Path,
    ) -> None:
        from astrbot.api.message_components import Record
        from astrbot.core.utils.media_utils import MediaResolver

        from core.audio_store import AudioStore

        # Store exactly like the plugin does, including the placeholder header
        # that DashScope returns, so the WAV fix is on the tested path too.
        store = AudioStore(tmp_path / "audio")
        wav = with_placeholder_sizes(_build_wav())
        path = await store.save(wav)

        record = Record.fromFileSystem(path=str(path))
        assert record.file.startswith("file://")

        # This is what the aiocqhttp adapter calls for every Record segment.
        encoded = await record.convert_to_base64()
        assert encoded, "AstrBot could not read the stored audio"
        assert len(encoded) > 100

        async with MediaResolver(str(path), media_type="audio", default_suffix=".wav").as_path(
            target_format="wav",
        ) as resolved:
            resolved_path = Path(resolved.path)
            assert resolved_path.is_file()
            header = resolved_path.read_bytes()
            assert header[:4] == b"RIFF"
            # The data size must match the payload, not the placeholder.
            assert struct.unpack_from("<I", header, 40)[0] == len(header) - 44


def _build_wav(*, rate: int = 24000, seconds: float = 0.2) -> bytes:
    import math

    frames = int(rate * seconds)
    payload = bytearray()
    for index in range(frames):
        payload += struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * index / rate)))
    data = bytes(payload)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
