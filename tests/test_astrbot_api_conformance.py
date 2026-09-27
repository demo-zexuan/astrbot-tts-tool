"""Conformance checks against a real AstrBot checkout.

I. Why this test exists

1. The rest of the suite runs against ``tests/astrbot_stub.py``. A stub can drift
   away from the real API, and that drift is exactly what breaks a plugin on a
   user's machine.
2. This module therefore reads a real AstrBot source tree — when one is
   available — and asserts that every symbol, signature and schema type the
   plugin relies on still exists.
3. The test is skipped (not failed) when no checkout is present, so the suite
   stays runnable anywhere.

Point it at a checkout with ``ASTRBOT_SOURCE=/path/to/AstrBot``.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

CANDIDATE_ROOTS = (
    os.environ.get("ASTRBOT_SOURCE", ""),
    str(REPO_ROOT.parent / "AstrBot"),
    str(Path.home() / "Content" / "AstrBot"),
    str(Path.home() / "AstrBot"),
)

SCHEMA_PATH = REPO_ROOT / "_conf_schema.json"
METADATA_PATH = REPO_ROOT / "metadata.yaml"
MAIN_PATH = REPO_ROOT / "main.py"


def find_astrbot_root() -> Path | None:
    for candidate in CANDIDATE_ROOTS:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "astrbot" / "api" / "__init__.py").is_file():
            return root
    return None


ASTRBOT_ROOT = find_astrbot_root()

pytestmark = pytest.mark.skipif(
    ASTRBOT_ROOT is None,
    reason="no AstrBot checkout found; set ASTRBOT_SOURCE to enable these checks",
)


def read(relative: str) -> str:
    assert ASTRBOT_ROOT is not None
    return (ASTRBOT_ROOT / relative).read_text(encoding="utf-8")


def astrbot_version() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', read("pyproject.toml"), re.MULTILINE)
    return match.group(1) if match else "unknown"


class TestImportedSymbols:
    """Everything ``main.py`` imports must exist in the checkout."""

    def test_api_logger_and_config(self) -> None:
        source = read("astrbot/api/__init__.py")
        assert '"logger"' in source or "logger =" in source
        assert "AstrBotConfig" in source

    def test_event_exports(self) -> None:
        source = read("astrbot/api/event/__init__.py")
        assert "AstrMessageEvent" in source

    def test_filter_exports(self) -> None:
        source = read("astrbot/api/event/filter/__init__.py")
        assert "register_command as command" in source
        assert "register_event_message_type as event_message_type" in source
        assert "EventMessageType" in source

    def test_event_message_type_has_all(self) -> None:
        source = read("astrbot/core/star/filter/event_message_type.py")
        assert "ALL = GROUP_MESSAGE | PRIVATE_MESSAGE | OTHER_MESSAGE" in source

    def test_star_exports(self) -> None:
        source = read("astrbot/api/star/__init__.py")
        assert "Context" in source and "Star" in source

    def test_message_components_reexport(self) -> None:
        source = read("astrbot/api/message_components.py")
        assert "from astrbot.core.message.components import *" in source


class TestRecordComponent:
    """``Record`` is how the synthesized audio reaches the user."""

    def test_record_class_and_helpers(self) -> None:
        source = read("astrbot/core/message/components.py")
        assert "class Record(BaseMessageComponent):" in source
        assert "def fromFileSystem(path, **_):" in source
        assert "def fromURL(url: str, **_):" in source

    def test_record_accepts_a_plain_file_path(self) -> None:
        # ``_resolve_file_source`` prefers ``os.path.exists(self.file)``, so a
        # plain absolute path is a valid fallback when fromFileSystem is absent.
        source = read("astrbot/core/message/components.py")
        assert "os.path.exists(self.file)" in source


class TestEventApi:
    """The event methods the plugin calls on every message."""

    REQUIRED = (
        "def plain_result(",
        "def chain_result(",
        "def stop_event(",
        "def is_stopped(",
        "def should_call_llm(",
        "def get_messages(",
        "def is_admin(",
        "def get_session_id(",
        "def get_sender_id(",
        "def get_sender_name(",
        "def unified_msg_origin(",
    )

    def test_all_methods_exist(self) -> None:
        source = read("astrbot/core/platform/astr_message_event.py")
        missing = [name for name in self.REQUIRED if name not in source]
        assert missing == [], f"AstrBot {astrbot_version()} is missing {missing}"

    def test_stop_event_survives_a_later_set_result(self) -> None:
        # main.py yields the audio and then calls stop_event(); set_result must
        # not clear the forced-stop flag or the LLM would answer twice.
        source = read("astrbot/core/platform/astr_message_event.py")
        assert "def set_result(" in source
        assert "self._force_stopped = True" in source
        assert "if self._force_stopped:" in source


class TestPluginLoading:
    """How AstrBot discovers and instantiates the plugin class."""

    def test_plugins_are_imported_as_data_plugins_packages(self) -> None:
        source = read("astrbot/core/star/star_manager.py")
        assert 'path = "data.plugins."' in source
        assert "__import__(path, fromlist=[module_str])" in source

    def test_instantiation_passes_context_and_config(self) -> None:
        source = read("astrbot/core/star/star_manager.py")
        assert "context=self.context" in source
        assert "config=plugin_config" in source

    def test_class_name_discovery_rule(self) -> None:
        # _get_classes keeps classes ending in "plugin" (case-insensitive).
        source = read("astrbot/core/star/star_manager.py")
        assert 'name.lower().endswith("plugin")' in source
        assert "class WakeWordTtsPlugin" in MAIN_PATH.read_text(encoding="utf-8")

    def test_initialize_and_terminate_are_awaited(self) -> None:
        source = read("astrbot/core/star/star_manager.py")
        assert "await metadata.star_cls.initialize()" in source
        assert "await star_metadata.star_cls.terminate()" in source


class TestLlmProviderApi:
    """The director stage reuses AstrBot's provider system; these are the hooks."""

    def test_provider_registry_accessors_exist(self) -> None:
        source = read("astrbot/core/star/context.py")
        assert "def get_provider_by_id(" in source
        assert "def get_using_provider(" in source
        assert "conversation_manager" in source

    def test_text_chat_accepts_the_arguments_the_adapter_passes(self) -> None:
        source = read("astrbot/core/provider/provider.py")
        for name in ("prompt", "contexts", "system_prompt", "model"):
            assert f"{name}:" in source, f"Provider.text_chat no longer accepts {name}"
        assert "**kwargs" in source

    def test_text_chat_is_stateless_for_conversation_history(self) -> None:
        # The adapter relies on this: a director call must not append to the
        # session's stored conversation.
        source = read("astrbot/core/provider/sources/openai_source.py")
        assert "self.conversation" not in source
        assert "context_query = copy.deepcopy(self._ensure_message_to_dicts(contexts))" in source

    def test_completion_text_is_available_on_the_response(self) -> None:
        source = read("astrbot/core/provider/entities.py")
        assert "class LLMResponse:" in source
        assert "completion_text" in source

    def test_custom_headers_reach_the_http_client(self) -> None:
        """This is how an Opencode-style gateway gets its required headers."""
        source = read("astrbot/core/provider/sources/openai_source.py")
        assert 'provider_config.get("custom_headers", {})' in source
        assert "default_headers=self.custom_headers" in source

    def test_anthropic_style_providers_also_support_custom_headers(self) -> None:
        source = read("astrbot/core/provider/sources/anthropic_source.py")
        assert "custom_headers" in source

    def test_conversation_manager_offers_history_lookup(self) -> None:
        source = read("astrbot/core/conversation_mgr.py")
        assert "async def get_curr_conversation_id(" in source
        assert "async def get_conversation(" in source

    def test_history_is_stored_as_json_text(self) -> None:
        # ``_conversation_history`` parses ``conversation.history`` with json.
        source = read("astrbot/core/utils/history_saver.py")
        assert 'json.loads(req.conversation.history or "[]")' in source

    def test_select_provider_is_a_supported_special_config(self) -> None:
        source = read("astrbot/core/config/default.py")
        assert '"select_provider"' in source


class TestSchemaConformance:
    """Every config type used must be one AstrBot can parse."""

    @staticmethod
    def supported_types() -> set[str]:
        source = read("astrbot/core/config/default.py")
        match = re.search(r"DEFAULT_VALUE_MAP = \{(.*?)\n\}", source, re.DOTALL)
        assert match, "DEFAULT_VALUE_MAP not found"
        return set(re.findall(r'"([a-z_]+)":', match.group(1)))

    @staticmethod
    def collect_types(node: object) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            type_name = node.get("type")
            if isinstance(type_name, str):
                found.add(type_name)
            for value in node.values():
                found |= TestSchemaConformance.collect_types(value)
        elif isinstance(node, list):
            for item in node:
                found |= TestSchemaConformance.collect_types(item)
        return found

    def test_schema_is_valid_json(self) -> None:
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_only_supported_types_are_used(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        used = self.collect_types(schema)
        unsupported = used - self.supported_types()
        assert unsupported == set(), f"unsupported schema types: {unsupported}"

    def test_template_list_entries_declare_items(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        profiles = schema["wake_profiles"]
        assert profiles["type"] == "template_list"
        for template in profiles["templates"].values():
            assert "items" in template
            for item in template["items"].values():
                assert item["type"] in self.supported_types()

    def test_every_option_has_a_matching_label(self) -> None:
        # The WebUI pairs options[i] with labels[i]; a mismatch renders blank
        # entries in the dropdown.
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        def walk(node: object, path: str = "") -> list[str]:
            problems: list[str] = []
            if isinstance(node, dict):
                options = node.get("options")
                labels = node.get("labels")
                if (
                    isinstance(options, list)
                    and isinstance(labels, list)
                    and len(options) != len(labels)
                ):
                    problems.append(f"{path}: {len(options)} options vs {len(labels)} labels")
                for key, value in node.items():
                    problems += walk(value, f"{path}.{key}" if path else key)
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    problems += walk(value, f"{path}[{index}]")
            return problems

        assert walk(schema) == []

    def test_config_keys_match_what_the_parser_reads(self) -> None:
        """Every schema key should be consumed by ``PluginSettings.from_dict``."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        parser_source = (REPO_ROOT / "core" / "config.py").read_text(encoding="utf-8")
        keys = set(re.findall(r'raw\.get\("([a-z_]+)"\)', parser_source))
        unread = sorted(set(schema) - keys)
        assert unread == [], f"schema keys never read by the parser: {unread}"


class TestMetadataConformance:
    REQUIRED_KEYS = ("name", "desc", "version", "author", "repo")

    def metadata(self) -> dict[str, str]:
        text = METADATA_PATH.read_text(encoding="utf-8")
        found: dict[str, str] = {}
        for key, value in re.findall(r"^([a-z_]+):\s*(.+)$", text, re.MULTILINE):
            found[key] = value.strip()
        return found

    def test_required_keys_present(self) -> None:
        metadata = self.metadata()
        missing = [key for key in self.REQUIRED_KEYS if not metadata.get(key)]
        assert missing == []

    def test_metadata_documents_the_llm_default(self) -> None:
        text = METADATA_PATH.read_text(encoding="utf-8")
        assert "qwen3-tts-flash" in text

    def test_plugin_name_follows_the_convention(self) -> None:
        name = self.metadata()["name"]
        assert name.startswith("astrbot_plugin_")
        assert " " not in name and name == name.lower()

    def test_declared_version_range_accepts_the_checkout(self) -> None:
        metadata = self.metadata()
        spec = metadata.get("astrbot_version", "")
        version = astrbot_version()
        if not spec or version == "unknown":
            pytest.skip("no version specifier declared")
        # Strip the quotes YAML keeps in the raw text.
        spec = spec.strip('"')
        match = re.match(r">=\s*(\d+)\.(\d+)\.(\d+)", spec)
        if not match:
            pytest.skip(f"unsupported specifier form: {spec}")
        minimum = tuple(int(part) for part in match.groups())
        current = tuple(int(part) for part in version.split(".")[:3])
        assert current >= minimum, f"AstrBot {version} does not satisfy {spec}"

    def test_support_platforms_are_known_adapter_names(self) -> None:
        text = METADATA_PATH.read_text(encoding="utf-8")
        block = re.search(
            r"^support_platforms:\s*\n((?:\s+-\s+\S+\n)+)",
            text,
            re.MULTILINE,
        )
        if not block:
            pytest.skip("no support_platforms declared")
        platforms = re.findall(r"-\s+(\S+)", block.group(1))
        if not platforms:
            pytest.skip("support_platforms is empty")
        source = read("astrbot/core/star/filter/platform_adapter_type.py")
        match = re.search(r"ADAPTER_NAME_2_TYPE = \{(.*?)\n\}", source, re.DOTALL)
        assert match, "ADAPTER_NAME_2_TYPE not found"
        known = set(re.findall(r'"([a-z_]+)":', match.group(1)))
        unknown = [name for name in platforms if name not in known]
        assert unknown == [], f"unknown support_platforms entries: {unknown}"


class TestSourceHygiene:
    """Cheap static guards that keep the plugin importable by AstrBot."""

    def test_main_module_has_no_import_time_side_effects(self) -> None:
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        allowed = (
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Assign,
            ast.AnnAssign,
            ast.Expr,
        )
        for node in tree.body:
            assert isinstance(node, allowed), (
                f"unexpected top-level statement: {ast.dump(node)[:80]}"
            )

    def test_no_third_party_imports_beyond_aiohttp(self) -> None:
        """The core package must stay dependency-free for unit testing."""
        allowed = {
            "asyncio",
            "base64",
            "binascii",
            "contextlib",
            "dataclasses",
            "enum",
            "hashlib",
            "json",
            "logging",
            "math",
            "os",
            "pathlib",
            "random",
            "re",
            "struct",
            "sys",
            "time",
            "types",
            "typing",
            "__future__",
            # Lazily imported inside the client so ``core`` stays importable
            # (and unit testable) without aiohttp installed.
            "aiohttp",
        }
        offenders: list[str] = []
        for path in sorted((REPO_ROOT / "core").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root not in allowed:
                            offenders.append(f"{path.name}: {root}")
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    root = node.module.split(".")[0]
                    if root not in allowed:
                        offenders.append(f"{path.name}: {root}")
        assert offenders == [], f"unexpected imports in core/: {offenders}"

    def test_core_never_imports_astrbot(self) -> None:
        for path in sorted((REPO_ROOT / "core").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "import astrbot" not in source, f"{path.name} couples core to AstrBot"
