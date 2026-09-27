"""Shared pytest fixtures for the wake-word TTS plugin test suite.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests import astrbot_stub, mock_dashscope, mock_llm  # noqa: E402

#: Module name AstrBot uses when it loads a plugin directory. Reused here so the
#: tests import ``main.py`` exactly the way production does (relative imports
#: included) instead of as a bare top-level module.
PLUGIN_DIR_NAME = REPO_ROOT.name
PLUGIN_MODULE = f"data.plugins.{PLUGIN_DIR_NAME}.main"
PLUGIN_MODULE_PREFIX = f"data.plugins.{PLUGIN_DIR_NAME}"


@dataclass
class PluginHarness:
    """A loaded plugin instance plus everything needed to drive it."""

    module: Any
    plugin: Any
    filters: Any
    runtime: Any
    data_dir: Path
    audio_dir: Path

    def new_event(self, message_str: str, **kwargs: Any) -> Any:
        return astrbot_stub.AstrMessageEvent(message_str=message_str, **kwargs)

    async def drive(self, handler: Any, event: Any) -> list[Any]:
        """Collect every non-empty result an async-generator handler yields."""
        results: list[Any] = []
        async for item in handler(event):
            if item is not None:
                results.append(item)
        return results


def texts_of(results: list[Any]) -> list[str]:
    """Flatten plain-text results into a list of strings."""
    collected: list[str] = []
    for result in results:
        for component in getattr(result, "chain", []) or []:
            text = getattr(component, "text", None)
            if text:
                collected.append(str(text))
    return collected


def records_of(results: list[Any]) -> list[Any]:
    """Return every ``Record`` component across the given results."""
    collected: list[Any] = []
    for result in results:
        for component in getattr(result, "chain", []) or []:
            if type(component).__name__ == "Record":
                collected.append(component)
    return collected


@pytest.fixture(autouse=True)
def _quiet_logging() -> Any:
    logging.getLogger(astrbot_stub.STUB_LOGGER_NAME).setLevel(logging.CRITICAL)
    yield
    logging.getLogger(astrbot_stub.STUB_LOGGER_NAME).setLevel(logging.NOTSET)


@pytest_asyncio.fixture
async def llm_server() -> Any:
    """A running fake OpenAI-compatible chat endpoint."""
    state = mock_llm.MockLlmState()
    server = await mock_llm.start(state)
    try:
        yield server
    finally:
        await server.aclose()


@pytest_asyncio.fixture
async def dashscope() -> Any:
    """A running fake DashScope endpoint."""
    state = mock_dashscope.MockDashScopeState()
    server = await mock_dashscope.start(state)
    try:
        yield server
    finally:
        await mock_dashscope.stop(server)


@pytest_asyncio.fixture
async def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load ``main.py`` through AstrBot's own import path.

    Yields a factory so a test can build as many plugin instances as it needs
    with different configurations. Modules are purged before every import, which
    mirrors AstrBot's own ``_purge_modules`` and keeps the stub binding fresh.
    """
    data_dir = tmp_path / "data"
    env = astrbot_stub.install(data_dir, wake_prefix=["/"])
    monkeypatch.setenv("ASTRBOT_ROOT", str(tmp_path))

    # Mirror the real layout: <root>/data/plugins/<plugin_dir>/main.py. A symlink
    # keeps the test on the actual source tree instead of a stale copy.
    plugin_link = data_dir / "plugins" / PLUGIN_DIR_NAME
    plugin_link.parent.mkdir(parents=True, exist_ok=True)
    if not plugin_link.exists():
        plugin_link.symlink_to(REPO_ROOT, target_is_directory=True)

    for name in list(sys.modules):
        if name == "data" or name.startswith("data."):
            del sys.modules[name]

    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(PLUGIN_MODULE)
    finally:
        sys.path.remove(str(tmp_path))

    created: list[Any] = []

    def factory(config: dict[str, Any] | None = None, **context_config: Any) -> PluginHarness:
        context = astrbot_stub.Context(
            {"wake_prefix": ["/"], **context_config},
            env.runtime,
        )
        plugin = module.WakeWordTtsPlugin(context, astrbot_stub.AstrBotConfig(config or {}))
        created.append(plugin)
        return PluginHarness(
            module=module,
            plugin=plugin,
            filters=env.filters,
            runtime=env.runtime,
            data_dir=data_dir,
            audio_dir=data_dir / "plugin_data" / PLUGIN_DIR_NAME / "audio",
        )

    try:
        yield factory
    finally:
        # Run the real AstrBot lifecycle teardown so background pruning tasks
        # and HTTP sessions never leak between tests.
        for plugin in created:
            try:
                await plugin.terminate()
            except Exception:  # pragma: no cover - best-effort cleanup
                service = getattr(plugin, "_service", None)
                if service is not None:
                    await service.aclose()


@pytest.fixture
def wav_bytes() -> bytes:
    return mock_dashscope.make_wav()
