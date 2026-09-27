"""A minimal in-memory stand-in for the ``astrbot`` package.

I. Why this exists

1. The plugin must be exercised end-to-end (wake word -> HTTP -> audio file ->
   message chain) without a running AstrBot and without real network access.
2. Installing AstrBot to run tests is impractical: it needs Python >= 3.12 and
   around a hundred dependencies including ``faiss-cpu`` and ``pandas``.
3. The shapes below mirror the real signatures verified in AstrBot v4.27.4
   (``astrbot/api``, ``astrbot/core/message/components.py``,
   ``astrbot/core/platform/astr_message_event.py``). The companion test
   ``test_astrbot_api_conformance.py`` re-checks those signatures against a real
   AstrBot checkout when one is present, so the stub cannot drift silently.

II. What is emulated

1. ``astrbot.api`` – ``logger``, ``AstrBotConfig``.
2. ``astrbot.api.event`` – ``filter`` decorators and ``filter.EventMessageType``.
3. ``astrbot.api.message_components`` – ``Plain``, ``Record``.
4. ``astrbot.api.star`` – ``Context``, ``Star``.
5. ``astrbot.core.utils.astrbot_path`` – ``get_astrbot_data_path``.
6. ``Context.get_provider_by_id`` / ``get_using_provider`` plus a fake chat
   provider, so the LLM director stage can be driven end to end.
7. ``Context.conversation_manager``, so context awareness can be tested without
   a database.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import enum
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STUB_LOGGER_NAME = "astrbot"


# ------------------------------------------------------------- components ---


class ComponentType(str, enum.Enum):  # noqa: UP042 - mirrors AstrBot's own class
    """Subset of the real component enum (``class ComponentType(str, Enum)``)."""

    Plain = "Plain"
    Record = "Record"
    Reply = "Reply"


class BaseMessageComponent:
    type: ComponentType

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class Plain(BaseMessageComponent):
    type = ComponentType.Plain

    def __init__(self, text: str = "") -> None:
        super().__init__(text=text)
        self.text = text


class Record(BaseMessageComponent):
    """Mirrors ``astrbot.core.message.components.Record``."""

    type = ComponentType.Record

    def __init__(self, file: str | None = "", **kwargs: Any) -> None:
        super().__init__(file=file, **kwargs)
        self.file = file
        self.url = kwargs.get("url", "")
        self.path = kwargs.get("path")
        self.text = kwargs.get("text")

    @staticmethod
    def fromFileSystem(path: str, **kwargs: Any) -> Record:
        resolved = Path(path).resolve(strict=False)
        return Record(file=resolved.as_uri(), path=str(resolved), **kwargs)

    @staticmethod
    def fromURL(url: str, **kwargs: Any) -> Record:
        if not url.startswith(("http://", "https://")):
            raise ValueError("not a valid url")
        return Record(file=url, **kwargs)


class Reply(BaseMessageComponent):
    """Mirrors the fields of the real ``Reply`` used by this plugin."""

    type = ComponentType.Reply

    def __init__(self, message_str: str = "", **kwargs: Any) -> None:
        super().__init__(message_str=message_str, **kwargs)
        self.message_str = message_str


# ------------------------------------------------------------------ LLM ---


class FakeLlmResponse:
    """Mirrors ``astrbot.core.provider.entities.LLMResponse`` where it matters."""

    def __init__(self, text: str) -> None:
        self.role = "assistant"
        self.completion_text = text
        self.usage: dict[str, Any] = {}


class FakeProvider:
    """A chat provider double that records how it was called."""

    def __init__(
        self,
        name: str = "fake",
        model: str = "deepseek-chat",
        reply: str = '{"text": "改写后的台词"}',
    ) -> None:
        self.name = name
        self.model = model
        self.reply = reply
        self.reply_factory: Any | None = None
        self.raise_error: Exception | None = None
        #: When set, ``text_chat`` is replaced by a coroutine exposing only
        #: ``prompt``, mirroring older/minimal provider implementations.
        self.minimal_signature = False
        self.calls: list[dict[str, Any]] = []

    def get_model(self) -> str:
        return self.model

    async def text_chat(self, **kwargs: Any) -> FakeLlmResponse:
        self.calls.append(kwargs)
        if self.raise_error is not None:
            raise self.raise_error
        if self.reply_factory is not None:
            return FakeLlmResponse(str(self.reply_factory(kwargs)))
        return FakeLlmResponse(self.reply)


class FakeConversation:
    def __init__(self, history: str = "[]") -> None:
        self.history = history


class FakeConversationManager:
    """Enough of ``ConversationManager`` for context-awareness tests."""

    def __init__(self, history: str = "[]") -> None:
        self.history = history
        self.calls: list[tuple[str, Any]] = []

    async def get_curr_conversation_id(self, unified_msg_origin: str) -> str | None:
        self.calls.append(("get_curr_conversation_id", unified_msg_origin))
        return "cid-1" if self.history is not None else None

    async def get_conversation(
        self,
        unified_msg_origin: str,
        conversation_id: str,
        create_if_not_exists: bool = False,
    ) -> FakeConversation | None:
        self.calls.append(("get_conversation", conversation_id))
        if self.history is None:
            return None
        return FakeConversation(self.history)


class StubRuntime:
    """Mutable state shared by the stub ``Context`` and the test body."""

    def __init__(self) -> None:
        self.providers: dict[str, FakeProvider] = {}
        self.using_provider: FakeProvider | None = None
        self.conversation_manager = FakeConversationManager()
        self.expose_conversation_manager = True


# ---------------------------------------------------------------- results ---


class MessageEventResult:
    """Captures what a handler asked AstrBot to send."""

    def __init__(self, chain: list[Any] | None = None) -> None:
        self.chain: list[Any] = chain or []
        self.stopped = False

    def stop_event(self) -> MessageEventResult:
        self.stopped = True
        return self

    def continue_event(self) -> MessageEventResult:
        self.stopped = False
        return self

    def is_stopped(self) -> bool:
        return self.stopped


class MessageChain:
    def __init__(self, chain: list[Any] | None = None) -> None:
        self.chain = chain or []


class AstrBotMessage:
    """Mirrors the fields the plugin reads from ``event.message_obj``."""

    def __init__(
        self,
        *,
        message_str: str = "",
        group_id: str = "",
        message_id: str = "1",
        self_id: str = "10000",
        session_id: str = "session",
        sender_id: str = "20000",
        sender_name: str = "tester",
        messages: list[Any] | None = None,
    ) -> None:
        self.message_str = message_str
        self.group_id = group_id
        self.message_id = message_id
        self.self_id = self_id
        self.session_id = session_id
        self.message = messages or []
        self.sender = types.SimpleNamespace(user_id=sender_id, nickname=sender_name)


class MessageType(enum.Enum):
    GROUP_MESSAGE = "GroupMessage"
    FRIEND_MESSAGE = "FriendMessage"
    OTHER_MESSAGE = "OtherMessage"


class AstrMessageEvent:
    """Test double for ``astrbot.core.platform.astr_message_event``."""

    def __init__(
        self,
        *,
        message_str: str = "",
        message_obj: AstrBotMessage | None = None,
        unified_msg_origin: str = "test:FriendMessage:20000",
        admin: bool = False,
        messages: list[Any] | None = None,
    ) -> None:
        self.message_str = message_str
        self.message_obj = message_obj or AstrBotMessage(message_str=message_str)
        self.unified_msg_origin = unified_msg_origin
        self.is_at_or_wake_command = False
        self.is_wake = False
        self.call_llm = True
        self._admin = admin
        self._messages = messages if messages is not None else self.message_obj.message
        self._result: MessageEventResult | None = None
        self._force_stopped = False

    # -- reads -------------------------------------------------------------

    def get_message_str(self) -> str:
        return self.message_str

    def get_session_id(self) -> str:
        return self.message_obj.session_id

    def get_group_id(self) -> str:
        return self.message_obj.group_id

    def get_sender_id(self) -> str:
        return str(self.message_obj.sender.user_id)

    def get_sender_name(self) -> str:
        return str(self.message_obj.sender.nickname)

    def get_platform_name(self) -> str:
        return "aiocqhttp"

    def get_messages(self) -> list[Any]:
        return list(self._messages)

    def get_message_type(self) -> MessageType:
        return (
            MessageType.GROUP_MESSAGE if self.message_obj.group_id else MessageType.FRIEND_MESSAGE
        )

    def is_private_chat(self) -> bool:
        return not self.message_obj.group_id

    def is_admin(self) -> bool:
        return self._admin

    # -- writes ------------------------------------------------------------

    def set_result(self, result: MessageEventResult | str) -> None:
        if isinstance(result, str):
            result = MessageEventResult([Plain(result)])
        self._result = result

    def get_result(self) -> MessageEventResult | None:
        return self._result

    def clear_result(self) -> None:
        self._result = None

    def stop_event(self) -> None:
        self._force_stopped = True
        if self._result is None:
            self.set_result(MessageEventResult().stop_event())
        else:
            self._result.stop_event()

    def continue_event(self) -> None:
        self._force_stopped = False

    def is_stopped(self) -> bool:
        if self._force_stopped:
            return True
        return bool(self._result and self._result.is_stopped())

    def should_call_llm(self, call_llm: bool) -> None:
        self.call_llm = call_llm

    def plain_result(self, text: str) -> MessageEventResult:
        return MessageEventResult([Plain(text)])

    def chain_result(self, chain: list[Any]) -> MessageEventResult:
        return MessageEventResult(list(chain))

    async def send(self, message: Any) -> None:
        self.set_result(message)


# ------------------------------------------------------------------ star ---


class Context:
    """The slice of ``astrbot.core.star.context.Context`` the plugin touches."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        runtime: StubRuntime | None = None,
    ) -> None:
        self._config = config or {}
        self._runtime = runtime or StubRuntime()
        if self._runtime.expose_conversation_manager:
            self.conversation_manager = self._runtime.conversation_manager

    def get_config(self) -> dict[str, Any]:
        return self._config

    def get_provider_by_id(self, provider_id: str) -> FakeProvider | None:
        return self._runtime.providers.get(provider_id)

    def get_using_provider(self, umo: str | None = None) -> FakeProvider | None:
        return self._runtime.using_provider


class Star:
    """Mirrors ``astrbot.core.star.base.Star``."""

    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        self.context = context
        self.logger = logging.getLogger(f"{STUB_LOGGER_NAME}.plugin.test")

    async def initialize(self) -> None: ...

    async def terminate(self) -> None: ...


class AstrBotConfig(dict):
    """``AstrBotConfig`` is a ``dict`` subclass in AstrBot."""


# ----------------------------------------------------------------- filter ---


class EventMessageType(enum.Flag):
    GROUP_MESSAGE = enum.auto()
    PRIVATE_MESSAGE = enum.auto()
    OTHER_MESSAGE = enum.auto()
    ALL = GROUP_MESSAGE | PRIVATE_MESSAGE | OTHER_MESSAGE


class _FilterRegistry:
    """Records decorator usage so tests can assert on registration."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, set[str]]] = []
        self.event_message_types: list[EventMessageType] = []

    def reset(self) -> None:
        self.commands.clear()
        self.event_message_types.clear()


class FilterModule:
    """Emulates ``astrbot.api.event.filter``."""

    EventMessageType = EventMessageType

    def __init__(self) -> None:
        self.registry = _FilterRegistry()

    def command(self, name: str, alias: set[str] | None = None, **_: Any):
        def decorator(func: Any) -> Any:
            self.registry.commands.append((name, set(alias or set())))
            func.__astrbot_command__ = name
            return func

        return decorator

    def command_group(self, name: str, **_: Any):
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def event_message_type(self, event_message_type: EventMessageType = EventMessageType.ALL):
        def decorator(func: Any) -> Any:
            self.registry.event_message_types.append(event_message_type)
            func.__astrbot_event_message_type__ = event_message_type
            return func

        return decorator

    def permission_type(self, *_args: Any, **_kwargs: Any):
        def decorator(func: Any) -> Any:
            return func

        return decorator


# ----------------------------------------------------------------- install ---


@dataclass
class StubEnv:
    """Handles returned by :func:`install`."""

    filters: FilterModule
    runtime: StubRuntime


def install(
    data_dir: Path | str,
    *,
    wake_prefix: list[str] | None = None,
    runtime: StubRuntime | None = None,
) -> StubEnv:
    """Register the stub package in ``sys.modules`` and return its handles.

    ``data_dir`` is what ``get_astrbot_data_path()`` will report, so the plugin
    writes its cache under a temporary directory during tests. ``runtime`` lets a
    test pre-seed fake LLM providers and conversation history.
    """
    filter_module = FilterModule()
    data_path = str(data_dir)

    root = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger(STUB_LOGGER_NAME)
    api.AstrBotConfig = AstrBotConfig

    event = types.ModuleType("astrbot.api.event")
    event.filter = filter_module
    event.AstrMessageEvent = AstrMessageEvent

    components = types.ModuleType("astrbot.api.message_components")
    components.Plain = Plain
    components.Record = Record
    components.Reply = Reply

    star = types.ModuleType("astrbot.api.star")
    star.Context = Context
    star.Star = Star

    core = types.ModuleType("astrbot.core")
    core_utils = types.ModuleType("astrbot.core.utils")
    astrbot_path = types.ModuleType("astrbot.core.utils.astrbot_path")

    def get_astrbot_data_path() -> str:
        return data_path

    astrbot_path.get_astrbot_data_path = get_astrbot_data_path

    root.api = api
    modules = {
        "astrbot": root,
        "astrbot.api": api,
        "astrbot.api.event": event,
        "astrbot.api.message_components": components,
        "astrbot.api.star": star,
        "astrbot.core": core,
        "astrbot.core.utils": core_utils,
        "astrbot.core.utils.astrbot_path": astrbot_path,
    }
    sys.modules.update(modules)
    return StubEnv(filters=filter_module, runtime=runtime or StubRuntime())
