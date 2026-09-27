"""A local HTTP server that behaves like an OpenAI-compatible chat endpoint.

Used to exercise the direct LLM mode for real: request shape, header handling
(the part that breaks against gateways), retries and error mapping.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aiohttp import web


@dataclass
class MockLlmState:
    """Mutable knobs controlling the fake endpoint."""

    #: Reply content returned in ``choices[0].message.content``.
    reply: str = (
        '{"text": "改写后的台词", "instructions": "温柔一点", "tags": [], "reason": "加了语气词"}'
    )
    #: Number of initial calls answered with HTTP 429.
    rate_limit_times: int = 0
    #: When set, answer with this HTTP status and an error body.
    error_status: int = 200
    error_body: dict[str, Any] | None = None
    #: Raw body override (e.g. HTML from a broken gateway).
    raw_body: str | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)


class MockLlm:
    """Running server handle."""

    def __init__(self, state: MockLlmState, runner: web.AppRunner, port: int) -> None:
        self.state = state
        self._runner = runner
        self.port = port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def bare_base_url(self) -> str:
        """Base URL without the ``/v1`` suffix, to test URL normalisation."""
        return f"http://127.0.0.1:{self.port}"

    async def aclose(self) -> None:
        await self._runner.cleanup()


async def start(state: MockLlmState | None = None) -> MockLlm:
    """Start the fake endpoint on an ephemeral localhost port."""
    state = state or MockLlmState()

    async def handle(request: web.Request) -> web.Response:
        state.paths.append(request.path)
        state.headers.append(dict(request.headers))
        state.requests.append(await request.json())

        if state.raw_body is not None:
            return web.Response(text=state.raw_body, content_type="application/json")
        if state.error_status != 200:
            return web.json_response(
                state.error_body
                or {"error": {"message": "bad key", "type": "authentication_error"}},
                status=state.error_status,
            )
        if state.rate_limit_times > 0:
            state.rate_limit_times -= 1
            return web.json_response(
                {"error": {"message": "slow down", "code": "rate_limit_exceeded"}},
                status=429,
            )
        return web.json_response(
            {
                "id": f"chatcmpl-{len(state.requests)}",
                "object": "chat.completion",
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": state.reply},
                        "finish_reason": "stop",
                    },
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_post("/chat/completions", handle)
    app.router.add_post("/{tail:.*}", handle)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    return MockLlm(state, runner, sockets[0].getsockname()[1])
