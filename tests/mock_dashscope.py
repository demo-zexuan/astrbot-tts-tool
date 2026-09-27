"""A local HTTP server that behaves like the DashScope Qwen-TTS endpoint.

The server lets the test suite exercise the real ``aiohttp`` code path — request
serialization, header handling, retries, downloads and error mapping — without
touching the network or needing an API key.

Behaviour is driven by :class:`MockDashScopeState` so a single server can act as
a happy path, a rate-limited endpoint, a payload-error endpoint or a CDN that is
down.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import base64
import math
import struct
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

SYNTHESIS_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


def make_wav(*, seconds: float = 0.2, rate: int = 24000, freq: float = 440.0) -> bytes:
    """Build a valid mono 16-bit PCM WAV file."""
    frames = int(rate * seconds)
    samples = bytearray()
    for index in range(frames):
        value = int(12000 * math.sin(2 * math.pi * freq * index / rate))
        samples += struct.pack("<h", value)
    data = bytes(samples)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    return header + data


@dataclass
class MockDashScopeState:
    """Mutable knobs controlling the fake endpoint."""

    #: ``url`` returns a download link, ``base64`` returns inline audio.
    mode: str = "url"
    #: Number of initial synthesis calls answered with HTTP 429.
    rate_limit_times: int = 0
    #: When set, answer HTTP 200 with this business error payload.
    error_payload: dict[str, Any] | None = None
    #: HTTP status used for the audio download route.
    download_status: int = 200
    #: Override the HTTP status of the synthesis route (e.g. 401).
    synthesis_status: int = 200
    #: Body returned by the synthesis route when it should not be valid JSON.
    raw_body: str | None = None
    audio: bytes = field(default_factory=make_wav)
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    downloads: int = 0


class MockDashScope:
    """Running server handle."""

    def __init__(self, state: MockDashScopeState, runner: web.AppRunner, port: int) -> None:
        self.state = state
        self._runner = runner
        self.port = port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1"

    @property
    def synthesis_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{SYNTHESIS_PATH}"

    async def aclose(self) -> None:
        await self._runner.cleanup()


async def start(state: MockDashScopeState | None = None) -> MockDashScope:
    """Start the fake endpoint on an ephemeral localhost port."""
    state = state or MockDashScopeState()
    # The port is only known after binding, so the handler reads it lazily.
    bound: dict[str, int] = {"port": 0}

    async def handle_synthesis(request: web.Request) -> web.Response:
        state.requests.append(await request.json())
        state.headers.append(dict(request.headers))

        if state.raw_body is not None:
            return web.Response(text=state.raw_body, content_type="application/json")
        if state.synthesis_status != 200:
            return web.json_response(
                {"code": "InvalidApiKey", "message": "Invalid API-key provided."},
                status=state.synthesis_status,
            )
        if state.rate_limit_times > 0:
            state.rate_limit_times -= 1
            return web.json_response(
                {"code": "Throttling.RateQuota", "message": "Requests rate limit exceeded."},
                status=429,
                headers={"Retry-After": "0"},
            )
        if state.error_payload is not None:
            return web.json_response(state.error_payload)

        audio = state.audio
        if state.mode == "base64":
            payload = {
                "status_code": 200,
                "request_id": f"req-{len(state.requests)}",
                "code": "",
                "message": "",
                "output": {"audio": {"data": base64.b64encode(audio).decode()}},
                "usage": {"characters": 12},
            }
        else:
            payload = {
                "status_code": 200,
                "request_id": f"req-{len(state.requests)}",
                "code": "",
                "message": "",
                "output": {
                    "audio": {
                        "data": "",
                        "url": f"http://127.0.0.1:{bound['port']}/audio/{len(state.requests)}.wav",
                    },
                },
                "usage": {"characters": 12},
            }
        return web.json_response(payload)

    async def handle_audio(request: web.Request) -> web.Response:
        state.downloads += 1
        if state.download_status != 200:
            return web.Response(status=state.download_status, text="not found")
        return web.Response(body=state.audio, content_type="audio/wav")

    app = web.Application()
    app.router.add_post(SYNTHESIS_PATH, handle_synthesis)
    app.router.add_get("/audio/{name}", handle_audio)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    bound["port"] = sockets[0].getsockname()[1]
    return MockDashScope(state, runner, bound["port"])


async def stop(server: MockDashScope) -> None:
    """Shut the fake endpoint down."""
    await server.aclose()
