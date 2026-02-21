"""Lightweight OpenAI-compatible stub LLM server for integration tests.

Serves ``/v1/chat/completions`` with configurable responses.
Used by summarizer and postprocessor integration tests via litellm.
"""

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class StubConfig:
    """Configure stub server behavior."""

    # Number of 429 responses before returning 200
    rate_limit_count: int = 0
    # Content to return in the completion
    response_content: str = "This is a stub LLM response."
    # Return null content (simulates edge case)
    null_content: bool = False


@dataclass
class StubLLMServer:
    """In-process stub LLM server for tests."""

    config: StubConfig = field(default_factory=StubConfig)
    port: int = 0
    _app: FastAPI = field(default=None, init=False)
    _server: uvicorn.Server = field(default=None, init=False)
    _thread: threading.Thread = field(default=None, init=False)
    _rate_limit_hits: int = field(default=0, init=False)
    # Track received requests for assertions
    requests: list[dict] = field(default_factory=list)

    def _create_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            self.requests.append(body)

            # Rate limiting simulation
            if self._rate_limit_hits < self.config.rate_limit_count:
                self._rate_limit_hits += 1
                return JSONResponse(
                    status_code=429,
                    content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                )

            content = None if self.config.null_content else self.config.response_content

            return JSONResponse(content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "test-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            })

        return app

    def start(self):
        self._app = self._create_app()
        config = uvicorn.Config(
            self._app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        # Wait for server to start
        for _ in range(50):
            if self._server.started:
                break
            time.sleep(0.1)
        # Get actual port
        for sock in self._server.servers[0].sockets:
            self.port = sock.getsockname()[1]
            break

    def stop(self):
        if self._server:
            self._server.should_exit = True
            self._thread.join(timeout=5)

    def reset(self):
        self._rate_limit_hits = 0
        self.requests.clear()


@contextmanager
def run_stub_server(config: StubConfig | None = None):
    """Context manager that starts/stops a stub LLM server."""
    server = StubLLMServer(config=config or StubConfig())
    server.start()
    try:
        yield server
    finally:
        server.stop()
