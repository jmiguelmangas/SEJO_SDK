import asyncio
import builtins
import sys
from types import ModuleType

import pytest

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.tools.chat_ws_tool import create_chat_app


class FakeFastAPI:
    def __init__(self, title):
        self.title = title
        self.websocket_handlers = {}

    def websocket(self, path):
        def decorator(func):
            self.websocket_handlers[path] = func
            return func

        return decorator


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = messages
        self.accepted = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self.messages:
            raise RuntimeError("done")
        return self.messages.pop(0)

    async def send_text(self, response):
        self.sent.append(response)


class FakeAgent:
    def run(self, message):
        return f"echo: {message}"


def test_create_chat_app_registers_websocket_and_responds(monkeypatch):
    fake_fastapi = ModuleType("fastapi")
    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.WebSocket = FakeWebSocket
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)

    app = create_chat_app(FakeAgent(), path="/ws")
    websocket = FakeWebSocket(["hello"])

    async def exercise():
        try:
            await app.websocket_handlers["/ws"](websocket)
        except RuntimeError as exc:
            assert str(exc) == "done"

    asyncio.run(exercise())

    assert app.title == "SEJO SDK Chat"
    assert websocket.accepted is True
    assert websocket.sent == ["echo: hello"]


def test_create_chat_app_raises_dependency_error_when_fastapi_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastapi":
            raise ImportError("missing fastapi")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "fastapi", raising=False)

    with pytest.raises(ProviderDependencyError, match="server support"):
        create_chat_app(FakeAgent())
