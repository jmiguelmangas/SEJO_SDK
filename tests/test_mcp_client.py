"""Tests for MCPClient — using a fake MCP session."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from SEJO_SDK.mcp_client import MCPClient, _mcp_tool_to_sejo
from SEJO_SDK.tools import Tool

# ---------------------------------------------------------------------------
# Fake MCP session
# ---------------------------------------------------------------------------

def _make_fake_tool(name: str, description: str, properties: dict | None = None):
    schema = {
        "type": "object",
        "properties": properties or {"query": {"type": "string"}},
        "required": list((properties or {"query": {}}).keys()),
    }
    return SimpleNamespace(name=name, description=description, inputSchema=schema)


class FakeMCPSession:
    def __init__(self, tools: list, call_results: dict[str, str] | None = None) -> None:
        self._tools = tools
        self._call_results = call_results or {}

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name: str, arguments: dict) -> Any:
        text = self._call_results.get(name, f"{name}_result")
        content = [SimpleNamespace(text=text)]
        return SimpleNamespace(content=content)


class FakeExitStack:
    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# _mcp_tool_to_sejo
# ---------------------------------------------------------------------------

def test_convert_mcp_tool_name():
    mcp_tool = _make_fake_tool("weather", "Get weather")
    tool = _mcp_tool_to_sejo(mcp_tool)
    assert isinstance(tool, Tool)
    assert tool.name == "weather"


def test_convert_mcp_tool_description():
    mcp_tool = _make_fake_tool("weather", "Get weather data")
    tool = _mcp_tool_to_sejo(mcp_tool)
    assert tool.description == "Get weather data"


def test_convert_mcp_tool_schema():
    mcp_tool = _make_fake_tool("search", "Search", {"q": {"type": "string"}})
    tool = _mcp_tool_to_sejo(mcp_tool)
    assert "q" in tool.parameters["properties"]


def test_convert_mcp_tool_sync_stub_raises():
    import pytest
    mcp_tool = _make_fake_tool("t", "d")
    tool = _mcp_tool_to_sejo(mcp_tool)
    with pytest.raises(NotImplementedError):
        tool.func()


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------

def test_list_tools_returns_sejo_tools():
    async def run():
        fake_tools = [
            _make_fake_tool("weather", "Get weather"),
            _make_fake_tool("search", "Search the web"),
        ]
        session = FakeMCPSession(fake_tools)
        client = MCPClient(session, FakeExitStack())
        tools = await client.list_tools()
        assert len(tools) == 2
        assert {t.name for t in tools} == {"weather", "search"}

    asyncio.run(run())


def test_list_tools_have_afunc():
    async def run():
        session = FakeMCPSession([_make_fake_tool("t", "desc")])
        client = MCPClient(session, FakeExitStack())
        tools = await client.list_tools()
        assert tools[0].afunc is not None

    asyncio.run(run())


def test_call_tool_returns_text():
    async def run():
        session = FakeMCPSession(
            [_make_fake_tool("weather", "Get weather")],
            call_results={"weather": "Sunny, 25°C"},
        )
        client = MCPClient(session, FakeExitStack())
        result = await client.call_tool("weather", {"city": "Madrid"})
        assert result == "Sunny, 25°C"

    asyncio.run(run())


def test_tool_afunc_calls_session():
    async def run():
        session = FakeMCPSession(
            [_make_fake_tool("echo", "Echo")],
            call_results={"echo": "echoed"},
        )
        client = MCPClient(session, FakeExitStack())
        tools = await client.list_tools()
        result = await tools[0].afunc(message="hello")
        assert result == "echoed"

    asyncio.run(run())


def test_call_tool_multiple_content_parts():
    async def run():
        class MultiSession:
            async def call_tool(self, name, args):
                parts = [SimpleNamespace(text="part1"), SimpleNamespace(text="part2")]
                return SimpleNamespace(content=parts)

        client = MCPClient(MultiSession(), FakeExitStack())
        result = await client.call_tool("t", {})
        assert "part1" in result
        assert "part2" in result

    asyncio.run(run())


def test_mcp_client_context_manager():
    async def run():
        session = FakeMCPSession([])
        client = MCPClient(session, FakeExitStack())
        async with client as c:
            assert c is client

    asyncio.run(run())
