"""Model Context Protocol (MCP) client for SEJO SDK.

Connects to any MCP server over stdio or SSE and exposes its tools as
SEJO ``Tool`` objects, so any ``Agent`` can call them natively.

Usage (stdio transport — most common for local servers)::

    from SEJO_SDK.mcp_client import MCPClient

    async with MCPClient.from_stdio(["python", "-m", "my_mcp_server"]) as client:
        tools = await client.list_tools()
        agent = Agent(model=..., tools=tools)
        result = await agent.arun_with_tools("What's the weather in Madrid?")

Usage (SSE transport — for remote servers)::

    async with MCPClient.from_sse("http://localhost:3000/sse") as client:
        tools = await client.list_tools()

Requires ``pip install sejo-sdk[mcp]`` (``mcp`` package from Anthropic).
"""

from __future__ import annotations

import json
from typing import Any

from SEJO_SDK.tools import Tool


def _mcp_tool_to_sejo(mcp_tool: Any) -> Tool:
    """Convert an MCP tool description to a SEJO Tool (placeholder func)."""
    name = mcp_tool.name
    description = getattr(mcp_tool, "description", "") or ""
    input_schema = getattr(mcp_tool, "inputSchema", None) or {
        "type": "object", "properties": {}, "required": [],
    }

    def _sync_stub(**kwargs: Any) -> str:
        raise NotImplementedError(
            f"MCP tool '{name}' must be called via arun / arun_with_tools."
        )

    # The real async implementation is set after the client is available.
    return Tool(
        name=name,
        description=description,
        func=_sync_stub,
        parameters=input_schema,
    )


class MCPClient:
    """Thin async wrapper around the MCP SDK.

    Do not instantiate directly; use the class-method constructors
    ``from_stdio`` or ``from_sse`` as async context managers.
    """

    def __init__(self, session: Any, exit_stack: Any) -> None:
        self._session = session
        self._exit_stack = exit_stack

    # ------------------------------------------------------------------
    # Constructors (async context managers)
    # ------------------------------------------------------------------

    @classmethod
    def from_stdio(cls, command: list[str]) -> _StdioContext:
        return _StdioContext(command)

    @classmethod
    def from_sse(cls, url: str) -> _SSEContext:
        return _SSEContext(url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[Tool]:
        """Return all tools exposed by the MCP server as SEJO Tool objects."""
        response = await self._session.list_tools()
        tools = []
        for mcp_tool in response.tools:
            tool = _mcp_tool_to_sejo(mcp_tool)
            tool.afunc = self._make_afunc(mcp_tool.name)
            tools.append(tool)
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """Call a named MCP tool and return the result as a string."""
        result = await self._session.call_tool(name, arguments or {})
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(json.dumps(content, default=str))
        return "\n".join(parts)

    def _make_afunc(self, tool_name: str):
        async def _call(**kwargs: Any) -> str:
            return await self.call_tool(tool_name, kwargs)
        return _call

    async def __aenter__(self) -> MCPClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._exit_stack.aclose()


# ---------------------------------------------------------------------------
# Transport-specific context managers
# ---------------------------------------------------------------------------

class _StdioContext:
    def __init__(self, command: list[str]) -> None:
        self._command = command

    async def __aenter__(self) -> MCPClient:
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            from SEJO_SDK.errors import ProviderDependencyError
            raise ProviderDependencyError(
                "Install MCP support with `pip install mcp`."
            ) from exc

        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=self._command[0], args=self._command[1:])
        stdio_transport = await self._stack.enter_async_context(stdio_client(params))
        read, write = stdio_transport
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return MCPClient(session, self._stack)

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._stack.aclose()


class _SSEContext:
    def __init__(self, url: str) -> None:
        self._url = url

    async def __aenter__(self) -> MCPClient:
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as exc:
            from SEJO_SDK.errors import ProviderDependencyError
            raise ProviderDependencyError(
                "Install MCP support with `pip install mcp`."
            ) from exc

        self._stack = AsyncExitStack()
        sse_transport = await self._stack.enter_async_context(sse_client(self._url))
        read, write = sse_transport
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return MCPClient(session, self._stack)

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._stack.aclose()
