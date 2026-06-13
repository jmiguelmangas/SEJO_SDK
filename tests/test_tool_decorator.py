"""Tests for the @tool decorator."""

from __future__ import annotations

import asyncio

import pytest

from SEJO_SDK.tools import Tool, tool

# ---------------------------------------------------------------------------
# Basic decoration
# ---------------------------------------------------------------------------

def test_tool_decorator_returns_tool():
    @tool
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"

    assert isinstance(greet, Tool)


def test_tool_name_from_function():
    @tool
    def get_weather(city: str) -> str:
        """Get weather."""
        return "sunny"

    assert get_weather.name == "get_weather"


def test_tool_description_from_docstring():
    @tool
    def search(query: str) -> str:
        """Search the web for information."""
        return "results"

    assert search.description == "Search the web for information."


def test_tool_runs_function():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add.run(a=2, b=3) == 5


def test_tool_schema_has_properties():
    @tool
    def lookup(flight_id: str, date: str) -> str:
        """Look up a flight."""
        return "ok"

    schema = lookup.parameters
    assert "flight_id" in schema["properties"]
    assert "date" in schema["properties"]


def test_tool_required_params_no_default():
    @tool
    def fn(required: str, optional: str = "default") -> str:
        """Test fn."""
        return required

    assert "required" in fn.parameters["required"]
    assert "optional" not in fn.parameters["required"]


def test_tool_type_hints_mapped():
    @tool
    def fn(name: str, count: int, ratio: float, active: bool) -> str:
        """Type test."""
        return ""

    props = fn.parameters["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["active"]["type"] == "boolean"


def test_tool_no_docstring():
    @tool
    def bare(x: str) -> str:
        return x

    assert bare.description == "bare"  # falls back to function name
    assert bare.run(x="hi") == "hi"


# ---------------------------------------------------------------------------
# Async tools
# ---------------------------------------------------------------------------

def test_async_tool_has_afunc():
    @tool
    async def async_search(query: str) -> str:
        """Search async."""
        return f"results for {query}"

    assert async_search.afunc is not None


def test_async_tool_sync_stub_raises():
    @tool
    async def async_fn(x: str) -> str:
        """Async only."""
        return x

    with pytest.raises(NotImplementedError):
        async_fn.func(x="test")


def test_async_tool_arun():
    @tool
    async def async_greet(name: str) -> str:
        """Greet async."""
        return f"Hi {name}"

    async def run():
        return await async_greet.arun(name="world")

    result = asyncio.run(run())
    assert result == "Hi world"


# ---------------------------------------------------------------------------
# Integration: tool in Agent
# ---------------------------------------------------------------------------

def test_tool_decorator_works_in_agent():
    from SEJO_SDK.agent import Agent
    from SEJO_SDK.messages import ModelResponse
    from SEJO_SDK.model import ModelClient

    @tool
    def echo_tool(text: str) -> str:
        """Echo the input text."""
        return text

    class ToolCallModel(ModelClient):
        _called = False

        def send_prompt(self, prompt, **_) -> ModelResponse:
            return ModelResponse(content="done")

        def send_messages(self, msgs, **kwargs) -> ModelResponse:
            if not self._called and kwargs.get("tools"):
                self.__class__._called = True
                from SEJO_SDK.messages import ToolCall
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="1", name="echo_tool", arguments={"text": "hello"})
                    ],
                )
            return ModelResponse(content="The echo says: hello")

        def stream_response(self, prompt, **_):
            yield "done"

        def stream_messages(self, msgs, **_):
            yield "done"

    ToolCallModel._called = False
    agent = Agent(model=ToolCallModel(), tools=[echo_tool])
    result = agent.run_with_tools("Echo hello")
    assert "hello" in result
