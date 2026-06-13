"""Tests for Gemini native tool calling and message adapters."""

from __future__ import annotations

import asyncio
import types
from typing import Any

from SEJO_SDK.messages import (
    ModelResponse,
    ToolCall,
    assistant_message_with_tools,
    system_message,
    tool_message,
    user_message,
)
from SEJO_SDK.models.message_adapters import (
    messages_to_gemini_contents,
    parse_gemini_response,
    tools_to_gemini,
)
from SEJO_SDK.models.model_gemini import AsyncGeminiModel, GeminiModel

# ---------------------------------------------------------------------------
# Fake Gemini client helpers
# ---------------------------------------------------------------------------

def _make_text_response(text: str) -> Any:
    """Fake generate_content response that returns a text message."""
    fc_part = types.SimpleNamespace(text=text, function_call=None)
    content = types.SimpleNamespace(parts=[fc_part])
    candidate = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(candidates=[candidate], text=text)


def _make_tool_call_response(name: str, args: dict) -> Any:
    """Fake response that triggers a function call."""
    fc = types.SimpleNamespace(name=name, args=args)
    part = types.SimpleNamespace(text=None, function_call=fc)
    content = types.SimpleNamespace(parts=[part])
    candidate = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(candidates=[candidate], text=None)


class FakeGeminiClient:
    """Mimics google.generativeai.GenerativeModel."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def _pop(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        return self._responses.pop(0)

    def generate_content(self, **kwargs) -> Any:
        return self._pop(**kwargs)

    async def generate_content_async(self, **kwargs) -> Any:
        return self._pop(**kwargs)


# ---------------------------------------------------------------------------
# messages_to_gemini_contents
# ---------------------------------------------------------------------------

def test_user_message_converted():
    _, contents = messages_to_gemini_contents([user_message("Hello")])
    assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]


def test_system_extracted():
    system, contents = messages_to_gemini_contents(
        [system_message("Be brief."), user_message("Hi")]
    )
    assert system == "Be brief."
    assert len(contents) == 1
    assert contents[0]["role"] == "user"


def test_assistant_tool_call_turn():
    tc = ToolCall(name="get_weather", arguments={"city": "London"})
    msg = assistant_message_with_tools("", [tc])
    _, contents = messages_to_gemini_contents([user_message("?"), msg])
    model_turn = contents[1]
    assert model_turn["role"] == "model"
    assert any("function_call" in p for p in model_turn["parts"])


def test_tool_result_turn():
    tr = tool_message("Sunny", name="get_weather", tool_call_id="1")
    _, contents = messages_to_gemini_contents([user_message("?"), tr])
    tool_turn = contents[1]
    assert tool_turn["role"] == "user"
    assert "function_response" in tool_turn["parts"][0]


def test_consecutive_tool_results_merged():
    t1 = tool_message("A", name="fn1", tool_call_id="1")
    t2 = tool_message("B", name="fn2", tool_call_id="2")
    _, contents = messages_to_gemini_contents([user_message("?"), t1, t2])
    # both tool results should be merged into a single user turn
    tool_turns = [
        c for c in contents
        if c["role"] == "user"
        and any("function_response" in p for p in c["parts"])
    ]
    assert len(tool_turns) == 1
    assert len(tool_turns[0]["parts"]) == 2


# ---------------------------------------------------------------------------
# tools_to_gemini
# ---------------------------------------------------------------------------

def test_tools_to_gemini_format():
    schema = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    result = tools_to_gemini(schema)
    assert len(result) == 1
    assert "function_declarations" in result[0]
    decl = result[0]["function_declarations"][0]
    assert decl["name"] == "get_weather"


# ---------------------------------------------------------------------------
# parse_gemini_response
# ---------------------------------------------------------------------------

def test_parse_text_response():
    resp = _make_text_response("Hello from Gemini")
    result = parse_gemini_response(resp)
    assert result.content == "Hello from Gemini"
    assert result.tool_calls == []


def test_parse_tool_call_response():
    resp = _make_tool_call_response("get_weather", {"city": "Paris"})
    result = parse_gemini_response(resp)
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "Paris"}


def test_parse_fallback_no_candidates():
    resp = types.SimpleNamespace(candidates=[], text="fallback")
    result = parse_gemini_response(resp)
    assert result.content == "fallback"


# ---------------------------------------------------------------------------
# GeminiModel.send_messages
# ---------------------------------------------------------------------------

def test_gemini_model_send_messages_text():
    client = FakeGeminiClient([_make_text_response("Nice day")])
    model = GeminiModel(api_key="k", model_name="gemini-pro", client=client)
    result = model.send_messages([user_message("How's the weather?")])
    assert isinstance(result, ModelResponse)
    assert result.content == "Nice day"


_SIMPLE_TOOL = [
    {
        "type": "function",
        "function": {"name": "fn", "description": "d", "parameters": {}},
    }
]
_WEATHER_TOOL = [
    {
        "type": "function",
        "function": {"name": "get_weather", "description": "d", "parameters": {}},
    }
]


def test_gemini_model_send_messages_with_tools_text():
    tools = _SIMPLE_TOOL
    client = FakeGeminiClient([_make_text_response("Done")])
    model = GeminiModel(api_key="k", model_name="gemini-pro", client=client)
    result = model.send_messages([user_message("Go")], tools=tools)
    assert result.content == "Done"
    assert "tools" in client.calls[0]


def test_gemini_model_returns_tool_call():
    tools = _WEATHER_TOOL
    client = FakeGeminiClient(
        [_make_tool_call_response("get_weather", {"city": "LHR"})]
    )
    model = GeminiModel(api_key="k", model_name="gemini-pro", client=client)
    result = model.send_messages([user_message("Weather?")], tools=tools)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"


def test_gemini_model_system_prepended():
    client = FakeGeminiClient([_make_text_response("ok")])
    model = GeminiModel(api_key="k", model_name="gemini-pro", client=client)
    model.send_messages([system_message("Be brief."), user_message("Hi")])
    contents = client.calls[0]["contents"]
    first_text = contents[0]["parts"][0]["text"]
    assert "[System]" in first_text


# ---------------------------------------------------------------------------
# AsyncGeminiModel
# ---------------------------------------------------------------------------

def test_async_gemini_model_send_messages():
    async def run():
        client = FakeGeminiClient([_make_text_response("async ok")])
        model = AsyncGeminiModel(api_key="k", model_name="gemini-pro", client=client)
        result = await model.send_messages([user_message("Hi")])
        assert result.content == "async ok"

    asyncio.run(run())


def test_async_gemini_model_tool_call():
    async def run():
        tools = _SIMPLE_TOOL
        client = FakeGeminiClient([_make_tool_call_response("fn", {"x": 1})])
        model = AsyncGeminiModel(api_key="k", model_name="gemini-pro", client=client)
        result = await model.send_messages([user_message("Go")], tools=tools)
        assert result.tool_calls[0].name == "fn"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Agent integration with Gemini tools
# ---------------------------------------------------------------------------

def test_agent_run_with_gemini_tools():
    """Agent calls a tool via Gemini native tool calling."""
    from SEJO_SDK.agent import Agent
    from SEJO_SDK.tools import Tool

    # First response: tool call. Second response: final answer.
    client = FakeGeminiClient([
        _make_tool_call_response("add", {"a": 3, "b": 4}),
        _make_text_response("The answer is 7"),
    ])
    model = GeminiModel(api_key="k", model_name="gemini-pro", client=client)

    add_tool = Tool(
        name="add",
        description="Add two numbers",
        func=lambda a, b: a + b,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )
    agent = Agent(model=model, tools=[add_tool])
    result = agent.run_with_tools("What is 3 + 4?")
    assert result == "The answer is 7"
