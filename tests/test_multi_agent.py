"""Tests for multi-agent patterns: as_tool() and delegate()."""

from __future__ import annotations

import asyncio
from typing import Any

from SEJO_SDK.agent import Agent
from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.tools import Tool

# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------

class EchoModel(ModelClient):
    """Returns the last user message prefixed with a tag."""

    def __init__(self, tag: str = "echo") -> None:
        self.tag = tag
        self.calls: list[str] = []

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.calls.append(prompt)
        return ModelResponse(content=f"[{self.tag}] {prompt}")

    def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        last = messages[-1].content if messages else ""
        self.calls.append(last)
        return ModelResponse(content=f"[{self.tag}] {last}")

    def stream_response(self, prompt: str, **_: Any):
        yield f"[{self.tag}] {prompt}"

    def stream_messages(self, messages: list[Message], **_: Any):
        yield f"[{self.tag}]"


class AsyncEchoModel(AsyncModelClient):
    def __init__(self, tag: str = "async") -> None:
        self.tag = tag

    async def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return ModelResponse(content=f"[{self.tag}] {prompt}")

    async def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        last = messages[-1].content if messages else ""
        return ModelResponse(content=f"[{self.tag}] {last}")

    async def stream_response(self, prompt: str, **_: Any):
        yield f"[{self.tag}] {prompt}"

    async def stream_messages(self, messages: list[Message], **_: Any):
        yield f"[{self.tag}]"


class ToolCallModel(ModelClient):
    """Returns a pre-set sequence of responses (text or tool calls)."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return self._responses.pop(0)

    def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        return self._responses.pop(0)

    def stream_response(self, prompt: str, **_: Any):
        yield "x"

    def stream_messages(self, messages: list[Message], **_: Any):
        yield "x"


# ---------------------------------------------------------------------------
# as_tool
# ---------------------------------------------------------------------------

def test_as_tool_returns_tool():
    agent = Agent(model=EchoModel())
    tool = agent.as_tool("helper", "A helper agent")
    assert isinstance(tool, Tool)
    assert tool.name == "helper"
    assert tool.description == "A helper agent"


def test_as_tool_schema_has_task_param():
    agent = Agent(model=EchoModel())
    tool = agent.as_tool("helper", "desc")
    schema = tool.to_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "task" in props


def test_as_tool_custom_param_name():
    agent = Agent(model=EchoModel())
    tool = agent.as_tool("helper", "desc", param="query")
    schema = tool.to_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "query" in props


def test_as_tool_run_delegates_to_agent():
    sub = Agent(model=EchoModel("sub"))
    tool = sub.as_tool("sub_agent", "Sub agent")
    result = tool.run(task="do something")
    assert "[sub]" in result
    assert "do something" in result


def test_as_tool_used_by_orchestrator():
    """Orchestrator calls a sub-agent as a tool via run_with_tools."""
    from SEJO_SDK.messages import ToolCall

    sub = Agent(model=EchoModel("researcher"), system_prompt="Research specialist")
    researcher_tool = sub.as_tool("research", "Research a topic")

    # Orchestrator first returns a tool call, then uses the result to respond
    orch_responses = [
        ModelResponse(
            content="",
            tool_calls=[
                ToolCall(name="research", arguments={"task": "AI in aviation"})
            ],
        ),
        ModelResponse(content="Final report based on research"),
    ]
    orchestrator = Agent(
        model=ToolCallModel(orch_responses),
        tools=[researcher_tool],
    )
    result = orchestrator.run_with_tools("Write a report on AI in aviation")
    assert result == "Final report based on research"


def test_as_tool_arun():
    async def run():
        sub = Agent(model=AsyncEchoModel("async-sub"))
        tool = sub.as_tool("async_helper", "desc")
        result = await tool.arun(task="async task")
        assert "[async-sub]" in result

    asyncio.run(run())


# ---------------------------------------------------------------------------
# delegate / adelegate
# ---------------------------------------------------------------------------

def test_delegate_runs_other_agent():
    sub = Agent(model=EchoModel("specialist"))
    orchestrator = Agent(model=EchoModel("main"))
    result = orchestrator.delegate(sub, "analyze flight BA123")
    assert "[specialist]" in result
    assert "BA123" in result


def test_delegate_does_not_affect_orchestrator_memory():
    sub = Agent(model=EchoModel("sub"))
    orchestrator = Agent(model=EchoModel("main"))
    orchestrator.run("first message")
    orchestrator.delegate(sub, "subtask")
    # Orchestrator memory should only have the first message exchange
    msgs = orchestrator.memory.get_messages()
    assert all("subtask" not in m.content for m in msgs)


def test_adelegate():
    async def run():
        sub = Agent(model=AsyncEchoModel("sub"))
        orch = Agent(model=AsyncEchoModel("orch"))
        result = await orch.adelegate(sub, "async subtask")
        assert "[sub]" in result

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Chained multi-agent
# ---------------------------------------------------------------------------

def test_three_agent_chain():
    """Orchestrator → researcher → summarizer chain via as_tool."""
    from SEJO_SDK.messages import ToolCall

    summarizer = Agent(model=EchoModel("summarizer"))
    summarize_tool = summarizer.as_tool("summarize", "Summarize text")

    researcher = Agent(model=EchoModel("researcher"), tools=[summarize_tool])

    orch_responses = [
        ModelResponse(
            content="",
            tool_calls=[ToolCall(name="research", arguments={"task": "topic"})],
        ),
        ModelResponse(content="Done"),
    ]
    orchestrator = Agent(
        model=ToolCallModel(orch_responses),
        tools=[researcher.as_tool("research", "Research")],
    )
    result = orchestrator.run_with_tools("Start")
    assert result == "Done"
