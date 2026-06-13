"""Tests for Agent.batch and Agent.abatch."""

from __future__ import annotations

import asyncio
from typing import Any

from SEJO_SDK.agent import Agent
from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import AsyncModelClient, ModelClient


class EchoModel(ModelClient):
    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return ModelResponse(content=f"echo:{prompt}")

    def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        last = msgs[-1].content if msgs else ""
        return ModelResponse(content=f"echo:{last}")

    def stream_response(self, prompt: str, **_: Any):
        yield f"echo:{prompt}"

    def stream_messages(self, msgs: list[Message], **_: Any):
        yield "echo"


class AsyncEchoModel(AsyncModelClient):
    async def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return ModelResponse(content=f"echo:{prompt}")

    async def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        last = msgs[-1].content if msgs else ""
        return ModelResponse(content=f"echo:{last}")

    async def stream_response(self, prompt: str, **_: Any):
        yield f"echo:{prompt}"

    async def stream_messages(self, msgs: list[Message], **_: Any):
        yield "echo"


# ---------------------------------------------------------------------------
# batch (sync)
# ---------------------------------------------------------------------------

def test_batch_returns_all_results():
    agent = Agent(model=EchoModel())
    results = agent.batch(["a", "b", "c"])
    assert len(results) == 3


def test_batch_preserves_order():
    agent = Agent(model=EchoModel())
    results = agent.batch(["x", "y", "z"])
    assert results[0] == "echo:x"
    assert results[1] == "echo:y"
    assert results[2] == "echo:z"


def test_batch_empty_list():
    agent = Agent(model=EchoModel())
    assert agent.batch([]) == []


def test_batch_single_item():
    agent = Agent(model=EchoModel())
    results = agent.batch(["only"])
    assert results == ["echo:only"]


# ---------------------------------------------------------------------------
# abatch (async)
# ---------------------------------------------------------------------------

def test_abatch_returns_all_results():
    async def run():
        agent = Agent(model=AsyncEchoModel())
        return await agent.abatch(["a", "b", "c"])

    results = asyncio.run(run())
    assert len(results) == 3


def test_abatch_preserves_order():
    async def run():
        agent = Agent(model=AsyncEchoModel())
        return await agent.abatch(["p", "q", "r"])

    results = asyncio.run(run())
    assert results[0] == "echo:p"
    assert results[1] == "echo:q"
    assert results[2] == "echo:r"


def test_abatch_empty():
    async def run():
        agent = Agent(model=AsyncEchoModel())
        return await agent.abatch([])

    assert asyncio.run(run()) == []


def test_abatch_max_concurrency():
    import time

    async def run():
        class SlowModel(AsyncModelClient):
            async def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
                await asyncio.sleep(0.01)
                return ModelResponse(content=prompt)

            async def send_messages(self, msgs, **_) -> ModelResponse:
                await asyncio.sleep(0.01)
                last = msgs[-1].content if msgs else ""
                return ModelResponse(content=last)

            async def stream_response(self, p, **_):
                yield p

            async def stream_messages(self, msgs, **_):
                yield ""

        agent = Agent(model=SlowModel())
        t0 = time.monotonic()
        results = await agent.abatch(["a"] * 6, max_concurrency=3)
        elapsed = time.monotonic() - t0
        assert len(results) == 6
        # With concurrency=3, 6 tasks of 10ms each should finish in ~20ms, not 60ms
        assert elapsed < 0.2

    asyncio.run(run())


# ---------------------------------------------------------------------------
# stream_with_tools
# ---------------------------------------------------------------------------

def test_stream_with_tools_no_tools():
    """When no tool calls are made, stream_with_tools should yield the response."""
    class PlainModel(ModelClient):
        def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
            return ModelResponse(content="answer")

        def send_messages(self, msgs: list[Message], **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="answer")

        def stream_response(self, prompt: str, **_: Any):
            yield "answer"

        def stream_messages(self, msgs: list[Message], **_: Any):
            yield "streamed-answer"

    agent = Agent(model=PlainModel())
    chunks = list(agent.stream_with_tools("What is 2+2?"))
    assert "".join(chunks) == "streamed-answer"


def test_stream_with_tools_with_tool_call():
    from SEJO_SDK.messages import ToolCall
    from SEJO_SDK.tools import Tool

    def add(a: int, b: int) -> int:
        return a + b

    t = Tool(
        name="add",
        description="Add two numbers.",
        func=add,
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )

    call_count = [0]

    class ToolModel(ModelClient):
        def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
            return ModelResponse(content="done")

        def send_messages(self, msgs: list[Message], **kwargs: Any) -> ModelResponse:
            if call_count[0] == 0:
                call_count[0] += 1
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="1", name="add", arguments={"a": 3, "b": 4})
                    ],
                )
            return ModelResponse(content="The answer is 7")

        def stream_response(self, prompt: str, **_: Any):
            yield "The answer is 7"

        def stream_messages(self, msgs: list[Message], **_: Any):
            yield "The answer is 7"

    agent = Agent(model=ToolModel(), tools=[t])
    result = "".join(agent.stream_with_tools("What is 3 + 4?"))
    assert "7" in result
