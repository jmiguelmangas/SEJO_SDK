"""Tests for HookRegistry and Agent hook integration."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from SEJO_SDK.agent import Agent
from SEJO_SDK.hooks import HookRegistry
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


class BoomModel(ModelClient):
    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        raise RuntimeError("boom")

    def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        raise RuntimeError("boom")

    def stream_response(self, prompt: str, **_: Any):
        raise RuntimeError()
        yield

    def stream_messages(self, msgs: list[Message], **_: Any):
        raise RuntimeError()
        yield


# ---------------------------------------------------------------------------
# HookRegistry unit tests
# ---------------------------------------------------------------------------

def test_on_turn_start_decorator():
    hooks = HookRegistry()
    calls: list[dict] = []

    @hooks.on_turn_start
    def capture(ctx):
        calls.append(dict(ctx))

    hooks.fire_start({"turn": 1, "input": "hi"})
    assert len(calls) == 1
    assert calls[0]["input"] == "hi"


def test_on_turn_end_decorator():
    hooks = HookRegistry()
    calls: list[dict] = []

    @hooks.on_turn_end
    def capture(ctx):
        calls.append(dict(ctx))

    hooks.fire_end({"turn": 1, "output": "bye"})
    assert calls[0]["output"] == "bye"


def test_multiple_hooks_all_called():
    hooks = HookRegistry()
    log: list[str] = []
    hooks.on_turn_start(lambda ctx: log.append("a"))
    hooks.on_turn_start(lambda ctx: log.append("b"))
    hooks.fire_start({"turn": 1})
    assert log == ["a", "b"]


def test_turn_counter_increments():
    hooks = HookRegistry()
    turns: list[int] = []
    hooks.on_turn_start(lambda ctx: turns.append(ctx["turn"]))
    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("first")
    agent.run("second")
    assert turns == [1, 2]


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

def test_agent_fires_start_and_end():
    hooks = HookRegistry()
    started: list[str] = []
    ended: list[str] = []

    hooks.on_turn_start(lambda ctx: started.append(ctx["input"]))
    hooks.on_turn_end(lambda ctx: ended.append(ctx["output"]))

    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("hello")

    assert started == ["hello"]
    assert ended == ["echo:hello"]


def test_agent_end_hook_receives_duration():
    hooks = HookRegistry()
    durations: list[float] = []
    hooks.on_turn_end(lambda ctx: durations.append(ctx["duration_ms"]))

    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("hi")

    assert len(durations) == 1
    assert durations[0] >= 0


def test_agent_end_hook_no_error_on_success():
    hooks = HookRegistry()
    errors: list = []
    hooks.on_turn_end(lambda ctx: errors.append(ctx["error"]))

    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("hi")

    assert errors == [None]


def test_agent_end_hook_captures_error():
    hooks = HookRegistry()
    errors: list = []
    hooks.on_turn_end(lambda ctx: errors.append(ctx["error"]))

    agent = Agent(model=BoomModel(), hooks=hooks)
    with pytest.raises(RuntimeError):
        agent.run("hi")

    assert errors[0] is not None
    assert "boom" in errors[0]


def test_agent_without_hooks_works_normally():
    agent = Agent(model=EchoModel())
    result = agent.run("no hooks")
    assert result == "echo:no hooks"


# ---------------------------------------------------------------------------
# Async hooks
# ---------------------------------------------------------------------------

def test_async_agent_fires_hooks():
    async def run():
        hooks = HookRegistry()
        inputs: list[str] = []
        outputs: list[str] = []
        hooks.on_turn_start(lambda ctx: inputs.append(ctx["input"]))
        hooks.on_turn_end(lambda ctx: outputs.append(ctx["output"]))

        agent = Agent(model=AsyncEchoModel(), hooks=hooks)
        await agent.arun("async-hello")

        assert inputs == ["async-hello"]
        assert outputs == ["echo:async-hello"]

    asyncio.run(run())


def test_async_hook_can_be_coroutine():
    async def run():
        hooks = HookRegistry()
        log: list[str] = []

        async def async_hook(ctx):
            log.append(ctx["input"])

        hooks.on_turn_start(async_hook)
        agent = Agent(model=AsyncEchoModel(), hooks=hooks)
        await agent.arun("coro-hook")

        assert log == ["coro-hook"]

    asyncio.run(run())


def test_hook_ctx_contains_agent_ref():
    hooks = HookRegistry()
    agents_seen: list = []
    hooks.on_turn_start(lambda ctx: agents_seen.append(ctx["agent"]))

    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("ref-check")

    assert agents_seen[0] is agent
