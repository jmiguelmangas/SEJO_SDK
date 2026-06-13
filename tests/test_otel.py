"""Tests for OTelHooks — using an in-process span collector."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")

import asyncio
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from SEJO_SDK.agent import Agent
from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.otel import OTelHooks


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
        raise RuntimeError("otel-boom")

    def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        raise RuntimeError("otel-boom")

    def stream_response(self, prompt: str, **_: Any):
        raise RuntimeError()
        yield

    def stream_messages(self, msgs: list[Message], **_: Any):
        raise RuntimeError()
        yield


def _make_hooks() -> tuple[OTelHooks, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    hooks = OTelHooks(provider=provider)
    return hooks, exporter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_span_emitted_per_turn():
    hooks, exporter = _make_hooks()
    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("hello")
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "sejo.agent.turn"


def test_span_attributes_set():
    hooks, exporter = _make_hooks()
    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("what is LHR?")
    span = exporter.get_finished_spans()[0]
    attrs = span.attributes
    assert attrs["sejo.input"] == "what is LHR?"
    assert "echo:" in attrs["sejo.output"]
    assert attrs["sejo.turn"] == 1


def test_multiple_turns_produce_multiple_spans():
    hooks, exporter = _make_hooks()
    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("first")
    agent.run("second")
    assert len(exporter.get_finished_spans()) == 2


def test_span_turn_counter():
    hooks, exporter = _make_hooks()
    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("a")
    agent.run("b")
    turns = [s.attributes["sejo.turn"] for s in exporter.get_finished_spans()]
    assert turns == [1, 2]


def test_span_status_ok_on_success():
    from opentelemetry.trace import StatusCode

    hooks, exporter = _make_hooks()
    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("ok")
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.OK


def test_span_status_error_on_failure():
    from opentelemetry.trace import StatusCode

    hooks, exporter = _make_hooks()
    agent = Agent(model=BoomModel(), hooks=hooks)
    with pytest.raises(RuntimeError):
        agent.run("crash")
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "otel-boom" in span.attributes.get("sejo.error", "")


def test_model_name_attribute():
    hooks, exporter = _make_hooks()
    agent = Agent(model=EchoModel(), hooks=hooks)
    agent.run("hi")
    span = exporter.get_finished_spans()[0]
    assert span.attributes.get("sejo.model") == "EchoModel"


def test_async_span_emitted():
    async def run():
        hooks, exporter = _make_hooks()
        agent = Agent(model=AsyncEchoModel(), hooks=hooks)
        await agent.arun("async-hi")
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["sejo.input"] == "async-hi"

    asyncio.run(run())
