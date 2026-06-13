"""Tests for built-in tracing and cost tracking."""


import asyncio

from SEJO_SDK.agent import Agent
from SEJO_SDK.messages import ModelResponse, ToolCall, Usage
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.tracing import Tracer


class FakeTracingModel(ModelClient):
    def __init__(self, response="traced ok"):
        self.response = response

    def send_prompt(self, prompt, **kwargs):
        return self.response

    def send_messages(self, messages, **kwargs):
        return ModelResponse(
            content=self.response,
            usage=Usage(input_tokens=10, output_tokens=20),
        )

    def stream_response(self, prompt, **kwargs):
        yield self.response


class FakeToolCallingTracingModel(ModelClient):
    def __init__(self):
        self.calls = 0

    def send_prompt(self, prompt, **kwargs):
        return "ok"

    def send_messages(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="add", arguments={"left": 1, "right": 2})
                ],
                usage=Usage(input_tokens=5, output_tokens=5),
            )
        return ModelResponse(
            content="The answer is 3.",
            usage=Usage(input_tokens=15, output_tokens=10),
        )

    def stream_response(self, prompt, **kwargs):
        yield "ok"


def test_tracer_records_run_turn():
    tracer = Tracer()
    model = FakeTracingModel()
    agent = Agent(model=model, tracer=tracer)

    agent.run("Hello")

    assert len(tracer.turns) == 1
    turn = tracer.turns[0]
    assert turn.input == "Hello"
    assert turn.output == "traced ok"
    assert turn.usage.input_tokens == 10
    assert turn.usage.output_tokens == 20


def test_tracer_summary_aggregates_multiple_turns():
    tracer = Tracer()
    model = FakeTracingModel()
    agent = Agent(model=model, tracer=tracer)

    agent.run("First")
    agent.run("Second")

    summary = tracer.summary()
    assert summary.turns == 2
    assert summary.total_input_tokens == 20
    assert summary.total_output_tokens == 40
    assert summary.total_tokens == 60


def test_tracer_estimates_cost_for_known_model():
    tracer = Tracer(model_name="gpt-4o-mini")
    model = FakeTracingModel()
    agent = Agent(model=model, tracer=tracer)

    agent.run("Hello")

    summary = tracer.summary()
    expected = (10 * 0.15 + 20 * 0.60) / 1_000_000
    assert abs(summary.total_cost_usd - expected) < 1e-9


def test_tracer_returns_zero_cost_for_unknown_model():
    tracer = Tracer(model_name="unknown-model-xyz")
    model = FakeTracingModel()
    agent = Agent(model=model, tracer=tracer)

    agent.run("Hello")

    assert tracer.summary().total_cost_usd == 0.0


def test_tracer_records_tool_calls():
    from SEJO_SDK.tools import Tool

    tracer = Tracer()
    model = FakeToolCallingTracingModel()
    tool = Tool(name="add", description="add", func=lambda left, right: left + right)
    agent = Agent(model=model, tools=[tool], tracer=tracer)

    agent.run_with_tools("What is 1+2?")

    summary = tracer.summary()
    assert summary.turns == 1
    assert summary.total_input_tokens == 20
    assert summary.total_output_tokens == 15


def test_tracer_reset_clears_turns():
    tracer = Tracer()
    model = FakeTracingModel()
    agent = Agent(model=model, tracer=tracer)

    agent.run("Hello")
    tracer.reset()

    assert len(tracer.turns) == 0
    assert tracer.summary().turns == 0


def test_tracer_duration_is_positive():
    tracer = Tracer()
    model = FakeTracingModel()
    agent = Agent(model=model, tracer=tracer)

    agent.run("Hello")

    assert tracer.turns[0].duration_ms >= 0


def test_tracer_summary_to_dict():
    tracer = Tracer(model_name="gpt-4o-mini")
    model = FakeTracingModel()
    agent = Agent(model=model, tracer=tracer)

    agent.run("Hello")

    d = tracer.summary().to_dict()
    assert "turns" in d
    assert "tokens" in d
    assert "cost_usd" in d
    assert "duration_ms" in d
    assert "tools_called" in d


def test_tracer_works_with_arun():
    async def exercise():
        class AsyncFakeModel(AsyncModelClient):
            async def send_prompt(self, prompt, **kwargs):
                return "async traced"

            async def send_messages(self, messages, **kwargs):
                return ModelResponse(
                    content="async traced",
                    usage=Usage(input_tokens=5, output_tokens=8),
                )

            async def stream_response(self, prompt, **kwargs):
                yield "async traced"

        tracer = Tracer()
        agent = Agent(model=AsyncFakeModel(), tracer=tracer)
        await agent.arun("Hello async")

        assert tracer.turns[0].usage.input_tokens == 5

    asyncio.run(exercise())
