"""Tests for the agent evaluation framework."""

from __future__ import annotations

import asyncio
from typing import Any

from SEJO_SDK.agent import Agent
from SEJO_SDK.evals import (
    EvalCase,
    EvalReport,
    EvalResult,
    EvalSuite,
    contains,
    contains_all,
    exact_match,
)
from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import AsyncModelClient, ModelClient

# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------

class EchoModel(ModelClient):
    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return ModelResponse(content=prompt)

    def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        last = msgs[-1].content if msgs else ""
        return ModelResponse(content=last)

    def stream_response(self, prompt: str, **_: Any):
        yield prompt

    def stream_messages(self, msgs: list[Message], **_: Any):
        yield ""


class FixedModel(ModelClient):
    """Always returns the same reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return ModelResponse(content=self.reply)

    def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        return ModelResponse(content=self.reply)

    def stream_response(self, prompt: str, **_: Any):
        yield self.reply

    def stream_messages(self, msgs: list[Message], **_: Any):
        yield self.reply


class FailModel(ModelClient):
    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        raise RuntimeError("model error")

    def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        raise RuntimeError("model error")

    def stream_response(self, prompt: str, **_: Any):
        raise RuntimeError()
        yield

    def stream_messages(self, msgs: list[Message], **_: Any):
        raise RuntimeError()
        yield


class AsyncFixedModel(AsyncModelClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return ModelResponse(content=self.reply)

    async def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        return ModelResponse(content=self.reply)

    async def stream_response(self, prompt: str, **_: Any):
        yield self.reply

    async def stream_messages(self, msgs: list[Message], **_: Any):
        yield self.reply


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def test_exact_match_pass():
    assert exact_match("hello", "hello") == 1.0


def test_exact_match_fail():
    assert exact_match("hello world", "hello") == 0.0


def test_exact_match_case_insensitive():
    assert exact_match("HELLO", "hello") == 1.0


def test_contains_pass():
    assert contains("the answer is LHR airport", "LHR") == 1.0


def test_contains_fail():
    assert contains("the answer is LHR airport", "JFK") == 0.0


def test_contains_all_full():
    assert contains_all("London Heathrow LHR", "London,LHR") == 1.0


def test_contains_all_partial():
    score = contains_all("London Heathrow", "London,LHR")
    assert score == 0.5


def test_contains_all_empty_expected():
    assert contains_all("anything", "") == 1.0


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------

def test_eval_case_auto_id():
    c = EvalCase("Hello world", "hi")
    assert c.id == "Hello_world"


def test_eval_case_explicit_id():
    c = EvalCase("q", "a", id="q1")
    assert c.id == "q1"


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------

def _make_report(passed: list[bool]) -> EvalReport:
    results = [
        EvalResult(
            case=EvalCase(f"q{i}", "exp"),
            actual="act",
            score=1.0 if p else 0.0,
            passed=p,
        )
        for i, p in enumerate(passed)
    ]
    return EvalReport(name="test", results=results)


def test_pass_rate_all_pass():
    r = _make_report([True, True, True])
    assert r.pass_rate() == 1.0


def test_pass_rate_half():
    r = _make_report([True, False])
    assert r.pass_rate() == 0.5


def test_pass_rate_empty():
    r = EvalReport("empty", [])
    assert r.pass_rate() == 0.0


def test_failed_returns_only_failures():
    r = _make_report([True, False, True])
    assert len(r.failed()) == 1


def test_summary_keys():
    r = _make_report([True, False])
    s = r.summary()
    assert {"name", "total", "passed", "failed", "pass_rate", "avg_score"} <= s.keys()


# ---------------------------------------------------------------------------
# EvalSuite.run
# ---------------------------------------------------------------------------

def test_suite_run_all_pass():
    cases = [
        EvalCase("LHR stands for Heathrow airport", "Heathrow"),
        EvalCase("Paris is the capital of France", "Paris"),
    ]
    # EchoModel repeats input; expected IS in the input → contains passes
    agent = Agent(model=EchoModel())
    suite = EvalSuite(cases=cases, scorer=contains, name="smoke")
    report = suite.run(agent)
    assert report.pass_rate() == 1.0


def test_suite_run_all_fail():
    cases = [EvalCase("hi", "NEVER_IN_RESPONSE")]
    agent = Agent(model=FixedModel("some other text"))
    suite = EvalSuite(cases=cases, scorer=contains)
    report = suite.run(agent)
    assert report.pass_rate() == 0.0


def test_suite_run_resets_memory():
    cases = [EvalCase("msg1", "msg1"), EvalCase("msg2", "msg2")]
    agent = Agent(model=EchoModel())
    suite = EvalSuite(cases=cases, scorer=contains)
    suite.run(agent, reset_memory=True)
    # Each case starts fresh, so memory has 2 messages per case, not 4
    assert len(agent.memory.get_messages()) == 2


def test_suite_run_handles_model_error():
    cases = [EvalCase("q", "a")]
    agent = Agent(model=FailModel())
    suite = EvalSuite(cases=cases)
    report = suite.run(agent)
    assert report.results[0].error is not None
    assert report.pass_rate() == 0.0


def test_suite_run_partial_scores():
    cases = [
        EvalCase("q", "London,LHR"),
        EvalCase("q2", "Paris,CDG"),
    ]
    # Model returns "London only" for all — partial match on first, no match on second
    agent = Agent(model=FixedModel("London only"))
    suite = EvalSuite(cases=cases, scorer=contains_all, threshold=0.6)
    report = suite.run(agent)
    # "London only" → London found (0.5 score), threshold=0.6 → fail
    assert report.results[0].passed is False
    assert report.results[0].score == 0.5
    for r in report.results:
        assert isinstance(r.score, float)


def test_suite_custom_scorer():
    def length_scorer(actual: str, expected: str) -> float:
        return 1.0 if len(actual) >= int(expected) else 0.0

    cases = [EvalCase("respond with at least 5 chars", "5")]
    agent = Agent(model=FixedModel("hello world"))
    suite = EvalSuite(cases=cases, scorer=length_scorer)
    report = suite.run(agent)
    assert report.pass_rate() == 1.0


# ---------------------------------------------------------------------------
# EvalSuite.arun
# ---------------------------------------------------------------------------

def test_suite_arun():
    async def run():
        cases = [EvalCase("hello", "hello")]
        agent = Agent(model=AsyncFixedModel("hello"))
        suite = EvalSuite(cases=cases, scorer=contains)
        report = await suite.arun(agent)
        assert report.pass_rate() == 1.0

    asyncio.run(run())


# ---------------------------------------------------------------------------
# EvalReport.print_report (smoke test — just must not raise)
# ---------------------------------------------------------------------------

def test_print_report_no_crash(capsys):
    r = _make_report([True, False])
    r.print_report()
    out = capsys.readouterr().out
    assert "✓" in out or "✗" in out
