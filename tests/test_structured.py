"""Tests for structured output parsing."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.structured import _extract_json, parse_structured, schema_prompt

try:
    from pydantic import BaseModel

    _PYDANTIC = True

    class FlightInfo(BaseModel):
        flight: str
        destination: str
        seats: int = 0

    class Crew(BaseModel):
        name: str
        role: str

except ImportError:
    _PYDANTIC = False
    FlightInfo = None  # type: ignore[assignment,misc]
    Crew = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(not _PYDANTIC, reason="pydantic not installed")


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

def test_extract_plain_json():
    text = '{"flight": "BA123", "destination": "LHR"}'
    assert _extract_json(text) == text


def test_extract_from_markdown_fence():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == '{"a": 1}'


def test_extract_from_plain_fence():
    text = '```\n{"a": 1}\n```'
    assert _extract_json(text) == '{"a": 1}'


def test_extract_json_embedded_in_text():
    text = 'Here is the result: {"flight": "BA1", "destination": "MAD"} — enjoy!'
    extracted = _extract_json(text)
    assert '"flight"' in extracted


# ---------------------------------------------------------------------------
# parse_structured
# ---------------------------------------------------------------------------

def test_parse_structured_plain_json():
    text = '{"flight": "BA456", "destination": "JFK", "seats": 12}'
    result = parse_structured(text, FlightInfo)
    assert result.flight == "BA456"
    assert result.destination == "JFK"
    assert result.seats == 12


def test_parse_structured_markdown_fence():
    text = '```json\n{"flight": "BA789", "destination": "CDG"}\n```'
    result = parse_structured(text, FlightInfo)
    assert result.flight == "BA789"


def test_parse_structured_invalid_json_raises():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_structured("not json at all", FlightInfo)


def test_parse_structured_wrong_schema_raises():
    # flight is required — omitting it should fail
    text = '{"destination": "LHR"}'
    with pytest.raises(ValueError, match="does not match schema"):
        parse_structured(text, FlightInfo)


# ---------------------------------------------------------------------------
# schema_prompt
# ---------------------------------------------------------------------------

def test_schema_prompt_contains_json_schema():
    prompt = schema_prompt(FlightInfo)
    assert "flight" in prompt
    assert "destination" in prompt
    assert "JSON" in prompt


# ---------------------------------------------------------------------------
# Agent.run_structured integration
# ---------------------------------------------------------------------------

class _FakeModel(ModelClient):
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return ModelResponse(content=self._reply)

    def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        return ModelResponse(content=self._reply)

    def stream_response(self, prompt: str, **_: Any):
        yield self._reply

    def stream_messages(self, messages: list[Message], **_: Any):
        yield self._reply


class _AsyncFakeModel(AsyncModelClient):
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        return ModelResponse(content=self._reply)

    async def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        return ModelResponse(content=self._reply)

    async def stream_response(self, prompt: str, **_: Any):
        yield self._reply

    async def stream_messages(self, messages: list[Message], **_: Any):
        yield self._reply


def test_agent_run_structured():
    from SEJO_SDK.agent import Agent

    reply = '{"flight": "BA100", "destination": "SYD", "seats": 5}'
    agent = Agent(model=_FakeModel(reply))
    result = agent.run_structured("Extract flight info", FlightInfo)
    assert isinstance(result, FlightInfo)
    assert result.flight == "BA100"
    assert result.destination == "SYD"
    assert result.seats == 5


def test_agent_arun_structured():
    async def run():
        from SEJO_SDK.agent import Agent

        reply = '{"flight": "BA200", "destination": "NYC"}'
        agent = Agent(model=_AsyncFakeModel(reply))
        result = await agent.arun_structured("Extract flight info", FlightInfo)
        assert isinstance(result, FlightInfo)
        assert result.flight == "BA200"

    asyncio.run(run())
