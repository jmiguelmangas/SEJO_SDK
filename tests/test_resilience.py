"""Tests for RetryModel, FallbackModel and async variants."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.resilience import (
    AsyncFallbackModel,
    AsyncRetryModel,
    FallbackModel,
    RetryModel,
)

# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------

class _OKModel(ModelClient):
    def __init__(self, reply: str = "ok") -> None:
        self.calls = 0
        self.reply = reply

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self.reply)

    def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self.reply)

    def stream_response(self, prompt: str, **_: Any):
        yield self.reply

    def stream_messages(self, messages: list[Message], **_: Any):
        yield self.reply


class _FailModel(ModelClient):
    """Always raises."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls = 0
        self._exc = exc or RuntimeError("model error")

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.calls += 1
        raise self._exc

    def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        self.calls += 1
        raise self._exc

    def stream_response(self, prompt: str, **_: Any):
        raise self._exc
        yield  # make it a generator

    def stream_messages(self, messages: list[Message], **_: Any):
        raise self._exc
        yield


class _FlakyModel(ModelClient):
    """Fails the first N calls, then succeeds."""

    def __init__(self, fail_times: int = 1, reply: str = "ok") -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.reply = reply

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient error")
        return ModelResponse(content=self.reply)

    def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        return self.send_prompt("", **_)

    def stream_response(self, prompt: str, **_: Any):
        yield self.reply

    def stream_messages(self, messages: list[Message], **_: Any):
        yield self.reply


class _AsyncOKModel(AsyncModelClient):
    def __init__(self, reply: str = "async ok") -> None:
        self.reply = reply
        self.calls = 0

    async def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self.reply)

    async def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self.reply)

    async def stream_response(self, prompt: str, **_: Any):
        yield self.reply

    async def stream_messages(self, messages: list[Message], **_: Any):
        yield self.reply


class _AsyncFailModel(AsyncModelClient):
    def __init__(self) -> None:
        self.calls = 0

    async def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.calls += 1
        raise RuntimeError("async error")

    async def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        self.calls += 1
        raise RuntimeError("async error")

    async def stream_response(self, prompt: str, **_: Any):
        raise RuntimeError("async error")
        yield

    async def stream_messages(self, messages: list[Message], **_: Any):
        raise RuntimeError("async error")
        yield


class _AsyncFlakyModel(AsyncModelClient):
    def __init__(self, fail_times: int = 1, reply: str = "async ok") -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.reply = reply

    async def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return ModelResponse(content=self.reply)

    async def send_messages(self, messages: list[Message], **_: Any) -> ModelResponse:
        return await self.send_prompt("")

    async def stream_response(self, prompt: str, **_: Any):
        yield self.reply

    async def stream_messages(self, messages: list[Message], **_: Any):
        yield self.reply


# ---------------------------------------------------------------------------
# RetryModel
# ---------------------------------------------------------------------------

def test_retry_succeeds_on_first_try():
    model = _OKModel("hello")
    retry = RetryModel(model, max_retries=3, delay=0)
    result = retry.send_prompt("hi")
    assert result.content == "hello"
    assert model.calls == 1


def test_retry_retries_until_success():
    flaky = _FlakyModel(fail_times=2, reply="recovered")
    retry = RetryModel(flaky, max_retries=3, delay=0)
    result = retry.send_prompt("hi")
    assert result.content == "recovered"
    assert flaky.calls == 3


def test_retry_exhausted_raises():
    fail = _FailModel()
    retry = RetryModel(fail, max_retries=2, delay=0)
    with pytest.raises(RuntimeError, match="model error"):
        retry.send_prompt("hi")
    assert fail.calls == 3  # 1 initial + 2 retries


def test_retry_stream_passes_through():
    model = _OKModel("chunk")
    retry = RetryModel(model, delay=0)
    result = list(retry.stream_response("hi"))
    assert result == ["chunk"]


# ---------------------------------------------------------------------------
# FallbackModel
# ---------------------------------------------------------------------------

def test_fallback_uses_first_on_success():
    m1 = _OKModel("primary")
    m2 = _OKModel("secondary")
    fb = FallbackModel(m1, m2)
    result = fb.send_prompt("hi")
    assert result.content == "primary"
    assert m1.calls == 1
    assert m2.calls == 0


def test_fallback_skips_to_next_on_failure():
    m1 = _FailModel()
    m2 = _OKModel("fallback")
    fb = FallbackModel(m1, m2)
    result = fb.send_prompt("hi")
    assert result.content == "fallback"


def test_fallback_all_fail_raises():
    fb = FallbackModel(_FailModel(), _FailModel())
    with pytest.raises(RuntimeError, match="All 2 models failed"):
        fb.send_prompt("hi")


def test_fallback_requires_at_least_one_model():
    with pytest.raises(ValueError):
        FallbackModel()


def test_fallback_stream():
    m1 = _FailModel()
    m2 = _OKModel("stream fallback")
    fb = FallbackModel(m1, m2)
    result = list(fb.stream_response("hi"))
    assert result == ["stream fallback"]


# ---------------------------------------------------------------------------
# AsyncRetryModel
# ---------------------------------------------------------------------------

def test_async_retry_success():
    async def run():
        model = _AsyncFlakyModel(fail_times=1, reply="async recovered")
        retry = AsyncRetryModel(model, max_retries=3, delay=0)
        result = await retry.send_prompt("hi")
        assert result.content == "async recovered"
        assert model.calls == 2

    asyncio.run(run())


def test_async_retry_exhausted():
    async def run():
        model = _AsyncFailModel()
        retry = AsyncRetryModel(model, max_retries=2, delay=0)
        with pytest.raises(RuntimeError, match="async error"):
            await retry.send_prompt("hi")
        assert model.calls == 3

    asyncio.run(run())


# ---------------------------------------------------------------------------
# AsyncFallbackModel
# ---------------------------------------------------------------------------

def test_async_fallback_uses_first():
    async def run():
        m1 = _AsyncOKModel("async primary")
        m2 = _AsyncOKModel("async secondary")
        fb = AsyncFallbackModel(m1, m2)
        result = await fb.send_prompt("hi")
        assert result.content == "async primary"
        assert m1.calls == 1
        assert m2.calls == 0

    asyncio.run(run())


def test_async_fallback_skips_to_next():
    async def run():
        m1 = _AsyncFailModel()
        m2 = _AsyncOKModel("async fallback")
        fb = AsyncFallbackModel(m1, m2)
        result = await fb.send_prompt("hi")
        assert result.content == "async fallback"

    asyncio.run(run())


def test_async_fallback_all_fail():
    async def run():
        fb = AsyncFallbackModel(_AsyncFailModel(), _AsyncFailModel())
        with pytest.raises(RuntimeError, match="All 2 models failed"):
            await fb.send_prompt("hi")

    asyncio.run(run())
