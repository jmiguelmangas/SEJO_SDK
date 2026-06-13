"""Retry and fallback wrappers for ModelClient / AsyncModelClient."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import AsyncModelClient, ModelClient


class RetryModel(ModelClient):
    """Wraps a ModelClient and retries on exception with exponential back-off.

    Example::

        model = RetryModel(OpenAIModel("gpt-4o"), max_retries=3, delay=1.0)
    """

    def __init__(
        self,
        model: ModelClient,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions

    def _retry(self, fn, *args, **kwargs):
        wait = self.delay
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except self.exceptions as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(wait)
                    wait *= self.backoff
        raise last_exc  # type: ignore[misc]

    def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        return self._retry(self.model.send_prompt, prompt, **kwargs)

    def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        return self._retry(self.model.send_messages, messages, **kwargs)

    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield from self.model.stream_response(prompt, **kwargs)

    def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> Iterator[str]:
        yield from self.model.stream_messages(messages, **kwargs)


class AsyncRetryModel(AsyncModelClient):
    """Wraps an AsyncModelClient and retries on exception with exponential back-off."""

    def __init__(
        self,
        model: AsyncModelClient,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions

    async def _retry(self, coro_fn, *args, **kwargs):
        wait = self.delay
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except self.exceptions as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(wait)
                    wait *= self.backoff
        raise last_exc  # type: ignore[misc]

    async def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        return await self._retry(self.model.send_prompt, prompt, **kwargs)

    async def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        return await self._retry(self.model.send_messages, messages, **kwargs)

    async def stream_response(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        async for chunk in self.model.stream_response(prompt, **kwargs):
            yield chunk

    async def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        async for chunk in self.model.stream_messages(messages, **kwargs):
            yield chunk


class FallbackModel(ModelClient):
    """Tries each ModelClient in order, returning the first successful response.

    Example::

        model = FallbackModel(
            OpenAIModel("gpt-4o"),
            AnthropicModel("claude-3-5-haiku-20241022"),
        )
    """

    def __init__(self, *models: ModelClient) -> None:
        if not models:
            raise ValueError("FallbackModel requires at least one model.")
        self.models = models

    def _try_all(self, method: str, *args, **kwargs) -> ModelResponse:
        last_exc: Exception | None = None
        for model in self.models:
            try:
                return getattr(model, method)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_exc}"
        ) from last_exc

    def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        return self._try_all("send_prompt", prompt, **kwargs)

    def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        return self._try_all("send_messages", messages, **kwargs)

    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        last_exc: Exception | None = None
        for model in self.models:
            try:
                yield from model.stream_response(prompt, **kwargs)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_exc}"
        ) from last_exc

    def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> Iterator[str]:
        last_exc: Exception | None = None
        for model in self.models:
            try:
                yield from model.stream_messages(messages, **kwargs)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_exc}"
        ) from last_exc


class AsyncFallbackModel(AsyncModelClient):
    """Async version of FallbackModel."""

    def __init__(self, *models: AsyncModelClient) -> None:
        if not models:
            raise ValueError("AsyncFallbackModel requires at least one model.")
        self.models = models

    async def _try_all(self, method: str, *args, **kwargs) -> ModelResponse:
        last_exc: Exception | None = None
        for model in self.models:
            try:
                return await getattr(model, method)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_exc}"
        ) from last_exc

    async def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        return await self._try_all("send_prompt", prompt, **kwargs)

    async def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        return await self._try_all("send_messages", messages, **kwargs)

    async def stream_response(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        last_exc: Exception | None = None
        for model in self.models:
            try:
                async for chunk in model.stream_response(prompt, **kwargs):
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_exc}"
        ) from last_exc

    async def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        last_exc: Exception | None = None
        for model in self.models:
            try:
                async for chunk in model.stream_messages(messages, **kwargs):
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_exc}"
        ) from last_exc
