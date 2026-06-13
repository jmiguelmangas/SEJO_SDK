"""Provider-agnostic model client interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any, Union

from SEJO_SDK.messages import Message, ModelResponse, messages_to_prompt

ModelOutput = Union[str, ModelResponse]


class ModelClient(ABC):
    """Base class for model adapters used by agents."""

    @abstractmethod
    def send_prompt(self, prompt: str, **kwargs: Any) -> ModelOutput:
        """Send a prompt to the model and return the response."""

        raise NotImplementedError

    def send_messages(self, messages: list[Message], **kwargs: Any) -> ModelOutput:
        """Send native chat messages to the model.

        Custom adapters that only implement prompts inherit a compatible
        fallback that renders messages into the legacy prompt format.
        """

        return self.send_prompt(f"{messages_to_prompt(messages)}\nAgent:", **kwargs)

    def stream_messages(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream a response for a list of chat messages.

        Custom adapters that only implement stream_response inherit a fallback
        that renders messages into the legacy prompt format.
        """

        yield from self.stream_response(
            f"{messages_to_prompt(messages)}\nAgent:", **kwargs
        )

    @abstractmethod
    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the model and return the response as a stream."""

        raise NotImplementedError


class AsyncModelClient(ABC):
    """Base class for async model adapters used by agents."""

    @abstractmethod
    async def send_prompt(self, prompt: str, **kwargs: Any) -> ModelOutput:
        """Send a prompt to the model and return the response."""

        raise NotImplementedError

    async def send_messages(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> ModelOutput:
        """Send native chat messages to the model.

        Custom async adapters that only implement prompts inherit a compatible
        fallback that renders messages into the legacy prompt format.
        """

        return await self.send_prompt(
            f"{messages_to_prompt(messages)}\nAgent:",
            **kwargs,
        )

    async def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a response for a list of chat messages.

        Custom async adapters that only implement stream_response inherit a
        fallback that renders messages into the legacy prompt format.
        """

        async for chunk in self.stream_response(
            f"{messages_to_prompt(messages)}\nAgent:", **kwargs
        ):
            yield chunk

    @abstractmethod
    def stream_response(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Send a prompt to the model and return the response as an async stream."""

        raise NotImplementedError


# Backwards-compatible alias for older imports.
Model_client = ModelClient
AsyncModel_client = AsyncModelClient
