"""Anthropic model implementation."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.messages import Message, ModelResponse, user_message
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.models.message_adapters import (
    parse_anthropic_response,
    split_anthropic_messages,
    tools_to_anthropic,
)


class AnthropicModel(ModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_tokens: int = 1024,
        client: Optional[Any] = None,
    ):
        """Initialize the Anthropic model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        if client is not None:
            self.client = client
        else:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install Anthropic support with `pip install sejo-sdk[anthropic]`."
                ) from exc
            self.client = Anthropic(api_key=api_key)

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the Anthropic model and return the response."""
        return self.send_messages([user_message(prompt)], **kwargs)

    def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        """Send messages to the Anthropic model and return the response."""
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        system, provider_messages = split_anthropic_messages(messages)
        if system is not None:
            kwargs["system"] = system
        if "tools" in kwargs:
            kwargs["tools"] = tools_to_anthropic(kwargs["tools"])
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=provider_messages,
            **kwargs,
        )
        return parse_anthropic_response(response)

    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the Anthropic model and return the response as a stream."""
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


class AsyncAnthropicModel(AsyncModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_tokens: int = 1024,
        client: Optional[Any] = None,
    ):
        """Initialize the async Anthropic model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        if client is not None:
            self.client = client
        else:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install Anthropic support with `pip install sejo-sdk[anthropic]`."
                ) from exc
            self.client = AsyncAnthropic(api_key=api_key)

    async def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the Anthropic model and return the response."""
        return await self.send_messages([user_message(prompt)], **kwargs)

    async def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        """Send messages to the Anthropic model and return the response."""
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        system, provider_messages = split_anthropic_messages(messages)
        if system is not None:
            kwargs["system"] = system
        if "tools" in kwargs:
            kwargs["tools"] = tools_to_anthropic(kwargs["tools"])
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=provider_messages,
            **kwargs,
        )
        return parse_anthropic_response(response)

    async def stream_response(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Send a prompt to the Anthropic model and return the response as a stream."""
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text
