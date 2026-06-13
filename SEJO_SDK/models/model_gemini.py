"""Gemini model implementation."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.messages import Message
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.models.message_adapters import messages_to_gemini_prompt


class GeminiModel(ModelClient):
    def __init__(self, api_key: str, model_name: str, client: Optional[Any] = None):
        """Initialize the Gemini model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        if client is not None:
            self.client = client
        else:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install Gemini support with `pip install sejo-sdk[gemini]`."
                ) from exc
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the Gemini model and return the response."""
        response = self.client.generate_content(prompt, **kwargs)
        return getattr(response, "text", "") or ""

    def send_messages(self, messages: list[Message], **kwargs: Any) -> str:
        """Send messages to the Gemini model and return the response."""
        return self.send_prompt(messages_to_gemini_prompt(messages), **kwargs)

    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the Gemini model and return the response as a stream."""
        response = self.client.generate_content(prompt, stream=True, **kwargs)
        for chunk in response:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield text


class AsyncGeminiModel(AsyncModelClient):
    def __init__(self, api_key: str, model_name: str, client: Optional[Any] = None):
        """Initialize the async Gemini model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        if client is not None:
            self.client = client
        else:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install Gemini support with `pip install sejo-sdk[gemini]`."
                ) from exc
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)

    async def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the Gemini model and return the response."""
        response = await self.client.generate_content_async(prompt, **kwargs)
        return getattr(response, "text", "") or ""

    async def send_messages(self, messages: list[Message], **kwargs: Any) -> str:
        """Send messages to the Gemini model and return the response."""
        return await self.send_prompt(messages_to_gemini_prompt(messages), **kwargs)

    async def stream_response(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Send a prompt to the Gemini model and return the response as a stream."""
        response = await self.client.generate_content_async(
            prompt,
            stream=True,
            **kwargs,
        )
        async for chunk in response:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield text
