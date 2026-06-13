"""Gemini model implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.models.message_adapters import (
    messages_to_gemini_contents,
    parse_gemini_response,
    tools_to_gemini,
)


class GeminiModel(ModelClient):
    def __init__(self, api_key: str, model_name: str, client: Any | None = None):
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

    def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Send a prompt to the Gemini model and return the response."""
        response = self.client.generate_content(prompt, **kwargs)
        return ModelResponse(content=getattr(response, "text", "") or "")

    def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        """Send messages to the Gemini model using the native content format.

        Passes ``tools`` to the API when provided so the model can request
        function calls.
        """
        tools_schemas: list[dict] = kwargs.pop("tools", None) or []
        system, contents = messages_to_gemini_contents(messages)

        call_kwargs: dict[str, Any] = dict(kwargs)
        if tools_schemas:
            call_kwargs["tools"] = tools_to_gemini(tools_schemas)

        # Inject system instruction as a leading model/user exchange when the
        # client does not support system_instruction directly.
        if system and contents:
            # Prepend as context to the first user turn (safe across SDK versions)
            first = contents[0]
            if first["role"] == "user":
                prefix = f"[System]: {system}\n\n"
                first_text = first["parts"][0].get("text", "")
                first["parts"] = [{"text": prefix + first_text}]

        response = self.client.generate_content(contents=contents, **call_kwargs)
        return parse_gemini_response(response)

    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the Gemini model and stream the response."""
        response = self.client.generate_content(prompt, stream=True, **kwargs)
        for chunk in response:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield text

    def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> Iterator[str]:
        """Stream a response from native Gemini content format."""
        _system, contents = messages_to_gemini_contents(messages)
        response = self.client.generate_content(
            contents=contents, stream=True, **kwargs
        )
        for chunk in response:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield text


class AsyncGeminiModel(AsyncModelClient):
    def __init__(self, api_key: str, model_name: str, client: Any | None = None):
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

    async def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Send a prompt to the Gemini model and return the response."""
        response = await self.client.generate_content_async(prompt, **kwargs)
        return ModelResponse(content=getattr(response, "text", "") or "")

    async def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        """Send messages with optional native tool calling support."""
        tools_schemas: list[dict] = kwargs.pop("tools", None) or []
        system, contents = messages_to_gemini_contents(messages)

        call_kwargs: dict[str, Any] = dict(kwargs)
        if tools_schemas:
            call_kwargs["tools"] = tools_to_gemini(tools_schemas)

        if system and contents:
            first = contents[0]
            if first["role"] == "user":
                prefix = f"[System]: {system}\n\n"
                first_text = first["parts"][0].get("text", "")
                first["parts"] = [{"text": prefix + first_text}]

        response = await self.client.generate_content_async(
            contents=contents, **call_kwargs
        )
        return parse_gemini_response(response)

    async def stream_response(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Send a prompt to the Gemini model and stream the response."""
        response = await self.client.generate_content_async(
            prompt, stream=True, **kwargs
        )
        async for chunk in response:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield text

    async def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a response from native Gemini content format."""
        _system, contents = messages_to_gemini_contents(messages)
        response = await self.client.generate_content_async(
            contents=contents, stream=True, **kwargs
        )
        async for chunk in response:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield text
