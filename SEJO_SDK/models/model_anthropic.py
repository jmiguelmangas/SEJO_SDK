"""Anthropic model implementation."""

from collections.abc import Iterator
from typing import Any, Optional

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.model import ModelClient


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
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )
        return self._extract_text(response.content)

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
