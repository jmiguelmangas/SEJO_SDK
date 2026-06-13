"""OpenAI model implementation."""

from collections.abc import Iterator
from typing import Any, Optional

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.model import ModelClient


class OpenAIModel(ModelClient):
    def __init__(self, api_key: str, model_name: str, client: Optional[Any] = None):
        """Initialize the OpenAI model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install OpenAI support with `pip install sejo-sdk[openai]`."
                ) from exc
            self.client = OpenAI(api_key=api_key)

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the OpenAI model and return the response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the OpenAI model and return the response as a stream."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            **kwargs,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
