"""DeepSeek model implementation."""

from collections.abc import Iterator
from typing import Any, Optional

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.model import ModelClient


class DeepSeekModel(ModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://api.deepseek.com",
        client: Optional[Any] = None,
    ):
        """Initialize the DeepSeek model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install DeepSeek support with `pip install sejo-sdk[openai]`."
                ) from exc
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the DeepSeek model and return the response."""
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
        """Send a prompt to the DeepSeek model and return the response as a stream."""
        for chunk in self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            **kwargs,
        ):
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
