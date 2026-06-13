"""DeepSeek model implementation."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional, Union

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.messages import Message, ModelResponse, user_message
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.models.openai_compat import messages_to_openai, parse_chat_message


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

    def send_prompt(self, prompt: str, **kwargs: Any) -> Union[str, ModelResponse]:
        """Send a prompt to the DeepSeek model and return the response."""
        return self.send_messages([user_message(prompt)], **kwargs)

    def send_messages(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Union[str, ModelResponse]:
        """Send messages to the DeepSeek model and return the response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_to_openai(messages),
            **kwargs,
        )
        return parse_chat_message(response.choices[0].message)

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


class AsyncDeepSeekModel(AsyncModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://api.deepseek.com",
        client: Optional[Any] = None,
    ):
        """Initialize the async DeepSeek model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        if client is not None:
            self.client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install DeepSeek support with `pip install sejo-sdk[openai]`."
                ) from exc
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def send_prompt(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Union[str, ModelResponse]:
        """Send a prompt to the DeepSeek model and return the response."""
        return await self.send_messages([user_message(prompt)], **kwargs)

    async def send_messages(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Union[str, ModelResponse]:
        """Send messages to the DeepSeek model and return the response."""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_to_openai(messages),
            **kwargs,
        )
        return parse_chat_message(response.choices[0].message)

    async def stream_response(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Send a prompt to the DeepSeek model and return the response as a stream."""
        response = await self.client.chat.completions.create(
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
        async for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
