"""AWS Bedrock model implementation.

Supports: anthropic.claude-*, amazon.titan-*, meta.llama*, mistral.*

Authentication follows standard boto3 credential resolution:
  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
  - ~/.aws/credentials profile
  - IAM role (EC2 / ECS / Lambda)

Usage:
    model = BedrockModel(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="eu-west-1",
    )
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from SEJO_SDK.errors import ProviderDependencyError
from SEJO_SDK.messages import Message, ModelResponse, user_message
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.models.bedrock_adapters import (
    build_request_body,
    parse_response_body,
    parse_stream_chunk,
)

_DEFAULT_MAX_TOKENS = 1024


class BedrockModel(ModelClient):
    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client: Any = None,
    ):
        """Sync Bedrock model adapter.

        Args:
            model_id:    Bedrock model ID (e.g. 'anthropic.claude-3-5-haiku-*').
            region_name: AWS region where Bedrock is enabled.
            max_tokens:  Default max tokens for responses.
            client:      Inject a pre-built boto3 bedrock-runtime client.
        """
        self.model_id = model_id
        self.region_name = region_name
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install AWS Bedrock support with `pip install sejo-sdk[bedrock]`."
                ) from exc
            self._client = boto3.client(
                "bedrock-runtime", region_name=region_name
            )

    def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        return self.send_messages([user_message(prompt)], **kwargs)

    def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        body = build_request_body(messages, self.model_id, max_tokens, **kwargs)
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        raw = response["body"].read()
        return parse_response_body(raw, self.model_id)

    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield from self.stream_messages([user_message(prompt)], **kwargs)

    def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> Iterator[str]:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        body = build_request_body(messages, self.model_id, max_tokens, **kwargs)
        response = self._client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        for event in response["body"]:
            chunk_bytes = event.get("chunk", {}).get("bytes", b"")
            if chunk_bytes:
                text = parse_stream_chunk(chunk_bytes, self.model_id)
                if text:
                    yield text


class AsyncBedrockModel(AsyncModelClient):
    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client: Any = None,
    ):
        """Async Bedrock model adapter (wraps boto3 calls in a thread pool).

        Args:
            model_id:    Bedrock model ID.
            region_name: AWS region where Bedrock is enabled.
            max_tokens:  Default max tokens for responses.
            client:      Inject a pre-built boto3 bedrock-runtime client.
        """
        self.model_id = model_id
        self.region_name = region_name
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise ProviderDependencyError(
                    "Install AWS Bedrock support with `pip install sejo-sdk[bedrock]`."
                ) from exc
            self._client = boto3.client(
                "bedrock-runtime", region_name=region_name
            )

    async def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        return await self.send_messages([user_message(prompt)], **kwargs)

    async def send_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> ModelResponse:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        body = build_request_body(messages, self.model_id, max_tokens, **kwargs)

        def _invoke() -> bytes:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            return response["body"].read()

        raw = await asyncio.to_thread(_invoke)
        return parse_response_body(raw, self.model_id)

    async def stream_response(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        async for chunk in self.stream_messages([user_message(prompt)], **kwargs):
            yield chunk

    async def stream_messages(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        body = build_request_body(messages, self.model_id, max_tokens, **kwargs)

        def _invoke_stream() -> list[str]:
            response = self._client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            chunks = []
            for event in response["body"]:
                chunk_bytes = event.get("chunk", {}).get("bytes", b"")
                if chunk_bytes:
                    text = parse_stream_chunk(chunk_bytes, self.model_id)
                    if text:
                        chunks.append(text)
            return chunks

        for text in await asyncio.to_thread(_invoke_stream):
            yield text
