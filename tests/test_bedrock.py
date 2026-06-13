"""Tests for AWS Bedrock model adapter using fake clients."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from typing import Any

from SEJO_SDK.messages import system_message, user_message
from SEJO_SDK.models.bedrock_adapters import (
    build_request_body,
    detect_family,
    parse_response_body,
    parse_stream_chunk,
)
from SEJO_SDK.models.model_bedrock import AsyncBedrockModel, BedrockModel

# Model IDs used across tests
CLAUDE_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
TITAN_ID = "amazon.titan-text-express-v1"
LLAMA_ID = "meta.llama3-8b-instruct-v1:0"
MISTRAL_ID = "mistral.mistral-7b-instruct-v0:2"

# ---------------------------------------------------------------------------
# Fake boto3 clients
# ---------------------------------------------------------------------------

def _make_response_body(payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload).encode()
    return {"body": BytesIO(raw)}


def _make_stream_body(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    events = [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]
    return {"body": iter(events)}


class FakeBedrockClient:
    """Mimics boto3 bedrock-runtime client."""

    def __init__(
        self, response_payload: dict[str, Any], stream_chunks: list | None = None
    ):
        self.calls: list[dict[str, Any]] = []
        self._response = response_payload
        self._stream_chunks = stream_chunks or []

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "invoke_model", **kwargs})
        return _make_response_body(self._response)

    def invoke_model_with_response_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "invoke_model_with_response_stream", **kwargs})
        return _make_stream_body(self._stream_chunks)


# ---------------------------------------------------------------------------
# detect_family
# ---------------------------------------------------------------------------

def test_detect_family_claude():
    assert detect_family(CLAUDE_ID) == "claude"


def test_detect_family_titan():
    assert detect_family(TITAN_ID) == "titan"


def test_detect_family_llama():
    assert detect_family(LLAMA_ID) == "llama"


def test_detect_family_mistral():
    assert detect_family(MISTRAL_ID) == "mistral"


def test_detect_family_unknown():
    assert detect_family("cohere.command-r-v1:0") == "unknown"


# ---------------------------------------------------------------------------
# build_request_body
# ---------------------------------------------------------------------------

def test_claude_request_body_has_messages():
    body = json.loads(
        build_request_body(
            [user_message("Hello")],
            CLAUDE_ID,
            max_tokens=512,
        )
    )
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 512
    assert body["messages"] == [{"role": "user", "content": "Hello"}]


def test_claude_request_body_extracts_system():
    body = json.loads(
        build_request_body(
            [system_message("Be concise."), user_message("Hi")],
            CLAUDE_ID,
        )
    )
    assert body["system"] == "Be concise."
    assert body["messages"] == [{"role": "user", "content": "Hi"}]


def test_titan_request_body_has_input_text():
    body = json.loads(
        build_request_body(
            [user_message("Hello")],
            TITAN_ID,
            max_tokens=256,
        )
    )
    assert "Human: Hello" in body["inputText"]
    assert body["textGenerationConfig"]["maxTokenCount"] == 256


def test_llama_request_body():
    body = json.loads(
        build_request_body(
            [user_message("Hello")],
            LLAMA_ID,
            max_tokens=128,
        )
    )
    assert "Human: Hello" in body["prompt"]
    assert body["max_gen_len"] == 128


def test_mistral_request_body():
    body = json.loads(
        build_request_body(
            [user_message("Hello")],
            MISTRAL_ID,
            max_tokens=64,
        )
    )
    assert "Human: Hello" in body["prompt"]
    assert body["max_tokens"] == 64


# ---------------------------------------------------------------------------
# parse_response_body
# ---------------------------------------------------------------------------

def test_parse_claude_response():
    payload = {
        "content": [{"type": "text", "text": "Hello from Claude"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    result = parse_response_body(json.dumps(payload).encode(), CLAUDE_ID)
    assert result.content == "Hello from Claude"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_parse_titan_response():
    payload = {
        "results": [{"outputText": "Titan response", "tokenCount": 8}],
        "inputTextTokenCount": 12,
    }
    result = parse_response_body(json.dumps(payload).encode(), TITAN_ID)
    assert result.content == "Titan response"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 8


def test_parse_llama_response():
    payload = {
        "generation": "Llama response",
        "prompt_token_count": 7,
        "generation_token_count": 4,
    }
    result = parse_response_body(json.dumps(payload).encode(), LLAMA_ID)
    assert result.content == "Llama response"
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 4


def test_parse_mistral_response():
    payload = {"outputs": [{"text": "Mistral response"}]}
    result = parse_response_body(json.dumps(payload).encode(), MISTRAL_ID)
    assert result.content == "Mistral response"


# ---------------------------------------------------------------------------
# parse_stream_chunk
# ---------------------------------------------------------------------------

def test_parse_claude_stream_chunk():
    chunk = {"type": "content_block_delta", "delta": {"text": "Hello"}}
    assert parse_stream_chunk(json.dumps(chunk).encode(), CLAUDE_ID) == "Hello"


def test_parse_titan_stream_chunk():
    chunk = {"outputText": "Titan chunk"}
    assert parse_stream_chunk(json.dumps(chunk).encode(), TITAN_ID) == "Titan chunk"


# ---------------------------------------------------------------------------
# BedrockModel (sync)
# ---------------------------------------------------------------------------

def test_bedrock_model_send_prompt_claude():
    payload = {
        "content": [{"type": "text", "text": "Hi from Bedrock Claude"}],
        "usage": {"input_tokens": 5, "output_tokens": 10},
    }
    client = FakeBedrockClient(payload)
    model = BedrockModel(
        model_id=CLAUDE_ID, client=client
    )

    result = model.send_prompt("Hello")

    assert result.content == "Hi from Bedrock Claude"
    assert result.usage.input_tokens == 5
    assert client.calls[0]["modelId"] == CLAUDE_ID


def test_bedrock_model_send_messages_with_system():
    payload = {"content": [{"type": "text", "text": "OK"}], "usage": {}}
    client = FakeBedrockClient(payload)
    model = BedrockModel(
        model_id=CLAUDE_ID, client=client
    )

    model.send_messages([system_message("Be brief."), user_message("Hi")])

    sent_body = json.loads(client.calls[0]["body"])
    assert sent_body["system"] == "Be brief."


def test_bedrock_model_titan_send_prompt():
    payload = {
        "results": [{"outputText": "Titan says hello", "tokenCount": 4}],
        "inputTextTokenCount": 6,
    }
    client = FakeBedrockClient(payload)
    model = BedrockModel(model_id=TITAN_ID, client=client)

    result = model.send_prompt("Hello")

    assert result.content == "Titan says hello"


def test_bedrock_model_stream_response():
    chunks = [
        {"type": "content_block_delta", "delta": {"text": "Hello"}},
        {"type": "content_block_delta", "delta": {"text": " world"}},
    ]
    client = FakeBedrockClient({}, stream_chunks=chunks)
    model = BedrockModel(
        model_id=CLAUDE_ID, client=client
    )

    result = list(model.stream_response("Hi"))

    assert result == ["Hello", " world"]


def test_bedrock_model_max_tokens_override():
    payload = {"content": [{"type": "text", "text": "ok"}], "usage": {}}
    client = FakeBedrockClient(payload)
    model = BedrockModel(
        model_id=CLAUDE_ID,
        max_tokens=512,
        client=client,
    )

    model.send_messages([user_message("Hi")], max_tokens=128)

    sent_body = json.loads(client.calls[0]["body"])
    assert sent_body["max_tokens"] == 128


# ---------------------------------------------------------------------------
# AsyncBedrockModel
# ---------------------------------------------------------------------------

def test_async_bedrock_model_send_prompt():
    async def exercise():
        payload = {
            "content": [{"type": "text", "text": "Async Claude"}],
            "usage": {"input_tokens": 3, "output_tokens": 6},
        }
        client = FakeBedrockClient(payload)
        model = AsyncBedrockModel(
            model_id=CLAUDE_ID, client=client
        )

        result = await model.send_prompt("Hello async")

        assert result.content == "Async Claude"
        assert result.usage.output_tokens == 6

    asyncio.run(exercise())


def test_async_bedrock_model_stream():
    async def exercise():
        chunks = [
            {"type": "content_block_delta", "delta": {"text": "async"}},
            {"type": "content_block_delta", "delta": {"text": " stream"}},
        ]
        client = FakeBedrockClient({}, stream_chunks=chunks)
        model = AsyncBedrockModel(
            model_id=CLAUDE_ID, client=client
        )

        result = []
        async for chunk in model.stream_response("Hi"):
            result.append(chunk)

        assert result == ["async", " stream"]

    asyncio.run(exercise())


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

def test_bedrock_agent_run():
    from SEJO_SDK.agent import Agent

    payload = {
        "content": [{"type": "text", "text": "Bedrock agent response"}],
        "usage": {"input_tokens": 8, "output_tokens": 4},
    }
    client = FakeBedrockClient(payload)
    model = BedrockModel(
        model_id=CLAUDE_ID,
        client=client,
    )
    agent = Agent(model=model, system_prompt="You are a BA assistant.")

    response = agent.run("What flights are available?")

    assert response == "Bedrock agent response"
    sent_body = json.loads(client.calls[0]["body"])
    assert sent_body["system"] == "You are a BA assistant."
