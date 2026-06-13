"""AWS Bedrock request/response adapters per model family.

Model ID prefixes:
    anthropic.claude-*   → Claude (Anthropic messages API via Bedrock)
    amazon.titan-*       → Amazon Titan
    meta.llama*          → Meta Llama
    mistral.*            → Mistral AI
"""

from __future__ import annotations

import json
from typing import Any

from SEJO_SDK.messages import Message, ModelResponse, ToolCall, Usage
from SEJO_SDK.models.message_adapters import split_anthropic_messages


def detect_family(model_id: str) -> str:
    if model_id.startswith("anthropic."):
        return "claude"
    if model_id.startswith("amazon.titan"):
        return "titan"
    if model_id.startswith("meta.llama"):
        return "llama"
    if model_id.startswith("mistral."):
        return "mistral"
    return "unknown"


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def build_request_body(
    messages: list[Message],
    model_id: str,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> str:
    family = detect_family(model_id)
    if family == "claude":
        return _claude_body(messages, max_tokens, **kwargs)
    if family == "titan":
        return _titan_body(messages, max_tokens, **kwargs)
    if family == "llama":
        return _llama_body(messages, max_tokens, **kwargs)
    if family == "mistral":
        return _mistral_body(messages, max_tokens, **kwargs)
    raise ValueError(f"Unsupported Bedrock model family for model_id: {model_id!r}")


def _claude_body(messages: list[Message], max_tokens: int, **kwargs: Any) -> str:
    system, provider_messages = split_anthropic_messages(messages)
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": provider_messages,
    }
    if system:
        body["system"] = system
    body.update(kwargs)
    return json.dumps(body)


def _titan_body(messages: list[Message], max_tokens: int, **kwargs: Any) -> str:
    prompt = _messages_to_prompt(messages)
    body: dict[str, Any] = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": max_tokens,
            "temperature": kwargs.pop("temperature", 0.7),
            "topP": kwargs.pop("top_p", 1.0),
        },
    }
    body.update(kwargs)
    return json.dumps(body)


def _llama_body(messages: list[Message], max_tokens: int, **kwargs: Any) -> str:
    body: dict[str, Any] = {
        "prompt": _messages_to_prompt(messages),
        "max_gen_len": max_tokens,
        "temperature": kwargs.pop("temperature", 0.5),
    }
    body.update(kwargs)
    return json.dumps(body)


def _mistral_body(messages: list[Message], max_tokens: int, **kwargs: Any) -> str:
    body: dict[str, Any] = {
        "prompt": _messages_to_prompt(messages),
        "max_tokens": max_tokens,
        "temperature": kwargs.pop("temperature", 0.7),
    }
    body.update(kwargs)
    return json.dumps(body)


def _messages_to_prompt(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"System: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
        else:
            parts.append(f"Human: {m.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_response_body(raw_body: bytes | str, model_id: str) -> ModelResponse:
    body = json.loads(raw_body) if isinstance(raw_body, bytes) else json.loads(raw_body)
    family = detect_family(model_id)
    if family == "claude":
        return _parse_claude(body)
    if family == "titan":
        return _parse_titan(body)
    if family == "llama":
        return _parse_llama(body)
    if family == "mistral":
        return _parse_mistral(body)
    return ModelResponse(content=str(body))


def _parse_claude(body: dict[str, Any]) -> ModelResponse:
    text_parts = []
    tool_calls: list[ToolCall] = []
    for block in body.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id"),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                )
            )
    raw_usage = body.get("usage", {})
    usage = Usage(
        input_tokens=raw_usage.get("input_tokens", 0),
        output_tokens=raw_usage.get("output_tokens", 0),
    ) if raw_usage else None
    return ModelResponse(
        content="".join(text_parts), tool_calls=tool_calls, usage=usage
    )


def _parse_titan(body: dict[str, Any]) -> ModelResponse:
    results = body.get("results", [{}])
    text = results[0].get("outputText", "") if results else ""
    input_tokens = body.get("inputTextTokenCount", 0)
    output_tokens = results[0].get("tokenCount", 0) if results else 0
    return ModelResponse(
        content=text.strip(),
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _parse_llama(body: dict[str, Any]) -> ModelResponse:
    text = body.get("generation", "")
    return ModelResponse(
        content=text.strip(),
        usage=Usage(
            input_tokens=body.get("prompt_token_count", 0),
            output_tokens=body.get("generation_token_count", 0),
        ),
    )


def _parse_mistral(body: dict[str, Any]) -> ModelResponse:
    outputs = body.get("outputs", [{}])
    text = outputs[0].get("text", "") if outputs else ""
    return ModelResponse(content=text.strip())


# ---------------------------------------------------------------------------
# Streaming chunk parsers
# ---------------------------------------------------------------------------

def parse_stream_chunk(chunk_bytes: bytes, model_id: str) -> str:
    """Extract the text delta from a single Bedrock streaming event."""
    try:
        chunk = json.loads(chunk_bytes)
    except (json.JSONDecodeError, ValueError):
        return ""

    family = detect_family(model_id)
    if family == "claude":
        if chunk.get("type") == "content_block_delta":
            return chunk.get("delta", {}).get("text", "")
        return ""
    if family == "titan":
        return chunk.get("outputText", "")
    if family == "llama":
        return chunk.get("generation", "")
    if family == "mistral":
        outputs = chunk.get("outputs", [{}])
        return outputs[0].get("text", "") if outputs else ""
    return ""
