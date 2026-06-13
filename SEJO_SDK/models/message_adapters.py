"""Provider-specific message conversion helpers."""

from __future__ import annotations

from typing import Any

from SEJO_SDK.messages import (
    Message,
    ModelResponse,
    ToolCall,
    Usage,
    messages_to_prompt,
)


def split_anthropic_messages(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert SDK messages to Anthropic API format.

    Returns (system_prompt_or_None, provider_messages).
    Handles tool_use assistant turns and tool_result user turns.
    """
    system_parts = []
    provider_messages: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue

        if message.role == "assistant" and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for tc in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id or f"call_{tc.name}",
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            provider_messages.append({"role": "assistant", "content": content})
            continue

        if message.role == "tool":
            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content,
            }
            # Merge consecutive tool results into one user message.
            if (
                provider_messages
                and provider_messages[-1]["role"] == "user"
                and isinstance(provider_messages[-1]["content"], list)
                and provider_messages[-1]["content"]
                and provider_messages[-1]["content"][0].get("type") == "tool_result"
            ):
                provider_messages[-1]["content"].append(tool_result)
            else:
                provider_messages.append({"role": "user", "content": [tool_result]})
            continue

        role = "assistant" if message.role == "assistant" else "user"
        provider_messages.append({"role": role, "content": message.content})

    return "\n".join(system_parts) or None, provider_messages


def parse_anthropic_response(response: Any) -> ModelResponse:
    """Parse an Anthropic API response into a ModelResponse."""
    content = getattr(response, "content", response)
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", None),
                    name=getattr(block, "name", ""),
                    arguments=dict(getattr(block, "input", {}) or {}),
                )
            )

    raw_usage = getattr(response, "usage", None)
    usage = None
    if raw_usage is not None:
        usage = Usage(
            input_tokens=getattr(raw_usage, "input_tokens", 0),
            output_tokens=getattr(raw_usage, "output_tokens", 0),
        )

    return ModelResponse(
        content="".join(text_parts), tool_calls=tool_calls, usage=usage
    )


def tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool schemas to Anthropic's format."""
    result = []
    for tool in tools:
        fn = tool.get("function", tool)
        result.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return result


def messages_to_gemini_prompt(messages: list[Message]) -> str:
    return messages_to_prompt(messages)


def messages_to_gemini_contents(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert SDK messages to Gemini content list format.

    Returns (system_instruction_or_None, gemini_contents).
    Handles assistant tool_use turns and tool result turns.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
            continue

        if msg.role == "assistant":
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "function_call": {
                                "name": tc.name,
                                "args": tc.arguments or {},
                            }
                        }
                    )
            contents.append({"role": "model", "parts": parts})
            continue

        if msg.role == "tool":
            fn_resp: dict[str, Any] = {
                "function_response": {
                    "name": msg.name or "",
                    "response": {"result": msg.content},
                }
            }
            # Merge consecutive tool results into a single user turn.
            if (
                contents
                and contents[-1]["role"] == "user"
                and any(
                    "function_response" in p for p in contents[-1]["parts"]
                )
            ):
                contents[-1]["parts"].append(fn_resp)
            else:
                contents.append({"role": "user", "parts": [fn_resp]})
            continue

        contents.append({"role": "user", "parts": [{"text": msg.content}]})

    return "\n".join(system_parts) or None, contents


def tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool schemas to Gemini function_declarations format."""
    declarations = []
    for tool in tools:
        fn = tool.get("function", tool)
        declarations.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return [{"function_declarations": declarations}]


def parse_gemini_response(response: Any) -> ModelResponse:
    """Parse a Gemini generate_content response into a ModelResponse.

    Works with the real google-generativeai SDK objects and with fakes that
    use SimpleNamespace / plain dicts.
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ModelResponse(content=getattr(response, "text", "") or "")

    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = list(getattr(content, "parts", []) if content else [])

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in parts:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(text)
        fc = getattr(part, "function_call", None)
        if fc:
            tool_calls.append(
                ToolCall(
                    name=getattr(fc, "name", ""),
                    arguments=dict(getattr(fc, "args", {}) or {}),
                )
            )

    return ModelResponse(content="".join(text_parts), tool_calls=tool_calls)
