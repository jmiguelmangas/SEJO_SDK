"""Provider-specific message conversion helpers."""

from typing import Any, Optional

from SEJO_SDK.messages import Message, ModelResponse, ToolCall, messages_to_prompt


def split_anthropic_messages(
    messages: list[Message],
) -> tuple[Optional[str], list[dict[str, Any]]]:
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


def parse_anthropic_response(content: Any) -> ModelResponse:
    """Parse an Anthropic API response into a ModelResponse."""
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

    return ModelResponse(content="".join(text_parts), tool_calls=tool_calls)


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
