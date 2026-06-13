"""Helpers for OpenAI-compatible chat completion responses."""

import json
from typing import Any, Union

from SEJO_SDK.messages import Message, ModelResponse, ToolCall


def messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    return [message.to_dict() for message in messages]


def parse_chat_message(message: Any) -> Union[str, ModelResponse]:
    """Parse an OpenAI-compatible message into text or a typed response."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return getattr(message, "content", None) or ""
    return ModelResponse(
        content=getattr(message, "content", None) or "",
        tool_calls=[parse_tool_call(tool_call) for tool_call in tool_calls],
    )


def parse_tool_call(tool_call: Any) -> ToolCall:
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", "")
    raw_arguments = getattr(function, "arguments", "{}") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = {"value": raw_arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return ToolCall(
        id=getattr(tool_call, "id", None),
        name=name,
        arguments=arguments,
    )
