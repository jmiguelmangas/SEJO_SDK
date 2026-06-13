"""Typed messages and model response primitives."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class Message:
    """A chat message with optional provider metadata."""

    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list["ToolCall"]] = None

    def to_dict(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            message["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return message

    @classmethod
    def from_dict(cls, message: dict[str, Any]) -> "Message":
        raw_tool_calls = message.get("tool_calls")
        tool_calls = None
        if raw_tool_calls is not None:
            tool_calls = [
                tc if isinstance(tc, ToolCall) else ToolCall.from_dict(tc)
                for tc in raw_tool_calls
            ]
        return cls(
            role=message["role"],
            content=message.get("content", ""),
            name=message.get("name"),
            tool_call_id=message.get("tool_call_id"),
            tool_calls=tool_calls,
        )


MessageLike = Union[Message, dict[str, Any]]


@dataclass
class ToolCall:
    """A provider-requested tool invocation."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        tool_call: dict[str, Any] = {
            "name": self.name,
            "arguments": self.arguments,
        }
        if self.id is not None:
            tool_call["id"] = self.id
        return tool_call

    @classmethod
    def from_dict(cls, tool_call: dict[str, Any]) -> "ToolCall":
        return cls(
            name=tool_call["name"],
            arguments=tool_call.get("arguments", {}),
            id=tool_call.get("id"),
        )


@dataclass
class Usage:
    """Token usage reported by a provider."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelResponse:
    """A model response that may request local tool calls."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Optional[Usage] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
        }


def system_message(content: str) -> Message:
    return Message(role="system", content=content)


def user_message(content: str) -> Message:
    return Message(role="user", content=content)


def assistant_message(content: str) -> Message:
    return Message(role="assistant", content=content)


def assistant_message_with_tools(content: str, tool_calls: list["ToolCall"]) -> Message:
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def tool_message(
    content: str,
    name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Message:
    return Message(
        role="tool",
        content=content,
        name=name,
        tool_call_id=tool_call_id,
    )


def ensure_message(message: MessageLike) -> Message:
    if isinstance(message, Message):
        return message
    return Message.from_dict(message)


def messages_to_dicts(messages: list[MessageLike]) -> list[dict[str, Any]]:
    return [ensure_message(message).to_dict() for message in messages]


def messages_to_prompt(messages: list[MessageLike]) -> str:
    formatted = []
    for message_like in messages:
        message = ensure_message(message_like)
        if message.role == "tool" and message.name:
            formatted.append(f"tool {message.name}: {message.content}")
        else:
            formatted.append(f"{message.role}: {message.content}")
    return "\n".join(formatted)
