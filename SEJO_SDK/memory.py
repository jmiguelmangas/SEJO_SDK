"""Conversation memory implementation."""

from typing import Any, Optional, Union

from SEJO_SDK.messages import (
    Message,
    assistant_message,
    messages_to_prompt,
    tool_message,
    user_message,
)

MessageDict = dict[str, Any]
MessageLike = Union[Message, MessageDict]


class Memory:
    def __init__(self, max_size: int = 20):
        self.history: list[MessageDict] = []
        self.max_size = max_size

    def add_ai_message(self, message: str) -> None:
        self._add(assistant_message(message))

    def add_user_message(self, message: str) -> None:
        self._add(user_message(message))

    def add_message(self, role: str, content: str) -> None:
        self._add(Message(role=role, content=content))

    def add_tool_message(
        self,
        name: str,
        content: str,
        tool_call_id: Optional[str] = None,
    ) -> None:
        self._add(tool_message(content=content, name=name, tool_call_id=tool_call_id))

    def _add(self, message: MessageLike) -> None:
        if isinstance(message, Message):
            message = message.to_dict()
        self.history.append(message)
        if len(self.history) > self.max_size:
            self.history.pop(0)

    def update(self, message: MessageLike) -> None:
        self._add(message)

    def clear(self) -> None:
        self.history.clear()

    def get_context(self) -> str:
        return messages_to_prompt(self.history)

    def get_messages(self) -> list[Message]:
        return [Message.from_dict(message) for message in self.history]

    # Backwards-compatible private aliases.
    _add_ai_message = add_ai_message
    _add_user_message = add_user_message
