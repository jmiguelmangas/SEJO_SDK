"""Conversation memory implementation."""

Message = dict[str, str]


class Memory:
    def __init__(self, max_size: int = 20):
        self.history: list[Message] = []
        self.max_size = max_size

    def add_ai_message(self, message: str) -> None:
        self._add({"role": "assistant", "content": message})

    def add_user_message(self, message: str) -> None:
        self._add({"role": "user", "content": message})

    def add_message(self, role: str, content: str) -> None:
        self._add({"role": role, "content": content})

    def _add(self, message: Message) -> None:
        self.history.append(message)
        if len(self.history) > self.max_size:
            self.history.pop(0)

    def update(self, message: Message) -> None:
        self._add(message)

    def clear(self) -> None:
        self.history.clear()

    def get_context(self) -> str:
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in self.history
        )

    # Backwards-compatible private aliases.
    _add_ai_message = add_ai_message
    _add_user_message = add_user_message
