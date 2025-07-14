"""Memory implementation."""

class Memory:
    def __init__(self, max_size: int = 20):
        self.history = []
        self.max_size = max_size

    def _add_ai_message(self, message: str):
        self._add({"role": "assistant", "content": message})

    def _add_user_message(self, message: str):

        self._add({"role": "user", "content": message}) 

    def _add(self, message: str):
        self.history.append(message)
        if len(self.history) > self.max_size:
            self.history.pop(0)
    def update(self, message: str):
        self.history.append(message)
        
    def get_context(self) -> str:
        return "\n".join(f"{message['role']}: {message['content']}" for message in self.history)   