from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any


class Model_client(ABC):
    @abstractmethod
    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the model and return the response."""
        pass
    @abstractmethod
    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the model and return the response as a stream."""
        pass
        