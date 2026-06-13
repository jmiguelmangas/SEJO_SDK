"""Provider-agnostic model client interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class ModelClient(ABC):
    """Base class for model adapters used by agents."""

    @abstractmethod
    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the model and return the response."""

        raise NotImplementedError

    @abstractmethod
    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the model and return the response as a stream."""

        raise NotImplementedError


# Backwards-compatible alias for older imports.
Model_client = ModelClient
