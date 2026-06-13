"""Base tool primitive."""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)
