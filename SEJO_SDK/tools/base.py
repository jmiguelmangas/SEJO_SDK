"""Base tool primitive."""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: Optional[dict[str, Any]] = None
    afunc: Optional[Callable[..., Any]] = None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        if self.afunc is not None:
            return await self.afunc(*args, **kwargs)
        result = self.run(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
