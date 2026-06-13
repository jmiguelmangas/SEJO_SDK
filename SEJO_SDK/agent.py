"""Agent implementation."""

from collections.abc import Iterable
from typing import Optional, Union

from SEJO_SDK.memory import Memory
from SEJO_SDK.model import ModelClient
from SEJO_SDK.tools import Tool


class Agent:
    def __init__(
        self,
        model: ModelClient,
        memory: Optional[Memory] = None,
        tools: Optional[Union[dict[str, Tool], Iterable[Tool]]] = None,
    ):
        """Initialize the agent with a model, optional memory and optional tools."""
        self.model = model
        self.memory = memory or Memory()
        self.tools = self._normalize_tools(tools)

    def run(self, user_input: str) -> str:
        """Run the agent with a prompt and return the response."""
        self.memory.add_user_message(user_input)
        prompt = f"{self.memory.get_context()}\nAgent:"
        response = self.model.send_prompt(prompt)
        self.memory.add_ai_message(response)
        return response

    def stream(self, user_input: str):
        """Run the agent and yield streaming response chunks."""
        self.memory.add_user_message(user_input)
        prompt = f"{self.memory.get_context()}\nAgent:"
        chunks = []
        for chunk in self.model.stream_response(prompt):
            chunks.append(chunk)
            yield chunk
        self.memory.add_ai_message("".join(chunks))

    @staticmethod
    def _normalize_tools(
        tools: Optional[Union[dict[str, Tool], Iterable[Tool]]]
    ) -> dict[str, Tool]:
        if tools is None:
            return {}
        if isinstance(tools, dict):
            return tools
        return {tool.name: tool for tool in tools}
