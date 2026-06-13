"""Agent implementation."""

import json
from collections.abc import Iterable
from typing import Any, Optional, Union

from SEJO_SDK.errors import ToolExecutionError, ToolNotFoundError
from SEJO_SDK.memory import Memory
from SEJO_SDK.messages import (
    Message,
    ModelResponse,
    ToolCall,
    assistant_message_with_tools,
    system_message,
)
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.tools import Tool


class Agent:
    def __init__(
        self,
        model: ModelClient,
        memory: Optional[Memory] = None,
        tools: Optional[Union[dict[str, Tool], Iterable[Tool]]] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the agent with a model, optional memory and optional tools."""
        self.model = model
        self.memory = memory or Memory()
        self.tools = self._normalize_tools(tools)
        self.system_prompt = system_prompt

    def run(self, user_input: str) -> str:
        """Run the agent with a prompt and return the response."""
        self.memory.add_user_message(user_input)
        response = self.model.send_messages(self._build_messages())
        response = self._coerce_model_response(response).content
        self.memory.add_ai_message(response)
        return response

    async def arun(self, user_input: str) -> str:
        """Run the agent asynchronously with a prompt and return the response."""
        if not isinstance(self.model, AsyncModelClient):
            raise TypeError("Agent.arun requires an AsyncModelClient model.")
        self.memory.add_user_message(user_input)
        response = await self.model.send_messages(self._build_messages())
        response = self._coerce_model_response(response).content
        self.memory.add_ai_message(response)
        return response

    def run_with_tools(self, user_input: str, max_tool_iterations: int = 5) -> str:
        """Run an agent loop that executes provider-requested tool calls."""
        self.memory.add_user_message(user_input)
        for _ in range(max_tool_iterations):
            response = self.model.send_messages(
                self._build_messages(),
                tools=self.tool_schemas(),
            )
            model_response = self._coerce_model_response(response)
            if not model_response.tool_calls:
                self.memory.add_ai_message(model_response.content)
                return model_response.content
            self._store_assistant_tool_request(model_response)
            for tool_call in model_response.tool_calls:
                result = self.run_tool(tool_call.name, **tool_call.arguments)
                self.memory.add_tool_message(
                    name=tool_call.name,
                    content=self._serialize_tool_result(result),
                    tool_call_id=tool_call.id,
                )
        raise RuntimeError("Maximum tool-calling iterations exceeded.")

    async def arun_with_tools(
        self,
        user_input: str,
        max_tool_iterations: int = 5,
    ) -> str:
        """Run an async agent loop that executes provider-requested tool calls."""
        if not isinstance(self.model, AsyncModelClient):
            raise TypeError("Agent.arun_with_tools requires an AsyncModelClient model.")
        self.memory.add_user_message(user_input)
        for _ in range(max_tool_iterations):
            response = await self.model.send_messages(
                self._build_messages(),
                tools=self.tool_schemas(),
            )
            model_response = self._coerce_model_response(response)
            if not model_response.tool_calls:
                self.memory.add_ai_message(model_response.content)
                return model_response.content
            self._store_assistant_tool_request(model_response)
            for tool_call in model_response.tool_calls:
                result = await self.arun_tool(tool_call.name, **tool_call.arguments)
                self.memory.add_tool_message(
                    name=tool_call.name,
                    content=self._serialize_tool_result(result),
                    tool_call_id=tool_call.id,
                )
        raise RuntimeError("Maximum tool-calling iterations exceeded.")

    def stream(self, user_input: str):
        """Run the agent and yield streaming response chunks."""
        self.memory.add_user_message(user_input)
        chunks = []
        for chunk in self.model.stream_messages(self._build_messages()):
            chunks.append(chunk)
            yield chunk
        self.memory.add_ai_message("".join(chunks))

    async def astream(self, user_input: str):
        """Run the agent and yield async streaming response chunks."""
        if not isinstance(self.model, AsyncModelClient):
            raise TypeError("Agent.astream requires an AsyncModelClient model.")
        self.memory.add_user_message(user_input)
        chunks = []
        async for chunk in self.model.stream_messages(self._build_messages()):
            chunks.append(chunk)
            yield chunk
        self.memory.add_ai_message("".join(chunks))

    def run_tool(self, name: str, **arguments):
        """Run a registered tool by name."""
        tool = self._get_tool(name)
        try:
            return tool.run(**arguments)
        except Exception as exc:
            raise ToolExecutionError(f"Tool `{name}` failed: {exc}") from exc

    async def arun_tool(self, name: str, **arguments):
        """Run a registered tool by name and await async tool functions."""
        tool = self._get_tool(name)
        try:
            return await tool.arun(**arguments)
        except Exception as exc:
            raise ToolExecutionError(f"Tool `{name}` failed: {exc}") from exc

    def tool_schemas(self) -> list[dict]:
        """Return provider-friendly tool schemas."""
        return [tool.to_schema() for tool in self.tools.values()]

    def _build_messages(self) -> list[Message]:
        messages = self.memory.get_messages()
        if self.system_prompt:
            return [system_message(self.system_prompt), *messages]
        return messages

    def _get_tool(self, name: str) -> Tool:
        try:
            return self.tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool `{name}` is not registered.") from exc

    def _store_assistant_tool_request(self, response: ModelResponse) -> None:
        msg = assistant_message_with_tools(
            response.content or "",
            response.tool_calls,
        )
        self.memory._add(msg)

    @staticmethod
    def _coerce_model_response(response: Any) -> ModelResponse:
        if isinstance(response, ModelResponse):
            return response
        if isinstance(response, str):
            return ModelResponse(content=response)
        if isinstance(response, dict):
            return ModelResponse(
                content=response.get("content", ""),
                tool_calls=[
                    tool_call
                    if isinstance(tool_call, ToolCall)
                    else ToolCall.from_dict(tool_call)
                    for tool_call in response.get("tool_calls", [])
                ],
            )
        content = getattr(response, "content", None)
        tool_calls = getattr(response, "tool_calls", None)
        if content is not None or tool_calls is not None:
            return ModelResponse(
                content=content or "",
                tool_calls=list(tool_calls or []),
            )
        return ModelResponse(content=str(response))

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except TypeError:
            return str(result)

    @staticmethod
    def _normalize_tools(
        tools: Optional[Union[dict[str, Tool], Iterable[Tool]]]
    ) -> dict[str, Tool]:
        if tools is None:
            return {}
        if isinstance(tools, dict):
            return tools
        return {tool.name: tool for tool in tools}
