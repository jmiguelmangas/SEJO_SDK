"""Agent implementation."""

import json
from collections.abc import Iterable
from typing import Any, Optional, TypeVar, Union

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
from SEJO_SDK.tracing import Tracer

T = TypeVar("T")


class Agent:
    def __init__(
        self,
        model: ModelClient,
        memory: Optional[Memory] = None,
        tools: Optional[Union[dict[str, Tool], Iterable[Tool]]] = None,
        system_prompt: Optional[str] = None,
        tracer: Optional[Tracer] = None,
    ):
        """Initialize the agent with a model, optional memory and optional tools."""
        self.model = model
        self.memory = memory or Memory()
        self.tools = self._normalize_tools(tools)
        self.system_prompt = system_prompt
        self.tracer = tracer

    def run(self, user_input: str) -> str:
        """Run the agent with a prompt and return the response."""
        self.memory.add_user_message(user_input)
        if self.tracer:
            self.tracer.start_turn()
        raw = self.model.send_messages(self._build_messages())
        model_response = self._coerce_model_response(raw)
        content = model_response.content
        self.memory.add_ai_message(content)
        if self.tracer:
            self.tracer.end_turn(
                role="assistant",
                input_text=user_input,
                output_text=content,
                usage=model_response.usage,
            )
        return content

    async def arun(self, user_input: str) -> str:
        """Run the agent asynchronously with a prompt and return the response."""
        if not isinstance(self.model, AsyncModelClient):
            raise TypeError("Agent.arun requires an AsyncModelClient model.")
        self.memory.add_user_message(user_input)
        if self.tracer:
            self.tracer.start_turn()
        raw = await self.model.send_messages(self._build_messages())
        model_response = self._coerce_model_response(raw)
        content = model_response.content
        self.memory.add_ai_message(content)
        if self.tracer:
            self.tracer.end_turn(
                role="assistant",
                input_text=user_input,
                output_text=content,
                usage=model_response.usage,
            )
        return content

    def run_with_tools(self, user_input: str, max_tool_iterations: int = 5) -> str:
        """Run an agent loop that executes provider-requested tool calls."""
        from SEJO_SDK.messages import Usage

        self.memory.add_user_message(user_input)
        if self.tracer:
            self.tracer.start_turn()
        accumulated_usage: Optional[Usage] = None
        tool_names: list[str] = []
        for _ in range(max_tool_iterations):
            response = self.model.send_messages(
                self._build_messages(),
                tools=self.tool_schemas(),
            )
            model_response = self._coerce_model_response(response)
            if model_response.usage:
                if accumulated_usage is None:
                    accumulated_usage = model_response.usage
                else:
                    accumulated_usage = Usage(
                        input_tokens=accumulated_usage.input_tokens
                        + model_response.usage.input_tokens,
                        output_tokens=accumulated_usage.output_tokens
                        + model_response.usage.output_tokens,
                    )
            if not model_response.tool_calls:
                self.memory.add_ai_message(model_response.content)
                if self.tracer:
                    self.tracer.end_turn(
                        role="assistant",
                        input_text=user_input,
                        output_text=model_response.content,
                        tool_calls=tool_names,
                        usage=accumulated_usage,
                    )
                return model_response.content
            tool_names.extend(tc.name for tc in model_response.tool_calls)
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
        from SEJO_SDK.messages import Usage

        if not isinstance(self.model, AsyncModelClient):
            raise TypeError("Agent.arun_with_tools requires an AsyncModelClient model.")
        self.memory.add_user_message(user_input)
        if self.tracer:
            self.tracer.start_turn()
        accumulated_usage: Optional[Usage] = None
        tool_names: list[str] = []
        for _ in range(max_tool_iterations):
            response = await self.model.send_messages(
                self._build_messages(),
                tools=self.tool_schemas(),
            )
            model_response = self._coerce_model_response(response)
            if model_response.usage:
                if accumulated_usage is None:
                    accumulated_usage = model_response.usage
                else:
                    accumulated_usage = Usage(
                        input_tokens=accumulated_usage.input_tokens
                        + model_response.usage.input_tokens,
                        output_tokens=accumulated_usage.output_tokens
                        + model_response.usage.output_tokens,
                    )
            if not model_response.tool_calls:
                self.memory.add_ai_message(model_response.content)
                if self.tracer:
                    self.tracer.end_turn(
                        role="assistant",
                        input_text=user_input,
                        output_text=model_response.content,
                        tool_calls=tool_names,
                        usage=accumulated_usage,
                    )
                return model_response.content
            tool_names.extend(tc.name for tc in model_response.tool_calls)
            self._store_assistant_tool_request(model_response)
            for tool_call in model_response.tool_calls:
                result = await self.arun_tool(tool_call.name, **tool_call.arguments)
                self.memory.add_tool_message(
                    name=tool_call.name,
                    content=self._serialize_tool_result(result),
                    tool_call_id=tool_call.id,
                )
        raise RuntimeError("Maximum tool-calling iterations exceeded.")

    def run_structured(self, user_input: str, schema: "type[T]") -> "T":
        """Run the agent and parse the response into a Pydantic model.

        The schema is appended to the prompt as a JSON instruction.

        Example::

            class FlightInfo(BaseModel):
                flight: str
                destination: str

            info = agent.run_structured("Extract flight BA123 to LHR", FlightInfo)
        """
        from SEJO_SDK.structured import parse_structured, schema_prompt

        full_input = user_input + schema_prompt(schema)
        text = self.run(full_input)
        return parse_structured(text, schema)

    async def arun_structured(self, user_input: str, schema: "type[T]") -> "T":
        """Async version of run_structured."""
        from SEJO_SDK.structured import parse_structured, schema_prompt

        full_input = user_input + schema_prompt(schema)
        text = await self.arun(full_input)
        return parse_structured(text, schema)

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
