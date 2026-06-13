import asyncio

from SEJO_SDK.agent import Agent
from SEJO_SDK.errors import ToolExecutionError, ToolNotFoundError
from SEJO_SDK.memory import Memory
from SEJO_SDK.messages import ModelResponse, ToolCall
from SEJO_SDK.model import AsyncModelClient, ModelClient
from SEJO_SDK.tools import Tool


class FakeModel(ModelClient):
    def __init__(self, response="ok"):
        self.response = response
        self.prompts = []
        self.messages = []

    def send_prompt(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        return self.response

    def send_messages(self, messages, **kwargs):
        self.messages.append(messages)
        return self.response

    def stream_response(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        yield "o"
        yield "k"


class FakeAsyncModel(AsyncModelClient):
    def __init__(self, response="async ok"):
        self.response = response
        self.prompts = []
        self.messages = []

    async def send_prompt(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        return self.response

    async def send_messages(self, messages, **kwargs):
        self.messages.append(messages)
        return self.response

    async def stream_response(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        yield "a"
        yield "sync"


class FakeToolCallingModel(ModelClient):
    def __init__(self):
        self.messages = []
        self.kwargs = []

    def send_prompt(self, prompt: str, **kwargs):
        return "unused"

    def send_messages(self, messages, **kwargs):
        self.messages.append(messages)
        self.kwargs.append(kwargs)
        if len(self.messages) == 1:
            return ModelResponse(
                content="I need to calculate.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="add",
                        arguments={"left": 2, "right": 3},
                    )
                ],
            )
        return "The answer is 5."

    def stream_response(self, prompt: str, **kwargs):
        yield "unused"


class FakeAsyncToolCallingModel(AsyncModelClient):
    def __init__(self):
        self.messages = []
        self.kwargs = []

    async def send_prompt(self, prompt: str, **kwargs):
        return "unused"

    async def send_messages(self, messages, **kwargs):
        self.messages.append(messages)
        self.kwargs.append(kwargs)
        if len(self.messages) == 1:
            return {
                "content": "I need to calculate.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "add",
                        "arguments": {"left": 2, "right": 3},
                    }
                ],
            }
        return ModelResponse(content="The async answer is 5.")

    async def stream_response(self, prompt: str, **kwargs):
        yield "unused"


class ResponseObject:
    content = "object response"
    tool_calls = []


def test_agent_run_builds_prompt_and_updates_memory():
    model = FakeModel(response="Hola")
    agent = Agent(model=model)

    response = agent.run("Que tal?")

    assert response == "Hola"
    assert [message.to_dict() for message in model.messages[0]] == [
        {"role": "user", "content": "Que tal?"}
    ]
    assert agent.memory.history == [
        {"role": "user", "content": "Que tal?"},
        {"role": "assistant", "content": "Hola"},
    ]


def test_agent_run_includes_system_prompt_in_native_messages():
    model = FakeModel(response="Hola")
    agent = Agent(model=model, system_prompt="Be concise.")

    agent.run("Que tal?")

    assert [message.to_dict() for message in model.messages[0]] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Que tal?"},
    ]


def test_model_send_messages_falls_back_to_prompt_for_legacy_adapters():
    class LegacyModel(ModelClient):
        def __init__(self):
            self.prompts = []

        def send_prompt(self, prompt: str, **kwargs):
            self.prompts.append(prompt)
            return "ok"

        def stream_response(self, prompt: str, **kwargs):
            yield "ok"

    model = LegacyModel()
    agent = Agent(model=model, system_prompt="Be concise.")

    assert agent.run("Hi") == "ok"
    assert model.prompts == ["system: Be concise.\nuser: Hi\nAgent:"]


def test_agent_stream_persists_joined_response():
    model = FakeModel()
    agent = Agent(model=model)

    chunks = list(agent.stream("Ping"))

    assert chunks == ["o", "k"]
    assert agent.memory.history[-1] == {"role": "assistant", "content": "ok"}


def test_agent_arun_builds_prompt_and_updates_memory():
    async def exercise():
        model = FakeAsyncModel(response="Hola async")
        agent = Agent(model=model)

        response = await agent.arun("Que tal?")

        assert response == "Hola async"
        assert [message.to_dict() for message in model.messages[0]] == [
            {"role": "user", "content": "Que tal?"}
        ]
        assert agent.memory.history == [
            {"role": "user", "content": "Que tal?"},
            {"role": "assistant", "content": "Hola async"},
        ]

    asyncio.run(exercise())


def test_agent_arun_requires_async_model():
    async def exercise():
        agent = Agent(model=FakeModel())

        try:
            await agent.arun("Que tal?")
        except TypeError as exc:
            assert "AsyncModelClient" in str(exc)
        else:
            raise AssertionError("Expected TypeError")

    asyncio.run(exercise())


def test_agent_astream_persists_joined_response():
    async def exercise():
        model = FakeAsyncModel()
        agent = Agent(model=model)

        chunks = []
        async for chunk in agent.astream("Ping"):
            chunks.append(chunk)

        assert chunks == ["a", "sync"]
        assert agent.memory.history[-1] == {"role": "assistant", "content": "async"}

    asyncio.run(exercise())


def test_agent_astream_requires_async_model():
    async def exercise():
        agent = Agent(model=FakeModel())

        try:
            async for _ in agent.astream("Ping"):
                pass
        except TypeError as exc:
            assert "AsyncModelClient" in str(exc)
        else:
            raise AssertionError("Expected TypeError")

    asyncio.run(exercise())


def test_agent_exposes_and_runs_tools():
    tool = Tool(
        name="double",
        description="Double a number",
        func=lambda value: value * 2,
        parameters={
            "type": "object",
            "properties": {"value": {"type": "number"}},
        },
    )
    agent = Agent(model=FakeModel(), tools=[tool])

    assert agent.run_tool("double", value=3) == 6
    assert agent.tool_schemas() == [tool.to_schema()]


def test_agent_arun_tool_supports_async_tools():
    async def double(value):
        return value * 2

    async def exercise():
        agent = Agent(
            model=FakeAsyncModel(),
            tools=[Tool(name="double", description="Double", func=double)],
        )

        assert await agent.arun_tool("double", value=3) == 6

    asyncio.run(exercise())


def test_agent_run_tool_unknown_tool_raises_sdk_error():
    agent = Agent(model=FakeModel())

    try:
        agent.run_tool("missing")
    except ToolNotFoundError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("Expected ToolNotFoundError")


def test_agent_run_tool_failure_raises_sdk_error():
    def fail():
        raise ValueError("boom")

    agent = Agent(
        model=FakeModel(),
        tools=[Tool(name="fail", description="Fail", func=fail)],
    )

    try:
        agent.run_tool("fail")
    except ToolExecutionError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("Expected ToolExecutionError")


def test_agent_arun_tool_failure_raises_sdk_error():
    async def fail():
        raise ValueError("async boom")

    async def exercise():
        agent = Agent(
            model=FakeAsyncModel(),
            tools=[Tool(name="fail", description="Fail", func=fail)],
        )

        try:
            await agent.arun_tool("fail")
        except ToolExecutionError as exc:
            assert "async boom" in str(exc)
        else:
            raise AssertionError("Expected ToolExecutionError")

    asyncio.run(exercise())


def test_agent_run_with_tools_executes_tool_loop():
    model = FakeToolCallingModel()
    agent = Agent(
        model=model,
        tools=[
            Tool(
                name="add",
                description="Add numbers",
                func=lambda left, right: left + right,
                parameters={
                    "type": "object",
                    "properties": {
                        "left": {"type": "integer"},
                        "right": {"type": "integer"},
                    },
                    "required": ["left", "right"],
                },
            )
        ],
    )

    response = agent.run_with_tools("What is 2 + 3?")

    assert response == "The answer is 5."
    assert model.kwargs[0]["tools"] == agent.tool_schemas()
    assert model.messages[1][-1].to_dict() == {
        "role": "tool",
        "content": "5",
        "name": "add",
        "tool_call_id": "call_1",
    }
    assert agent.memory.history[-1] == {
        "role": "assistant",
        "content": "The answer is 5.",
    }


def test_agent_arun_with_tools_executes_tool_loop():
    async def add(left, right):
        return left + right

    async def exercise():
        model = FakeAsyncToolCallingModel()
        agent = Agent(
            model=model,
            tools=[Tool(name="add", description="Add numbers", func=add)],
        )

        response = await agent.arun_with_tools("What is 2 + 3?")

        assert response == "The async answer is 5."
        assert model.kwargs[0]["tools"] == agent.tool_schemas()
        assert model.messages[1][-1].to_dict() == {
            "role": "tool",
            "content": "5",
            "name": "add",
            "tool_call_id": "call_1",
        }

    asyncio.run(exercise())


def test_agent_arun_with_tools_requires_async_model():
    async def exercise():
        agent = Agent(model=FakeModel())

        try:
            await agent.arun_with_tools("What is 2 + 3?")
        except TypeError as exc:
            assert "AsyncModelClient" in str(exc)
        else:
            raise AssertionError("Expected TypeError")

    asyncio.run(exercise())


def test_agent_run_with_tools_stops_after_max_iterations():
    class LoopingModel(FakeToolCallingModel):
        def send_prompt(self, prompt: str, **kwargs):
            return ModelResponse(
                tool_calls=[ToolCall(name="add", arguments={"left": 1, "right": 1})]
            )

    agent = Agent(
        model=LoopingModel(),
        tools=[
            Tool(
                name="add",
                description="Add",
                func=lambda left, right: left + right,
            )
        ],
    )

    try:
        agent.run_with_tools("loop", max_tool_iterations=1)
    except RuntimeError as exc:
        assert "Maximum tool-calling iterations" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_agent_coerces_object_responses_and_serializes_unserializable_results():
    response = Agent._coerce_model_response(ResponseObject())

    assert response.content == "object response"
    assert Agent._serialize_tool_result({"ok": True}) == '{"ok": true}'
    assert Agent._serialize_tool_result({"bad": object()}).startswith("{'bad':")


def test_agent_accepts_tools_dictionary():
    tool = Tool(name="double", description="Double", func=lambda value: value * 2)
    agent = Agent(model=FakeModel(), tools={"double": tool})

    assert agent.run_tool("double", value=4) == 8


def test_memory_respects_max_size():
    memory = Memory(max_size=2)

    memory.add_user_message("one")
    memory.add_ai_message("two")
    memory.add_user_message("three")

    assert memory.history == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


def test_memory_helpers_update_clear_and_context():
    memory = Memory()

    memory.add_message("system", "Be terse")
    memory.update({"role": "user", "content": "Hi"})

    assert memory.get_context() == "system: Be terse\nuser: Hi"

    memory.clear()

    assert memory.history == []


def test_memory_accepts_typed_tool_messages():
    memory = Memory()

    memory.add_tool_message("lookup", "result", tool_call_id="call_1")

    assert memory.history == [
        {
            "role": "tool",
            "content": "result",
            "name": "lookup",
            "tool_call_id": "call_1",
        }
    ]


def test_agent_stream_uses_native_messages():
    """Agent.stream should call stream_messages, not stream_response."""

    class TrackingModel(ModelClient):
        def __init__(self):
            self.stream_messages_calls = []
            self.stream_response_calls = []

        def send_prompt(self, prompt, **kwargs):
            return "ok"

        def send_messages(self, messages, **kwargs):
            return "ok"

        def stream_messages(self, messages, **kwargs):
            self.stream_messages_calls.append(messages)
            yield "na"
            yield "tive"

        def stream_response(self, prompt, **kwargs):
            self.stream_response_calls.append(prompt)
            yield "fallback"

    model = TrackingModel()
    agent = Agent(model=model, system_prompt="Be brief.")

    chunks = list(agent.stream("Hello"))

    assert chunks == ["na", "tive"]
    assert len(model.stream_messages_calls) == 1
    assert model.stream_messages_calls[0][0].role == "system"
    assert len(model.stream_response_calls) == 0


def test_agent_astream_uses_native_messages():
    """Agent.astream should call stream_messages, not stream_response."""

    async def exercise():
        class AsyncTrackingModel(AsyncModelClient):
            def __init__(self):
                self.stream_messages_calls = []

            async def send_prompt(self, prompt, **kwargs):
                return "ok"

            async def send_messages(self, messages, **kwargs):
                return "ok"

            async def stream_messages(self, messages, **kwargs):
                self.stream_messages_calls.append(messages)
                yield "na"
                yield "tive"

            async def stream_response(self, prompt, **kwargs):
                yield "fallback"

        model = AsyncTrackingModel()
        agent = Agent(model=model, system_prompt="Be brief.")

        chunks = []
        async for chunk in agent.astream("Hello"):
            chunks.append(chunk)

        assert chunks == ["na", "tive"]
        assert len(model.stream_messages_calls) == 1
        assert model.stream_messages_calls[0][0].role == "system"

    asyncio.run(exercise())


def test_model_client_stream_messages_fallback():
    """stream_messages fallback renders messages to prompt for stream_response."""
    from SEJO_SDK.messages import system_message, user_message

    class PromptOnlyModel(ModelClient):
        def __init__(self):
            self.prompts = []

        def send_prompt(self, prompt, **kwargs):
            return "ok"

        def stream_response(self, prompt, **kwargs):
            self.prompts.append(prompt)
            yield "chunk"

    model = PromptOnlyModel()
    chunks = list(model.stream_messages([system_message("Rules"), user_message("Hi")]))

    assert chunks == ["chunk"]
    assert "system: Rules" in model.prompts[0]
    assert "user: Hi" in model.prompts[0]
