import asyncio
from types import SimpleNamespace

from SEJO_SDK.messages import (
    ModelResponse,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from SEJO_SDK.models.message_adapters import split_anthropic_messages
from SEJO_SDK.models.model_anthropic import AnthropicModel, AsyncAnthropicModel
from SEJO_SDK.models.model_deepseek import AsyncDeepSeekModel, DeepSeekModel
from SEJO_SDK.models.model_gemini import AsyncGeminiModel, GeminiModel


class FakeMessages:
    def __init__(self, stream):
        self.calls = []
        self._stream = stream

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="anthropic response")]
        )

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self._stream


class FakeAnthropicStream:
    text_stream = ["an", "", "thropic"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessages(FakeAnthropicStream())


class FakeAsyncMessages:
    def __init__(self, stream):
        self.calls = []
        self._stream = stream

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="async anthropic")]
        )

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self._stream


class FakeAsyncAnthropicStream:
    def __init__(self):
        self.text_stream = self._text_stream()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def _text_stream(self):
        yield "async "
        yield ""
        yield "anthropic"


class FakeAsyncAnthropicClient:
    def __init__(self):
        self.messages = FakeAsyncMessages(FakeAsyncAnthropicStream())


class FakeGeminiClient:
    def __init__(self, chunks=None):
        self.calls = []
        self.chunks = chunks or []

    def generate_content(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if kwargs.get("stream"):
            return self.chunks
        return SimpleNamespace(text="gemini response")

    async def generate_content_async(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if kwargs.get("stream"):
            return self._stream_chunks()
        return SimpleNamespace(text="async gemini")

    async def _stream_chunks(self):
        for chunk in self.chunks:
            yield chunk


class FakeChatCompletions:
    def __init__(self, chunks=None):
        self.calls = []
        self.chunks = chunks or []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.chunks
        message = SimpleNamespace(content="deepseek response")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeDeepSeekClient:
    def __init__(self, chunks=None):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(chunks))


class FakeAsyncChatCompletions:
    def __init__(self, chunks=None):
        self.calls = []
        self.chunks = chunks or []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream_chunks()
        message = SimpleNamespace(content="async deepseek")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def _stream_chunks(self):
        for chunk in self.chunks:
            yield chunk


class FakeAsyncDeepSeekClient:
    def __init__(self, chunks=None):
        completions = FakeAsyncChatCompletions(chunks)
        self.chat = SimpleNamespace(completions=completions)


def test_anthropic_model_sends_messages_request():
    client = FakeAnthropicClient()
    model = AnthropicModel(api_key="test", model_name="claude", client=client)

    response = model.send_prompt("Hello", temperature=0)

    assert response == ModelResponse(content="anthropic response")
    assert client.messages.calls[0]["model"] == "claude"
    assert client.messages.calls[0]["max_tokens"] == 1024
    assert client.messages.calls[0]["temperature"] == 0


def test_anthropic_model_sends_native_messages_with_system():
    client = FakeAnthropicClient()
    model = AnthropicModel(api_key="test", model_name="claude", client=client)

    response = model.send_messages(
        [
            system_message("Be concise."),
            user_message("Hello"),
            assistant_message("Hi"),
            tool_message("42", name="lookup"),
        ],
        max_tokens=12,
    )

    assert response == ModelResponse(content="anthropic response")
    assert client.messages.calls[0]["system"] == "Be concise."
    assert client.messages.calls[0]["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "", "content": "42"}],
        },
    ]
    assert client.messages.calls[0]["max_tokens"] == 12


def test_anthropic_model_streams_text():
    client = FakeAnthropicClient()
    model = AnthropicModel(api_key="test", model_name="claude", client=client)

    assert list(model.stream_response("Hello")) == ["an", "thropic"]


def test_async_anthropic_model_sends_messages_request():
    async def exercise():
        client = FakeAsyncAnthropicClient()
        model = AsyncAnthropicModel(api_key="test", model_name="claude", client=client)

        response = await model.send_prompt("Hello", max_tokens=12)

        assert response == ModelResponse(content="async anthropic")
        assert client.messages.calls[0]["model"] == "claude"
        assert client.messages.calls[0]["max_tokens"] == 12

    asyncio.run(exercise())


def test_async_anthropic_model_sends_native_messages_with_system():
    async def exercise():
        client = FakeAsyncAnthropicClient()
        model = AsyncAnthropicModel(api_key="test", model_name="claude", client=client)

        response = await model.send_messages(
            [system_message("Be concise."), user_message("Hello")]
        )

        assert response == ModelResponse(content="async anthropic")
        assert client.messages.calls[0]["system"] == "Be concise."
        assert client.messages.calls[0]["messages"] == [
            {"role": "user", "content": "Hello"}
        ]

    asyncio.run(exercise())


def test_async_anthropic_model_streams_text():
    async def exercise():
        client = FakeAsyncAnthropicClient()
        model = AsyncAnthropicModel(api_key="test", model_name="claude", client=client)

        chunks = []
        async for chunk in model.stream_response("Hello"):
            chunks.append(chunk)

        assert chunks == ["async ", "anthropic"]

    asyncio.run(exercise())


def test_gemini_model_sends_and_streams_content():
    chunks = [
        SimpleNamespace(text="ge"),
        SimpleNamespace(text=""),
        SimpleNamespace(text="mini"),
    ]
    client = FakeGeminiClient(chunks=chunks)
    model = GeminiModel(api_key="test", model_name="gemini", client=client)

    assert model.send_prompt("Hello") == "gemini response"
    assert list(model.stream_response("Hello")) == ["ge", "mini"]


def test_gemini_model_sends_native_messages_as_prompt():
    client = FakeGeminiClient()
    model = GeminiModel(api_key="test", model_name="gemini", client=client)

    assert model.send_messages([system_message("Rules"), user_message("Hello")])
    assert client.calls[0][0] == "system: Rules\nuser: Hello"


def test_async_gemini_model_sends_and_streams_content():
    async def exercise():
        chunks = [
            SimpleNamespace(text="async "),
            SimpleNamespace(text=""),
            SimpleNamespace(text="gemini"),
        ]
        client = FakeGeminiClient(chunks=chunks)
        model = AsyncGeminiModel(api_key="test", model_name="gemini", client=client)

        assert await model.send_prompt("Hello") == "async gemini"

        streamed = []
        async for chunk in model.stream_response("Hello"):
            streamed.append(chunk)

        assert streamed == ["async ", "gemini"]

    asyncio.run(exercise())


def test_async_gemini_model_sends_native_messages_as_prompt():
    async def exercise():
        client = FakeGeminiClient()
        model = AsyncGeminiModel(api_key="test", model_name="gemini", client=client)

        response = await model.send_messages(
            [system_message("Rules"), user_message("Hello")]
        )

        assert response
        assert client.calls[0][0] == "system: Rules\nuser: Hello"

    asyncio.run(exercise())


def test_deepseek_model_sends_and_streams_chat_completion():
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="deep"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="seek"))]),
    ]
    client = FakeDeepSeekClient(chunks=chunks)
    model = DeepSeekModel(api_key="test", model_name="deepseek", client=client)

    assert model.send_prompt("Hello") == "deepseek response"
    assert list(model.stream_response("Hello")) == ["deep", "seek"]


def test_deepseek_model_sends_native_messages():
    client = FakeDeepSeekClient()
    model = DeepSeekModel(api_key="test", model_name="deepseek", client=client)

    assert model.send_messages([system_message("Rules"), user_message("Hello")])
    assert client.chat.completions.calls[0]["messages"] == [
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "Hello"},
    ]


def test_async_deepseek_model_sends_and_streams_chat_completion():
    async def exercise():
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="async "))]
            ),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="deepseek"))]
            ),
        ]
        client = FakeAsyncDeepSeekClient(chunks=chunks)
        model = AsyncDeepSeekModel(api_key="test", model_name="deepseek", client=client)

        assert await model.send_prompt("Hello") == "async deepseek"

        streamed = []
        async for chunk in model.stream_response("Hello"):
            streamed.append(chunk)

        assert streamed == ["async ", "deepseek"]

    asyncio.run(exercise())


def test_split_anthropic_messages_without_system_returns_none():
    system, messages = split_anthropic_messages([user_message("Hello")])

    assert system is None
    assert messages == [{"role": "user", "content": "Hello"}]


def test_anthropic_model_returns_model_response_with_tool_calls():
    from SEJO_SDK.messages import ToolCall
    from SEJO_SDK.models.message_adapters import parse_anthropic_response

    content = [
        SimpleNamespace(type="text", text="Thinking..."),
        SimpleNamespace(
            type="tool_use", id="call_1", name="search", input={"q": "test"}
        ),
    ]
    result = parse_anthropic_response(content)

    assert isinstance(result, ModelResponse)
    assert result.content == "Thinking..."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0] == ToolCall(
        id="call_1", name="search", arguments={"q": "test"}
    )


def test_anthropic_model_converts_tool_schemas():
    from SEJO_SDK.models.message_adapters import tools_to_anthropic

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]
    result = tools_to_anthropic(openai_tools)

    assert result == [
        {
            "name": "search",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]


def test_split_anthropic_messages_tool_use_turn():
    from SEJO_SDK.messages import ToolCall, assistant_message_with_tools, tool_message

    messages = [
        user_message("What is 2+2?"),
        assistant_message_with_tools(
            "Let me calculate.",
            [ToolCall(id="call_1", name="calc", arguments={"expr": "2+2"})],
        ),
        tool_message("4", name="calc", tool_call_id="call_1"),
    ]
    system, provider_messages = split_anthropic_messages(messages)

    assert system is None
    assert provider_messages[0] == {"role": "user", "content": "What is 2+2?"}
    assert provider_messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me calculate."},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "calc",
                "input": {"expr": "2+2"},
            },
        ],
    }
    assert provider_messages[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "4"}],
    }
