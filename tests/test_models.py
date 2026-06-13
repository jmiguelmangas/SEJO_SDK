import asyncio
from types import SimpleNamespace

from SEJO_SDK.messages import ModelResponse, system_message, user_message
from SEJO_SDK.models.model_openai import AsyncOpenAIModel, OpenAIModel
from SEJO_SDK.models.openai_compat import parse_tool_call


class FakeOpenAIClient:
    def __init__(self, chunks=None, message=None):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )
        self.calls = []
        self.chunks = chunks or []
        self.message = message

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.chunks
        message = self.message or SimpleNamespace(content="provider response")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeAsyncOpenAIClient:
    def __init__(self, chunks=None, message=None):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )
        self.calls = []
        self.chunks = chunks or []
        self.message = message

    async def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream_chunks()
        message = self.message or SimpleNamespace(content="async provider response")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    async def _stream_chunks(self):
        for chunk in self.chunks:
            yield chunk


def test_openai_model_sends_chat_completion_request():
    client = FakeOpenAIClient()
    model = OpenAIModel(api_key="test", model_name="test-model", client=client)

    response = model.send_prompt("Hello", temperature=0)

    assert response == "provider response"
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert client.calls[0]["temperature"] == 0


def test_openai_model_sends_native_messages():
    client = FakeOpenAIClient()
    model = OpenAIModel(api_key="test", model_name="test-model", client=client)

    response = model.send_messages(
        [system_message("Be concise."), user_message("Hello")],
        temperature=0,
    )

    assert response == "provider response"
    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]


def test_openai_model_streams_non_empty_chunks():
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="he"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="llo"))]),
    ]
    client = FakeOpenAIClient(chunks=chunks)
    model = OpenAIModel(api_key="test", model_name="test-model", client=client)

    assert list(model.stream_response("Hello")) == ["he", "llo"]


def test_openai_model_returns_tool_calls():
    message = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="add", arguments='{"left": 2}'),
            )
        ],
    )
    client = FakeOpenAIClient(message=message)
    model = OpenAIModel(api_key="test", model_name="test-model", client=client)

    response = model.send_prompt("Hello", tools=[])

    assert isinstance(response, ModelResponse)
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "add"
    assert response.tool_calls[0].arguments == {"left": 2}


def test_openai_tool_call_parser_handles_invalid_json_arguments():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="lookup", arguments="{not-json"),
    )

    parsed = parse_tool_call(tool_call)

    assert parsed.arguments == {"value": "{not-json"}


def test_openai_tool_call_parser_wraps_non_object_arguments():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="choose", arguments='["a", "b"]'),
    )

    parsed = parse_tool_call(tool_call)

    assert parsed.arguments == {"value": ["a", "b"]}


def test_async_openai_model_sends_chat_completion_request():
    async def exercise():
        client = FakeAsyncOpenAIClient()
        model = AsyncOpenAIModel(api_key="test", model_name="test-model", client=client)

        response = await model.send_prompt("Hello", temperature=0)

        assert response == "async provider response"
        assert client.calls[0]["model"] == "test-model"
        assert client.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]
        assert client.calls[0]["temperature"] == 0

    asyncio.run(exercise())


def test_async_openai_model_sends_native_messages():
    async def exercise():
        client = FakeAsyncOpenAIClient()
        model = AsyncOpenAIModel(api_key="test", model_name="test-model", client=client)

        response = await model.send_messages(
            [system_message("Be concise."), user_message("Hello")],
            temperature=0,
        )

        assert response == "async provider response"
        assert client.calls[0]["messages"] == [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]

    asyncio.run(exercise())


def test_async_openai_model_returns_tool_calls():
    async def exercise():
        message = SimpleNamespace(
            content="",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="add", arguments='{"left": 2}'),
                )
            ],
        )
        client = FakeAsyncOpenAIClient(message=message)
        model = AsyncOpenAIModel(api_key="test", model_name="test-model", client=client)

        response = await model.send_prompt("Hello", tools=[])

        assert isinstance(response, ModelResponse)
        assert response.tool_calls[0].arguments == {"left": 2}

    asyncio.run(exercise())


def test_async_openai_model_streams_non_empty_chunks():
    async def exercise():
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="he"))]
            ),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="llo"))]
            ),
        ]
        client = FakeAsyncOpenAIClient(chunks=chunks)
        model = AsyncOpenAIModel(api_key="test", model_name="test-model", client=client)

        streamed = []
        async for chunk in model.stream_response("Hello"):
            streamed.append(chunk)

        assert streamed == ["he", "llo"]

    asyncio.run(exercise())
