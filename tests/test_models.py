from types import SimpleNamespace

from SEJO_SDK.models.model_openai import OpenAIModel


class FakeOpenAIClient:
    def __init__(self, chunks=None):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )
        self.calls = []
        self.chunks = chunks or []

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.chunks
        message = SimpleNamespace(content="provider response")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def test_openai_model_sends_chat_completion_request():
    client = FakeOpenAIClient()
    model = OpenAIModel(api_key="test", model_name="test-model", client=client)

    response = model.send_prompt("Hello", temperature=0)

    assert response == "provider response"
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert client.calls[0]["temperature"] == 0


def test_openai_model_streams_non_empty_chunks():
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="he"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="llo"))]),
    ]
    client = FakeOpenAIClient(chunks=chunks)
    model = OpenAIModel(api_key="test", model_name="test-model", client=client)

    assert list(model.stream_response("Hello")) == ["he", "llo"]
