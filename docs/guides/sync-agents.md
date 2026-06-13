# Sync Agents

Use sync agents when your application code is synchronous or when the provider
client is synchronous.

## Basic Run

```python
from SEJO_SDK.agent import Agent
from SEJO_SDK.models import OpenAIModel


model = OpenAIModel(
    api_key="your-api-key",
    model_name="your-model-name",
)

agent = Agent(model=model)
response = agent.run("Write a one-sentence project tagline.")

print(response)
```

## Prompt Construction

`Agent.run` appends the user message to memory, then sends the full context to
the model.

If memory contains:

```python
[
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]
```

And the next input is:

```text
What can you do?
```

The prompt sent to the model is:

```text
user: Hello
assistant: Hi
user: What can you do?
Agent:
```

## Streaming

`Agent.stream` yields chunks from the model:

```python
for chunk in agent.stream("Explain vector databases."):
    print(chunk, end="")
```

The agent collects chunks while yielding them. When the stream finishes, the
joined text is stored as an assistant message in memory.

`Agent.stream` uses `model.stream_messages` internally, so the system prompt
and full conversation history are passed natively to the provider. Custom
adapters that only implement `stream_response` automatically fall back to a
prompt-rendered version.

## Custom Sync Model

```python
from SEJO_SDK.model import ModelClient


class StaticModel(ModelClient):
    def send_prompt(self, prompt: str, **kwargs) -> str:
        return "Static response"

    def stream_response(self, prompt: str, **kwargs):
        yield "Static"
        yield " response"
```

To override streaming with native message support, implement `stream_messages`:

```python
class NativeStreamModel(ModelClient):
    def send_prompt(self, prompt: str, **kwargs) -> str:
        return "ok"

    def stream_messages(self, messages, **kwargs):
        for message in messages:
            yield message.content

    def stream_response(self, prompt: str, **kwargs):
        yield prompt
```

## Passing Provider Options

Provider adapters pass extra keyword arguments through to the provider SDK.

```python
response = model.send_prompt(
    "Be concise.",
    temperature=0,
)
```

When called through `Agent.run`, the current implementation does not expose
provider keyword arguments. If your application needs per-call provider options,
call the model directly or add a small wrapper method in your own application
layer.

## Error Handling

Missing provider dependencies raise `ProviderDependencyError` when the adapter
tries to create a real provider client.

```python
from SEJO_SDK.errors import ProviderDependencyError


try:
    model = OpenAIModel(api_key="key", model_name="model")
except ProviderDependencyError as exc:
    print(exc)
```

## Testing Sync Agents

Use a fake model:

```python
from SEJO_SDK.model import ModelClient


class FakeModel(ModelClient):
    def __init__(self, response="ok"):
        self.response = response
        self.prompts = []

    def send_prompt(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        return self.response

    def stream_response(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        yield self.response
```
