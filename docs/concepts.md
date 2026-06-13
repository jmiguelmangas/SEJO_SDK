# Core Concepts

SEJO SDK is built around a few small concepts that compose together.

## Model Clients

A model client is the adapter between SEJO SDK and an LLM provider.

Sync adapters implement `ModelClient`:

```python
from SEJO_SDK.model import ModelClient


class MyModel(ModelClient):
    def send_prompt(self, prompt: str, **kwargs) -> str:
        ...

    def stream_response(self, prompt: str, **kwargs):
        ...
```

Async adapters implement `AsyncModelClient`:

```python
from SEJO_SDK.model import AsyncModelClient


class MyAsyncModel(AsyncModelClient):
    async def send_prompt(self, prompt: str, **kwargs) -> str:
        ...

    async def stream_response(self, prompt: str, **kwargs):
        ...
```

The built-in provider adapters live in `SEJO_SDK.models`.

Models can also implement `send_messages` for native chat messages. The base
classes provide a fallback that renders messages into the older prompt format,
so existing custom adapters keep working.

## Agents

`Agent` owns three things:

- a model
- a memory object
- an optional tool registry
- an optional system prompt

On each run, the agent:

1. Adds the user message to memory.
2. Builds native messages from memory.
3. Sends those messages to the model.
4. Adds the model response to memory.
5. Returns the response.

The default prompt format is intentionally simple:

```text
user: Hello
Agent:
```

For providers with native message support, the agent sends typed `Message`
objects instead of only a rendered prompt.

After a response, memory contains both sides of the conversation:

```text
user: Hello
assistant: Hi there
```

## Memory

`Memory` stores messages as dictionaries with `role` and `content` keys.

```python
memory.history == [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]
```

The default memory keeps the latest 20 messages. When the limit is exceeded,
the oldest message is removed.

## Tools

A tool wraps a Python callable:

```python
from SEJO_SDK.tools import Tool


tool = Tool(
    name="add",
    description="Add two integers.",
    func=lambda left, right: left + right,
)
```

Tools can be executed locally:

```python
agent.run_tool("add", left=2, right=3)
```

Tools can also expose JSON-schema-like metadata:

```python
tool.to_schema()
```

This is useful when passing tool definitions to a provider that supports
function calling.

## Tool Calls

Tool-calling models can return a `ModelResponse` with one or more `ToolCall`
objects:

```python
from SEJO_SDK.messages import ModelResponse, ToolCall


response = ModelResponse(
    content="I need a calculation.",
    tool_calls=[
        ToolCall(
            name="add",
            arguments={"left": 2, "right": 3},
        )
    ],
)
```

`Agent.run_with_tools` and `Agent.arun_with_tools` execute those calls, store
tool results in memory, and ask the model again until it returns a final
assistant response.

## Provider Extras

The core package does not import provider SDKs at import time. Provider SDKs are
loaded only when their adapters are instantiated without an injected client.

For example, `OpenAIModel` imports `openai.OpenAI` only when needed. If the
dependency is missing, SEJO SDK raises `ProviderDependencyError` with an install
hint.

## Testing Philosophy

The runtime is built so provider calls can be tested with fakes:

- Inject a fake provider client into the adapter.
- Assert the request shape.
- Return simple objects that mimic provider responses.
- Avoid API keys and network calls in unit tests.

This keeps the fast test suite deterministic.
