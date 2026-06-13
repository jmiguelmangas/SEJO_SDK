# Providers

Provider adapters live in `SEJO_SDK.models`.

The package currently includes adapters for:

- OpenAI
- Anthropic
- Gemini
- DeepSeek

Each provider has sync and async variants.

## Optional Dependencies

The core package does not install provider SDKs by default.

Install only what you need:

```bash
pip install "sejo-sdk[openai]"
pip install "sejo-sdk[anthropic]"
pip install "sejo-sdk[gemini]"
```

DeepSeek uses the OpenAI-compatible client:

```bash
pip install "sejo-sdk[openai]"
```

Install every provider and optional integration:

```bash
pip install "sejo-sdk[all]"
```

## OpenAI

Sync:

```python
from SEJO_SDK.models import OpenAIModel


model = OpenAIModel(
    api_key="your-api-key",
    model_name="your-model-name",
)

response = model.send_prompt("Hello")
```

Async:

```python
from SEJO_SDK.models import AsyncOpenAIModel


model = AsyncOpenAIModel(
    api_key="your-api-key",
    model_name="your-model-name",
)

response = await model.send_prompt("Hello")
```

Native messages:

```python
response = model.send_messages([
    system_message("Be concise."),
    user_message("Hello"),
])
```

Streaming:

```python
for chunk in model.stream_response("Hello"):
    print(chunk, end="")
```

Async streaming:

```python
async for chunk in model.stream_response("Hello"):
    print(chunk, end="")
```

When OpenAI returns `message.tool_calls`, the adapter returns a
`ModelResponse` with parsed `ToolCall` objects. This can be passed through
`Agent.run_with_tools`.

## Anthropic

Sync:

```python
from SEJO_SDK.models import AnthropicModel


model = AnthropicModel(
    api_key="your-api-key",
    model_name="your-model-name",
    max_tokens=1024,
)
```

Async:

```python
from SEJO_SDK.models import AsyncAnthropicModel


model = AsyncAnthropicModel(
    api_key="your-api-key",
    model_name="your-model-name",
    max_tokens=1024,
)
```

The Anthropic adapter passes `max_tokens` by default. You can override it per
model call:

```python
response = model.send_prompt("Hello", max_tokens=256)
```

For `send_messages`, system messages are sent through Anthropic's `system`
argument and the rest are sent as chat messages.

### Tool Calling

The Anthropic adapter supports native tool calling through `Agent.run_with_tools`.
Tool schemas are automatically converted from the OpenAI-compatible format to
Anthropic's `input_schema` shape. When the model responds with `tool_use`
blocks, the adapter returns a `ModelResponse` with parsed `ToolCall` objects,
and tool results are serialised as `tool_result` content blocks on the next
user turn.

```python
from SEJO_SDK.agent import Agent
from SEJO_SDK.tools import Tool


def search(query: str) -> str:
    return f"Results for: {query}"


agent = Agent(
    model=AnthropicModel(api_key="...", model_name="claude-3-5-haiku-20241022"),
    tools=[Tool(name="search", description="Search the web.", func=search)],
)

response = agent.run_with_tools("Search for SEJO SDK")
```

## Gemini

Sync:

```python
from SEJO_SDK.models import GeminiModel


model = GeminiModel(
    api_key="your-api-key",
    model_name="your-model-name",
)
```

Async:

```python
from SEJO_SDK.models import AsyncGeminiModel


model = AsyncGeminiModel(
    api_key="your-api-key",
    model_name="your-model-name",
)
```

The Gemini adapter configures `google.generativeai` with the provided API key
when no client is injected.

`send_messages` renders messages into a structured prompt string for Gemini.

## DeepSeek

DeepSeek uses an OpenAI-compatible API shape.

Sync:

```python
from SEJO_SDK.models import DeepSeekModel


model = DeepSeekModel(
    api_key="your-api-key",
    model_name="your-model-name",
)
```

Async:

```python
from SEJO_SDK.models import AsyncDeepSeekModel


model = AsyncDeepSeekModel(
    api_key="your-api-key",
    model_name="your-model-name",
)
```

Override the base URL if needed:

```python
model = DeepSeekModel(
    api_key="your-api-key",
    model_name="your-model-name",
    base_url="https://api.deepseek.com",
)
```

DeepSeek uses the same OpenAI-compatible tool-call parser as `OpenAIModel`.
It also accepts native `send_messages` calls using the OpenAI-compatible chat
message format.

## Injecting Clients

Every provider adapter accepts a `client` argument. This is the recommended way
to test request construction without network calls.

```python
model = OpenAIModel(
    api_key="test",
    model_name="test-model",
    client=fake_client,
)
```

When a client is injected, the adapter does not import or initialize the real
provider SDK.

## Passing Provider Options

Provider-specific options are passed through with `**kwargs`:

```python
response = model.send_prompt(
    "Write a haiku.",
    temperature=0.2,
)
```

For streaming:

```python
for chunk in model.stream_response("Write slowly.", temperature=0.2):
    print(chunk, end="")
```

## Dependency Errors

If a provider SDK is missing, the adapter raises `ProviderDependencyError` with
an installation hint.

```python
from SEJO_SDK.errors import ProviderDependencyError


try:
    model = OpenAIModel(api_key="key", model_name="model")
except ProviderDependencyError as exc:
    print(exc)
```
