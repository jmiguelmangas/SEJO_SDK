# Examples

The repository includes runnable examples in the `examples/` directory.

Run them from the repository root.

## Sync Agent

File:

```text
examples/sync_agent.py
```

Run:

```bash
python examples/sync_agent.py
```

What it demonstrates:

- implementing `ModelClient`
- creating an `Agent`
- calling `Agent.run`

## Async Agent

File:

```text
examples/async_agent.py
```

Run:

```bash
python examples/async_agent.py
```

What it demonstrates:

- implementing `AsyncModelClient`
- calling `Agent.arun`
- streaming with `Agent.astream`

## Tool Schema

File:

```text
examples/tool_schema.py
```

Run:

```bash
python examples/tool_schema.py
```

What it demonstrates:

- creating a `Tool`
- defining JSON-schema-like parameters
- generating a provider-friendly schema
- running a registered tool locally

## Tool-Calling Agent

File:

```text
examples/tool_calling_agent.py
```

Run:

```bash
python examples/tool_calling_agent.py
```

What it demonstrates:

- returning `ModelResponse` from a model
- requesting a `ToolCall`
- running `Agent.run_with_tools`
- feeding tool results back into the next model call

## Native Messages

File:

```text
examples/native_messages.py
```

Run:

```bash
python examples/native_messages.py
```

What it demonstrates:

- `Agent(system_prompt=...)`
- model-level `send_messages`
- fallback-compatible custom model adapters

## Provider Examples

Provider examples require API keys and optional dependencies.

OpenAI:

```bash
pip install "sejo-sdk[openai]"
```

```python
from SEJO_SDK.agent import Agent
from SEJO_SDK.models import OpenAIModel


model = OpenAIModel(
    api_key="your-api-key",
    model_name="your-model-name",
)

agent = Agent(model=model)
print(agent.run("Hello"))
```

Async OpenAI:

```python
import asyncio

from SEJO_SDK.agent import Agent
from SEJO_SDK.models import AsyncOpenAIModel


async def main():
    model = AsyncOpenAIModel(
        api_key="your-api-key",
        model_name="your-model-name",
    )
    agent = Agent(model=model)
    print(await agent.arun("Hello"))


asyncio.run(main())
```

## Adding More Examples

Good examples should:

- run without external services when possible
- use fake or echo models for basic behavior
- keep provider examples clearly marked as requiring credentials
- avoid hidden environment assumptions
- show one concept at a time
