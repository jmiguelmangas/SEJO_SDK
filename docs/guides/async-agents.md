# Async Agents

Use async agents when your application already runs inside an event loop or
when your provider SDK supports async clients.

## Basic Async Run

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

    response = await agent.arun("Write a one-sentence project tagline.")
    print(response)


asyncio.run(main())
```

## Async Streaming

```python
async for chunk in agent.astream("Explain async iterators."):
    print(chunk, end="")
```

As with sync streaming, the agent stores the joined response in memory after the
stream completes.

## Custom Async Model

```python
from SEJO_SDK.model import AsyncModelClient


class AsyncEchoModel(AsyncModelClient):
    async def send_prompt(self, prompt: str, **kwargs) -> str:
        return f"Echo: {prompt}"

    async def stream_response(self, prompt: str, **kwargs):
        yield "Echo: "
        yield prompt
```

## Runtime Type Checks

`Agent.arun` and `Agent.astream` require the model to implement
`AsyncModelClient`.

If a sync model is passed to async agent methods, SEJO SDK raises `TypeError`:

```python
await agent.arun("Hello")
```

The error message explains that an async model is required.

## Async Provider Adapters

The following async adapters are available:

- `AsyncOpenAIModel`
- `AsyncAnthropicModel`
- `AsyncGeminiModel`
- `AsyncDeepSeekModel`

They are exported from `SEJO_SDK.models`.

## Testing Async Agents

You can test async agents without `pytest-asyncio` by wrapping the async flow in
`asyncio.run`:

```python
import asyncio


def test_async_agent():
    async def exercise():
        response = await agent.arun("Hello")
        assert response == "ok"

    asyncio.run(exercise())
```

This keeps the test dependencies small.
