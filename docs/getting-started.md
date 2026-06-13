# Getting Started

This guide takes you from installation to a working local agent.

## Requirements

SEJO SDK supports Python 3.9 and newer.

The base package has no provider dependency installed by default. This keeps the
core install small and lets you choose only the provider SDKs you need.

## Installation

Install the core package from PyPI:

```bash
pip install sejo-sdk
```

Install provider extras as needed:

```bash
pip install "sejo-sdk[openai]"
pip install "sejo-sdk[anthropic]"
pip install "sejo-sdk[gemini]"
pip install "sejo-sdk[websearch]"
pip install "sejo-sdk[postgres]"
pip install "sejo-sdk[server]"
```

Install every optional dependency:

```bash
pip install "sejo-sdk[all]"
```

## Local Development

Clone the repository and install the development extra:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Run lint:

```bash
python -m ruff check .
```

Run the CLI health check:

```bash
sejo doctor
```

Run the docs site:

```bash
python -m mkdocs serve
```

Run coverage:

```bash
python -m pytest --cov=SEJO_SDK --cov-report=term-missing
```

## First Agent

The simplest possible model adapter can echo the prompt:

```python
from SEJO_SDK.agent import Agent
from SEJO_SDK.model import ModelClient


class EchoModel(ModelClient):
    def send_prompt(self, prompt: str, **kwargs) -> str:
        return f"Echo: {prompt}"

    def stream_response(self, prompt: str, **kwargs):
        yield "Echo: "
        yield prompt


agent = Agent(model=EchoModel())

print(agent.run("Hello SEJO"))
```

This is the same pattern used by the test suite: model adapters are easy to
fake because the runtime depends on a small interface instead of a concrete
provider SDK.

## First Provider Agent

Install OpenAI support:

```bash
pip install "sejo-sdk[openai]"
```

Create an agent:

```python
from SEJO_SDK.agent import Agent
from SEJO_SDK.models import OpenAIModel


model = OpenAIModel(
    api_key="your-api-key",
    model_name="your-model-name",
)

agent = Agent(model=model)
response = agent.run("What is the capital of Spain?")

print(response)
```

## Streaming

Any sync model adapter implements `stream_response`, so `Agent.stream` can yield
chunks:

```python
for chunk in agent.stream("Explain machine learning in one paragraph."):
    print(chunk, end="")
```

The agent stores the joined response in memory after streaming finishes.

## Async Usage

Async adapters implement `AsyncModelClient` and work with `Agent.arun` and
`Agent.astream`:

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

    response = await agent.arun("What is the capital of Spain?")
    print(response)


asyncio.run(main())
```

## What To Read Next

Read [Core Concepts](concepts.md) for the mental model, then choose the guide
that matches your workflow:

- [Sync Agents](guides/sync-agents.md)
- [Async Agents](guides/async-agents.md)
- [Tools](guides/tools.md)
