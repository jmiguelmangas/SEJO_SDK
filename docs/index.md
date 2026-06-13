# SEJO SDK

SEJO SDK is a lightweight Python SDK for building provider-agnostic AI agents.

It provides:

- sync and async model interfaces
- provider adapters for OpenAI, Anthropic, Gemini and DeepSeek
- conversation memory
- streaming
- typed messages
- local tools and provider-friendly tool schemas
- sync and async tool-calling loops
- no-network unit testing patterns

## Quick Start

Install the core package:

```bash
pip install sejo-sdk
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
print(agent.run("What is the capital of Spain?"))
```

## Learn The SDK

Start here:

- [Getting Started](getting-started.md)
- [Core Concepts](concepts.md)
- [Sync Agents](guides/sync-agents.md)
- [Async Agents](guides/async-agents.md)
- [Native Messages](guides/native-messages.md)
- [Tools](guides/tools.md)

Then use the reference docs:

- [Providers](providers.md)
- [Memory](memory.md)
- [CLI](cli.md)
- [API Reference](api-reference.md)
- [Testing](testing.md)
- [Release Checklist](release-checklist.md)

Project files:

- [Changelog](https://github.com/jmiguelmangas/SEJO_SDK/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/jmiguelmangas/SEJO_SDK/blob/main/CONTRIBUTING.md)
- [Security Policy](https://github.com/jmiguelmangas/SEJO_SDK/blob/main/SECURITY.md)

## Local Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mkdocs serve
```
