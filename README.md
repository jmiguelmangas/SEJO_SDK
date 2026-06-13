# SEJO SDK

[![SEJO SDK CI](https://github.com/jmiguelmangas/SEJO_SDK/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jmiguelmangas/SEJO_SDK/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-0F766E)
![Status](https://img.shields.io/badge/Status-alpha-F59E0B)
![LLM Providers](https://img.shields.io/badge/Providers-OpenAI%20%7C%20Anthropic%20%7C%20Gemini%20%7C%20DeepSeek-111827)

<p align="center">
  <img src="SEJO_SDK/logo/Sejo_logo.png" alt="SEJO SDK Logo" width="260">
</p>

SEJO SDK is a lightweight Python SDK for building provider-agnostic AI agents.
It gives you one small interface for model adapters, conversation memory,
streaming responses and optional tools.

The project is currently alpha, but the core package is designed to be
installable without pulling every provider dependency. Add only the extras you
need.

## Features

- Unified `ModelClient` interface for provider adapters.
- Agent runtime with conversation memory.
- Streaming and non-streaming responses.
- Optional provider extras for OpenAI, Anthropic, Gemini and DeepSeek.
- Optional tools for web search, PostgreSQL and FastAPI WebSocket chat.
- Tests that run without API keys or live provider calls.

## Installation

Install the core package:

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

For local development:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Basic Usage

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

```python
for chunk in agent.stream("Explain machine learning in one paragraph."):
    print(chunk, end="")
```

## Custom Model Adapter

```python
from typing import Iterator

from SEJO_SDK.model import ModelClient


class EchoModel(ModelClient):
    def send_prompt(self, prompt: str, **kwargs) -> str:
        return f"Echo: {prompt}"

    def stream_response(self, prompt: str, **kwargs) -> Iterator[str]:
        yield "Echo: "
        yield prompt
```

## Project Structure

```text
SEJO_SDK/
  agent.py                  Agent runtime.
  memory.py                 Conversation memory.
  model.py                  Provider-agnostic model interface.
  models/                   OpenAI, Anthropic, Gemini and DeepSeek adapters.
  tools/                    Tool primitives and optional integrations.
  utils/                    Utility helpers and database connector.
tests/                      Unit tests without live provider calls.
```

## Quality

```bash
python -m compileall SEJO_SDK tests
ruff check .
pytest --cov=SEJO_SDK --cov-report=term-missing
```

## Roadmap

- Add async model clients.
- Add tool-calling orchestration.
- Add provider-specific integration tests behind opt-in environment variables.
- Add typed message objects for system/developer/tool messages.
- Publish generated API documentation.

## Contact

- GitHub: [jmiguelmangas](https://github.com/jmiguelmangas)
- Email: [jmmangas@gmail.com](mailto:jmmangas@gmail.com)

## License

MIT. See [LICENSE](LICENSE).
