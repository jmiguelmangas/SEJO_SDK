# SEJO SDK

[![SEJO SDK CI](https://github.com/jmiguelmangas/SEJO_SDK/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jmiguelmangas/SEJO_SDK/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-0F766E)
![Status](https://img.shields.io/badge/Status-alpha-F59E0B)

<p align="center">
  <img src="SEJO_SDK/logo/Sejo_logo.png" alt="SEJO SDK Logo" width="260">
</p>

A lightweight Python SDK for building provider-agnostic AI agents.
One unified interface for model calls, tool use, streaming, sessions and evals —
no hard runtime dependencies, add only the provider extras you need.

## Provider support

| Provider | Messages | Tool calling | Streaming |
|---|---|---|---|
| OpenAI / DeepSeek | ✅ | ✅ native | ✅ |
| Anthropic | ✅ | ✅ native | ✅ |
| Gemini | ✅ | ✅ native | ✅ |
| AWS Bedrock — Claude | ✅ | ✅ native | ✅ |
| AWS Bedrock — Titan / Llama / Mistral | ✅ | — | ✅ |

## Features

| Area | What you get |
|---|---|
| **Agents** | `Agent.run()` / `arun()` / `stream()` / `astream()` with conversation memory and system prompts |
| **Tools** | Native tool-calling loops (`run_with_tools` / `arun_with_tools`) for all supported providers |
| **Structured output** | `agent.run_structured(prompt, Schema)` — Pydantic-validated responses |
| **Resilience** | `RetryModel` (exponential back-off) · `FallbackModel` (provider chain) |
| **Prompts** | `PromptTemplate` — named variable substitution |
| **Tracing** | `Tracer` — per-turn token counts and cost estimates |
| **Sessions** | `InMemorySessionStore` · `PostgresSessionStore` · `RedisSessionStore` |
| **Server** | `create_agent_app()` — FastAPI with REST + WebSocket endpoints |
| **Multi-agent** | `agent.as_tool()` wraps any agent as a tool · `agent.delegate()` |
| **Evals** | `EvalSuite` — run a test dataset, score with `exact_match` / `contains` / `llm_judge` |

## Installation

```bash
pip install sejo-sdk
```

Add only the extras you need:

```bash
pip install "sejo-sdk[openai]"
pip install "sejo-sdk[anthropic]"
pip install "sejo-sdk[gemini]"
pip install "sejo-sdk[bedrock]"
pip install "sejo-sdk[postgres]"
pip install "sejo-sdk[redis]"
pip install "sejo-sdk[server]"
pip install "sejo-sdk[structured]"
pip install "sejo-sdk[all]"
```

Local development:

```bash
pip install -e ".[dev]"
pytest
ruff check SEJO_SDK/ tests/
```

## Quick start

```python
from SEJO_SDK.agent import Agent
from SEJO_SDK.models import OpenAIModel

agent = Agent(
    model=OpenAIModel(api_key="...", model_name="gpt-4o-mini"),
    system_prompt="You are a helpful assistant.",
)

print(agent.run("What is the capital of Spain?"))
```

## Core concepts

### Streaming

```python
for chunk in agent.stream("Explain async Python in one paragraph."):
    print(chunk, end="", flush=True)
```

### Async

```python
import asyncio
from SEJO_SDK.models import AsyncOpenAIModel

agent = Agent(model=AsyncOpenAIModel(api_key="...", model_name="gpt-4o"))

async def main():
    print(await agent.arun("Hello"))
    async for chunk in agent.astream("Tell me a story."):
        print(chunk, end="")

asyncio.run(main())
```

### Tool calling

```python
from SEJO_SDK.tools import Tool

def get_flight(flight_id: str) -> str:
    return f"BA{flight_id}: London → Madrid, on time"

agent = Agent(
    model=OpenAIModel(api_key="...", model_name="gpt-4o"),
    tools=[Tool(
        name="get_flight",
        description="Get live flight status.",
        func=get_flight,
        parameters={
            "type": "object",
            "properties": {"flight_id": {"type": "string"}},
            "required": ["flight_id"],
        },
    )],
)

print(agent.run_with_tools("What is the status of flight BA123?"))
```

Works identically with `AnthropicModel`, `GeminiModel` and
`BedrockModel("anthropic.claude-3-5-haiku-*")` — schemas are converted
to each provider's native format automatically.

### Structured output

```python
from pydantic import BaseModel
from SEJO_SDK.models import AnthropicModel

class FlightInfo(BaseModel):
    flight: str
    destination: str
    status: str

agent = Agent(model=AnthropicModel(api_key="...", model_name="claude-3-5-haiku-20241022"))
info = agent.run_structured("Extract: BA123 to LHR, delayed 20 min", FlightInfo)
# → FlightInfo(flight='BA123', destination='LHR', status='delayed 20 min')
```

### Resilience

```python
from SEJO_SDK import RetryModel, FallbackModel
from SEJO_SDK.models import OpenAIModel, AnthropicModel

# Retry with exponential back-off on transient errors
model = RetryModel(OpenAIModel(api_key="...", model_name="gpt-4o"), max_retries=3)

# Try providers in order; use the first that succeeds
model = FallbackModel(
    OpenAIModel(api_key="...", model_name="gpt-4o"),
    AnthropicModel(api_key="...", model_name="claude-3-5-haiku-20241022"),
)

agent = Agent(model=model)
```

### Prompt templates

```python
from SEJO_SDK import PromptTemplate

t = PromptTemplate("You are a {role} assistant for {airline}.")
system = t.render(role="crew scheduling", airline="British Airways")
agent = Agent(model=model, system_prompt=system)
```

### Tracing & cost

```python
from SEJO_SDK import Tracer

tracer = Tracer(model_name="gpt-4o")
agent = Agent(model=model, tracer=tracer)

agent.run("Hello")
print(tracer.summary())
# TraceSummary(turns=1, total_tokens=42, estimated_cost_usd=0.000126)
```

### Sessions

```python
# In-memory (development)
from SEJO_SDK import InMemorySessionStore

store = InMemorySessionStore()
session = store.get_or_create("user-42")
agent = Agent(model=model, memory=session.memory)
agent.run("Remember me?")

# PostgreSQL (production)
from SEJO_SDK import PostgresSessionStore

store = PostgresSessionStore("postgresql://user:pass@localhost/db")
session = store.get_or_create("user-42")
agent = Agent(model=model, memory=session.memory)
agent.run("Hello again")
store.save(session)

# Redis (production, with TTL)
from SEJO_SDK import RedisSessionStore

store = RedisSessionStore("redis://localhost:6379", ttl=86400)
session = store.get_or_create("user-42")
agent = Agent(model=model, memory=session.memory)
agent.run("Hello")
store.save(session)
```

### Multi-agent

```python
from SEJO_SDK.agent import Agent
from SEJO_SDK.models import OpenAIModel

researcher = Agent(
    model=OpenAIModel(api_key="...", model_name="gpt-4o"),
    system_prompt="You are a research specialist.",
)
writer = Agent(
    model=OpenAIModel(api_key="...", model_name="gpt-4o"),
    system_prompt="You are a concise writer.",
)

orchestrator = Agent(
    model=OpenAIModel(api_key="...", model_name="gpt-4o"),
    tools=[
        researcher.as_tool("research", "Research a topic in depth."),
        writer.as_tool("write", "Write a short report from research notes."),
    ],
)

result = orchestrator.run_with_tools("Write a brief report on AI in aviation.")
```

### Evals

```python
from SEJO_SDK import EvalSuite, EvalCase, contains

suite = EvalSuite(
    name="BA assistant smoke test",
    scorer=contains,
    cases=[
        EvalCase("IATA code for Heathrow?", expected="LHR"),
        EvalCase("Hub airport of British Airways?", expected="Heathrow"),
        EvalCase("What terminal does BA use at LHR?", expected="5"),
    ],
)

report = suite.run(agent)
report.print_report()
# ============================================================
#   Eval: BA assistant smoke test
#   3/3 passed  (100%)  avg score 1.00
# ============================================================

# LLM-as-judge for open-ended questions
from SEJO_SDK import llm_judge

judge_suite = EvalSuite(
    name="quality check",
    scorer=llm_judge(judge_model),
    cases=[EvalCase("Explain crew rest rules.", expected="adequate rest")],
)
```

### Agent as a service

```python
from SEJO_SDK.server import create_agent_app
import uvicorn

app = create_agent_app(
    model=OpenAIModel(api_key="...", model_name="gpt-4o"),
    system_prompt="You are a crew scheduling assistant.",
    title="BA Crew Assistant",
)

uvicorn.run(app, host="0.0.0.0", port=8000)
# POST /chat         — single-turn
# POST /chat/{id}    — multi-turn session
# WS   /ws/{id}      — streaming WebSocket
# GET  /sessions     — list sessions
# GET  /trace        — token usage
```

### AWS Bedrock

```python
from SEJO_SDK.models import BedrockModel

# Claude on Bedrock — full tool calling support
agent = Agent(
    model=BedrockModel(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="eu-west-1",
    ),
    tools=[...],
)

# Also supported: amazon.titan-*, meta.llama*, mistral.*
```

### Custom model adapter

```python
from collections.abc import Iterator
from SEJO_SDK.model import ModelClient
from SEJO_SDK.messages import Message, ModelResponse

class MyModel(ModelClient):
    def send_prompt(self, prompt: str, **kwargs) -> ModelResponse:
        return ModelResponse(content=f"Echo: {prompt}")

    def send_messages(self, messages: list[Message], **kwargs) -> ModelResponse:
        last = messages[-1].content if messages else ""
        return ModelResponse(content=f"Echo: {last}")

    def stream_response(self, prompt: str, **kwargs) -> Iterator[str]:
        yield "Echo: "
        yield prompt

    def stream_messages(self, messages: list[Message], **kwargs) -> Iterator[str]:
        yield from self.stream_response(messages[-1].content if messages else "")
```

## Project structure

```text
SEJO_SDK/
  agent.py             Agent runtime (run, stream, tools, multi-agent, evals)
  model.py             ModelClient / AsyncModelClient base interfaces
  messages.py          Message, ToolCall, ModelResponse, Usage
  memory.py            Conversation memory
  models/              Provider adapters (OpenAI, Anthropic, Gemini, DeepSeek, Bedrock)
  tools/               Tool primitive and built-in integrations
  evals.py             EvalSuite, scorers, EvalReport
  sessions.py          InMemorySessionStore, PostgresSessionStore
  sessions_redis.py    RedisSessionStore
  resilience.py        RetryModel, FallbackModel
  structured.py        parse_structured, schema_prompt
  prompts.py           PromptTemplate
  tracing.py           Tracer, TraceSummary
  server.py            FastAPI app factory
  errors.py            SDK error types
tests/                 229 tests — no API keys required
```

## Documentation

Extended docs in [`docs/`](docs/index.md) — run locally with:

```bash
mkdocs serve
```

## Links

- [Changelog](CHANGELOG.md)
- GitHub: [jmiguelmangas](https://github.com/jmiguelmangas)
- Email: [jmmangas@gmail.com](mailto:jmmangas@gmail.com)

## License

MIT. See [LICENSE](LICENSE).
