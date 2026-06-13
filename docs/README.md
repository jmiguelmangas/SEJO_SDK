# SEJO SDK Documentation

Welcome to the SEJO SDK documentation.

SEJO SDK is a small Python package for building provider-agnostic AI agents. It
keeps the core package lightweight, lets provider SDKs live behind optional
extras, and gives you a shared runtime for memory, streaming, async clients and
local tools.

## Documentation Map

- [Getting Started](getting-started.md): installation, first agent and local
  development.
- [Core Concepts](concepts.md): models, agents, memory, tools and provider
  extras.
- [Sync Agents](guides/sync-agents.md): using `Agent.run` and
  `Agent.stream`.
- [Async Agents](guides/async-agents.md): using `Agent.arun` and
  `Agent.astream`.
- [Native Messages](guides/native-messages.md): `Message`, `send_messages` and
  system prompts.
- [Tools](guides/tools.md): local tool execution and schema generation.
- [Providers](providers.md): OpenAI, Anthropic, Gemini and DeepSeek adapters.
- [Memory](memory.md): conversation context behavior and customization.
- [Errors](errors.md): exception hierarchy and dependency errors.
- [CLI](cli.md): `sejo doctor` and local environment checks.
- [API Reference](api-reference.md): public classes and methods.
- [Examples](examples.md): runnable examples included in the repository.
- [Testing](testing.md): unit tests, coverage and no-network provider fakes.
- [Release Checklist](release-checklist.md): PyPI build and validation flow.

## Current Package Shape

```text
SEJO_SDK/
  agent.py                  Agent runtime.
  memory.py                 Conversation memory.
  model.py                  Sync and async model interfaces.
  messages.py               Typed messages, tool calls and model responses.
  models/                   Provider adapters.
  tools/                    Tool primitive and optional tool integrations.
  utils/                    Utility helpers.
examples/                   Small runnable examples.
tests/                      No-network unit tests.
docs/                       Project documentation.
```

## Design Goals

SEJO SDK is intentionally modest:

- The core install should stay small.
- Optional provider dependencies should be installed only when needed.
- Adapters should be easy to fake in tests.
- Sync and async usage should feel similar.
- Tool support should expose provider-friendly schemas and a local execution
  loop while keeping provider-specific details isolated.

## Quick Links

Install the package locally:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest
```

Run coverage:

```bash
python -m pytest --cov=SEJO_SDK --cov-report=term-missing
```

Build package artifacts:

```bash
python -m build
python -m twine check dist/*
```
