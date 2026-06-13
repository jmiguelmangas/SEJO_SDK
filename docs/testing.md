# Testing

SEJO SDK keeps tests fast and deterministic.

## Run Tests

```bash
python -m pytest
```

## Run Coverage

```bash
python -m pytest --cov=SEJO_SDK --cov-report=term-missing
```

## Run Lint

```bash
python -m ruff check .
```

## Compile Check

```bash
python -m compileall SEJO_SDK tests examples
```

## Build Docs

```bash
python -m mkdocs build --strict
```

## Test Strategy

The suite avoids live provider calls.

Provider adapters accept injected clients, so tests can:

- record request arguments
- return small fake response objects
- simulate streaming chunks
- avoid API keys
- run without network access

## Testing Agents

Use fake models:

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

## Testing Async Agents

Use `asyncio.run`:

```python
import asyncio


def test_agent_arun():
    async def exercise():
        response = await agent.arun("Hello")
        assert response == "ok"

    asyncio.run(exercise())
```

## Testing Provider Adapters

Inject a fake provider client:

```python
model = OpenAIModel(
    api_key="test",
    model_name="test-model",
    client=fake_client,
)
```

Then assert the fake received the expected arguments:

```python
assert fake_client.calls[0]["model"] == "test-model"
```

## Testing Tools

Test local execution:

```python
tool = Tool(
    name="double",
    description="Double a number",
    func=lambda value: value * 2,
)

assert tool.run(21) == 42
```

Test schemas:

```python
assert tool.to_schema()["function"]["name"] == "double"
```

## Coverage Targets

Useful high-signal coverage areas:

- agent prompt construction
- memory size behavior
- sync and async streaming
- provider request shape
- tool-calling loops
- missing optional dependency errors
- tool schema generation
- utility helpers
- package build metadata

## Local Verification Checklist

Before publishing or opening a release PR:

```bash
python -m ruff check .
python -m pytest --cov=SEJO_SDK --cov-report=term-missing
python -m compileall SEJO_SDK tests examples
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```
