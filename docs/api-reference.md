# API Reference

This page documents the public API surface.

## Package Exports

From `SEJO_SDK`:

- `Agent`
- `Memory`
- `Message`
- `ModelClient`
- `AsyncModelClient`
- `ModelResponse`
- `Model_client`
- `AsyncModel_client`
- `Tool`
- `ToolCall`
- `SejoSDKError`
- `ProviderDependencyError`
- `ProviderResponseError`
- `ToolNotFoundError`
- `ToolExecutionError`

## Agent

Import:

```python
from SEJO_SDK.agent import Agent
```

Constructor:

```python
Agent(
    model,
    memory=None,
    tools=None,
    system_prompt=None,
)
```

Arguments:

- `model`: a `ModelClient` or `AsyncModelClient`.
- `memory`: optional `Memory` instance.
- `tools`: optional dictionary of tools or iterable of `Tool` objects.
- `system_prompt`: optional system instruction prepended to native messages.

Methods:

```python
run(user_input: str) -> str
```

Runs a sync model with native messages and stores the response in memory.

```python
stream(user_input: str)
```

Yields sync streaming chunks and stores the joined response in memory.

```python
async arun(user_input: str) -> str
```

Runs an async model and stores the response in memory.

```python
async astream(user_input: str)
```

Yields async streaming chunks and stores the joined response in memory.

```python
run_with_tools(user_input: str, max_tool_iterations: int = 5) -> str
```

Runs a sync tool-calling loop.

```python
async arun_with_tools(user_input: str, max_tool_iterations: int = 5) -> str
```

Runs an async tool-calling loop.

```python
run_tool(name: str, **arguments)
```

Runs a registered tool by name.

```python
async arun_tool(name: str, **arguments)
```

Runs a registered tool and awaits async callables.

```python
tool_schemas() -> list[dict]
```

Returns schemas for all registered tools.

## Memory

Import:

```python
from SEJO_SDK.memory import Memory
```

Constructor:

```python
Memory(max_size: int = 20)
```

Methods:

- `add_ai_message(message: str) -> None`
- `add_user_message(message: str) -> None`
- `add_message(role: str, content: str) -> None`
- `add_tool_message(name: str, content: str, tool_call_id=None) -> None`
- `update(message: dict[str, str]) -> None`
- `clear() -> None`
- `get_context() -> str`

Attributes:

- `history`: list of message dictionaries.
- `max_size`: maximum number of stored messages.

## ModelClient

Import:

```python
from SEJO_SDK.model import ModelClient
```

Abstract methods:

`send_prompt` returns either a string or a `ModelResponse`.

```python
send_messages(messages: list[Message], **kwargs)
```

Sends native chat messages. The base class falls back to rendering messages as a
legacy prompt.

```python
stream_response(prompt: str, **kwargs) -> Iterator[str]
```

## AsyncModelClient

Import:

```python
from SEJO_SDK.model import AsyncModelClient
```

Abstract methods:

`send_prompt` returns either a string or a `ModelResponse`.

```python
async send_messages(messages: list[Message], **kwargs)
```

Sends native chat messages. The base class falls back to rendering messages as a
legacy prompt.

```python
stream_response(prompt: str, **kwargs) -> AsyncIterator[str]
```

## Messages

Import:

```python
from SEJO_SDK.messages import Message, ModelResponse, ToolCall
```

`Message` fields:

- `role`
- `content`
- `name`
- `tool_call_id`

`ToolCall` fields:

- `name`
- `arguments`
- `id`

`ModelResponse` fields:

- `content`
- `tool_calls`

Helper constructors:

- `system_message`
- `user_message`
- `assistant_message`
- `tool_message`
- `messages_to_prompt`
- `messages_to_dicts`

## Tool

Import:

```python
from SEJO_SDK.tools import Tool
```

Constructor:

```python
Tool(
    name: str,
    description: str,
    func,
    parameters=None,
)
```

Methods:

```python
run(*args, **kwargs)
```

Runs the wrapped callable.

```python
async arun(*args, **kwargs)
```

Runs the callable and awaits the result when it is awaitable.

```python
to_schema() -> dict
```

Returns a provider-friendly function schema.

## Provider Models

From `SEJO_SDK.models`:

- `OpenAIModel`
- `AsyncOpenAIModel`
- `AnthropicModel`
- `AsyncAnthropicModel`
- `GeminiModel`
- `AsyncGeminiModel`
- `DeepSeekModel`
- `AsyncDeepSeekModel`

All provider adapters accept an optional injected `client`.

## Optional Tool Integrations

From `SEJO_SDK.tools`:

- `WebSearchTool`
- `create_web_search_tool`
- `Tool`

Additional modules:

- `SEJO_SDK.tools.database_postgresql.DatabasePostgreSQLTool`
- `SEJO_SDK.tools.chat_ws_tool.create_chat_app`

## Utilities

From `SEJO_SDK.utils`:

- `Utils`
- `PostgresqlConnector`

`Utils` includes helpers for JSON files, environment variables and logging.

`PostgresqlConnector` provides connect, disconnect and query execution helpers
for PostgreSQL.
