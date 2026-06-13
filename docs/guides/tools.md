# Tools

Tools let an agent expose local Python functions through a named registry.

SEJO SDK currently provides:

- local sync execution
- local async execution
- provider-friendly tool schemas
- sync and async tool-calling loops

## Define A Tool

```python
from SEJO_SDK.tools import Tool


def add(left: int, right: int) -> int:
    return left + right


add_tool = Tool(
    name="add",
    description="Add two integers.",
    func=add,
)
```

## Register Tools On An Agent

```python
from SEJO_SDK.agent import Agent


agent = Agent(
    model=model,
    tools=[add_tool],
)
```

Tools can also be passed as a dictionary:

```python
agent = Agent(
    model=model,
    tools={"add": add_tool},
)
```

## Run A Tool

```python
result = agent.run_tool("add", left=2, right=3)
```

## Async Tool Execution

`Tool.arun` awaits async callables and returns normal values unchanged for sync
callables.

```python
async def fetch_user(user_id: str):
    return {"id": user_id}


tool = Tool(
    name="fetch_user",
    description="Fetch a user by id.",
    func=fetch_user,
)

result = await tool.arun(user_id="user_123")
```

The same behavior is exposed through the agent:

```python
result = await agent.arun_tool("fetch_user", user_id="user_123")
```

## Tool Schemas

Pass `parameters` to describe the callable arguments:

```python
add_tool = Tool(
    name="add",
    description="Add two integers.",
    func=add,
    parameters={
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
    },
)
```

Generate a provider-friendly schema:

```python
add_tool.to_schema()
```

The shape is:

```python
{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
        },
    },
}
```

Generate all schemas from an agent:

```python
schemas = agent.tool_schemas()
```

## Run A Tool-Calling Loop

Models can request tool calls by returning `ModelResponse` objects:

```python
from SEJO_SDK.messages import ModelResponse, ToolCall


class ToolCallingModel(ModelClient):
    def __init__(self):
        self.calls = 0

    def send_prompt(self, prompt: str, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="I will calculate that.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="add",
                        arguments={"left": 2, "right": 3},
                    )
                ],
            )
        return "2 + 3 is 5."

    def stream_response(self, prompt: str, **kwargs):
        yield "unused"
```

Run the loop:

```python
response = agent.run_with_tools("What is 2 + 3?")
```

The loop:

1. Sends the prompt with `tools=agent.tool_schemas()`.
2. Reads tool calls from the model response.
3. Executes local tools.
4. Stores tool result messages in memory.
5. Sends the updated context back to the model.
6. Returns the final assistant response.

Async agents use the same flow:

```python
response = await agent.arun_with_tools("What is 2 + 3?")
```

## Default Empty Parameters

If no `parameters` are provided, `Tool.to_schema()` returns an empty object
schema:

```python
{
    "type": "object",
    "properties": {},
}
```

## Current Tool Integrations

SEJO SDK includes optional tool integrations:

- `WebSearchTool` using DuckDuckGo search.
- `DatabasePostgreSQLTool` for wrapping PostgreSQL query functions.
- `create_chat_app` for exposing an agent over FastAPI WebSocket.

These integrations live behind optional dependencies.

## Web Search Tool

Install the web search extra:

```bash
pip install "sejo-sdk[websearch]"
```

Create the tool:

```python
from SEJO_SDK.tools import create_web_search_tool


tool = create_web_search_tool(max_results=3)
```

## PostgreSQL Tool

Install the postgres extra:

```bash
pip install "sejo-sdk[postgres]"
```

The lower-level connector is available as `PostgresqlConnector`. You can wrap
your own query function with `DatabasePostgreSQLTool`.

## Provider Support

| Provider | Tool calling |
|----------|-------------|
| OpenAI   | Native — returns `ModelResponse` with `ToolCall` objects. |
| DeepSeek | Same OpenAI-compatible parser. |
| Anthropic | Native — schemas converted to `input_schema`, responses parsed from `tool_use` blocks. |
| Gemini   | Prompt-based fallback (native tool calling planned). |
