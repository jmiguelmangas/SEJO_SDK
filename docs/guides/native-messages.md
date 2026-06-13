# Native Messages

SEJO SDK supports typed chat messages through `Message` and provider-native
`send_messages` calls.

The older prompt-based API still works. Custom models that only implement
`send_prompt` inherit a fallback that renders messages into the legacy prompt
format.

## Message Helpers

```python
from SEJO_SDK.messages import (
    assistant_message,
    system_message,
    tool_message,
    user_message,
)


messages = [
    system_message("Be concise."),
    user_message("What is SEJO SDK?"),
]
```

## Send Messages Directly

```python
from SEJO_SDK.models import OpenAIModel


model = OpenAIModel(
    api_key="your-api-key",
    model_name="your-model-name",
)

response = model.send_messages(messages)
```

Provider adapters convert messages to their native shape where possible:

- OpenAI and DeepSeek use chat message dictionaries directly.
- Anthropic passes system messages through the provider `system` argument.
- Gemini renders messages to a structured prompt string for the current adapter.

## Agent System Prompts

```python
from SEJO_SDK.agent import Agent


agent = Agent(
    model=model,
    system_prompt="Answer in one sentence.",
)

response = agent.run("What is SEJO SDK?")
```

The system prompt is prepended to the native message list sent to the model.

## Legacy Fallback

If a custom model only implements `send_prompt`, it still works. The base class
converts messages to:

```text
system: Be concise.
user: Hello
Agent:
```

## Tool Messages

Tool results can be represented explicitly:

```python
tool_message(
    "42",
    name="calculator",
    tool_call_id="call_1",
)
```

OpenAI and Anthropic providers receive the `tool_call_id` metadata and
serialise results in their native format. Providers without native tool-result
support receive a readable text representation.

## Assistant Messages With Tool Calls

When replaying a conversation that included tool calls, you can preserve the
tool request in the assistant turn:

```python
from SEJO_SDK.messages import ToolCall, assistant_message_with_tools


msg = assistant_message_with_tools(
    "Let me look that up.",
    tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "SEJO"})],
)
```

Anthropic adapters convert these into `tool_use` content blocks so the provider
can correctly attribute tool results on the following user turn.
Agents do this automatically inside `run_with_tools` and `arun_with_tools`.
