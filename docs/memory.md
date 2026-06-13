# Memory

`Memory` stores conversation history for an agent.

## Message Shape

Messages are dictionaries with two keys:

```python
{"role": "user", "content": "Hello"}
```

Common roles:

- `user`
- `assistant`
- `system`
- `tool`

The memory object does not restrict role names. It stores whatever role string
you provide.

## Create Memory

```python
from SEJO_SDK.memory import Memory


memory = Memory()
```

Set a custom maximum size:

```python
memory = Memory(max_size=50)
```

The default maximum size is 20 messages.

## Add Messages

```python
memory.add_user_message("Hello")
memory.add_ai_message("Hi")
memory.add_message("system", "Be concise")
```

`add_ai_message` stores an `assistant` role. The method name is kept for
readability and backward compatibility.

## Update Memory

```python
memory.update({"role": "user", "content": "Next question"})
```

`update` appends a message and applies the same maximum-size behavior.

## Clear Memory

```python
memory.clear()
```

## Build Context

```python
context = memory.get_context()
```

The context is a newline-separated string:

```text
user: Hello
assistant: Hi
```

`Agent` uses this string when building prompts.

## Size Limit Behavior

When the number of messages exceeds `max_size`, the oldest message is removed.

```python
memory = Memory(max_size=2)

memory.add_user_message("one")
memory.add_ai_message("two")
memory.add_user_message("three")

assert memory.history == [
    {"role": "assistant", "content": "two"},
    {"role": "user", "content": "three"},
]
```

## Custom Memory

You can pass a memory instance into `Agent`:

```python
from SEJO_SDK.agent import Agent
from SEJO_SDK.memory import Memory


memory = Memory(max_size=100)
agent = Agent(model=model, memory=memory)
```

For advanced behavior, create your own object with the methods used by
`Agent`:

- `add_user_message`
- `add_ai_message`
- `get_context`

The built-in `Memory` class is intentionally simple so custom memory backends
can be introduced later without changing the agent API.
