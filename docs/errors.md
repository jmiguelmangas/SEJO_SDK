# Errors

SEJO SDK exposes a small exception hierarchy in `SEJO_SDK.errors`.

## Exception Hierarchy

```text
SejoSDKError
  ProviderDependencyError
  ProviderResponseError
  ToolNotFoundError
  ToolExecutionError
```

## SejoSDKError

Base exception for SDK-specific errors.

Catch this when you want to handle all SEJO SDK errors together:

```python
from SEJO_SDK.errors import SejoSDKError


try:
    ...
except SejoSDKError as exc:
    print(exc)
```

## ProviderDependencyError

Raised when an optional provider dependency is required but not installed.

Example:

```python
from SEJO_SDK.models import OpenAIModel


model = OpenAIModel(api_key="key", model_name="model")
```

If `openai` is not installed, SEJO SDK raises:

```text
Install OpenAI support with `pip install sejo-sdk[openai]`.
```

## ProviderResponseError

Reserved for cases where a provider response cannot be parsed.

The current adapters mostly return empty strings for missing text fields and
avoid raising this error. It exists as the common error type for stricter
response parsing in future releases.

## ToolNotFoundError

Raised when an agent is asked to execute a tool that is not registered.

```python
agent.run_tool("missing")
```

## ToolExecutionError

Raised when a registered tool fails during execution.

The original exception is preserved as the cause, so callers can inspect the
underlying failure when needed.

```python
try:
    agent.run_tool("lookup", query="SEJO")
except ToolExecutionError as exc:
    print(exc)
```

## Runtime Errors From Optional Integrations

Some optional integrations may raise `RuntimeError` for operational failures.

Examples:

- PostgreSQL connection failure.
- PostgreSQL query execution failure.
- Query execution attempted before connecting.

These errors wrap lower-level exceptions with a clearer SDK-facing message.

## Recommended Handling

For application code:

```python
from SEJO_SDK.errors import ProviderDependencyError, SejoSDKError


try:
    response = agent.run("Hello")
except ProviderDependencyError as exc:
    print(f"Missing dependency: {exc}")
except SejoSDKError as exc:
    print(f"SDK error: {exc}")
```

For tests, assert the exact error type where possible.
