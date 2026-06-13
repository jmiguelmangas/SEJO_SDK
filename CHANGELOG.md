# Changelog

All notable changes to SEJO SDK will be documented in this file.

The project follows semantic versioning.

## Unreleased

### Added

- **RedisSessionStore** — session persistence backed by Redis with optional TTL;
  injectable client for testing without a live Redis instance.
- **EvalSuite / EvalReport** — dataset-based agent evaluation framework with
  built-in scorers (`exact_match`, `contains`, `contains_all`, `llm_judge`).
- **PostgresSessionStore** — session persistence backed by PostgreSQL; table
  auto-created on first use; injectable connection factory for testing.
- **Multi-agent**: `Agent.as_tool()` wraps any agent as a `Tool` so an
  orchestrator can call it natively in `run_with_tools`; `Agent.delegate()`
  and `adelegate()` for direct sub-agent invocation.
- **Gemini native tool calling** — `messages_to_gemini_contents`,
  `tools_to_gemini` and `parse_gemini_response` adapters; `GeminiModel` now
  passes tool schemas to the API and parses `function_call` parts in responses.
- **PromptTemplate** — named variable substitution with `render(**kwargs)`;
  `dedent_template` helper for triple-quoted prompts.
- **RetryModel / AsyncRetryModel** — transparent retry with exponential
  back-off around any `ModelClient`.
- **FallbackModel / AsyncFallbackModel** — tries providers in order, returning
  the first successful response.
- **Structured outputs** — `agent.run_structured(prompt, Schema)` and
  `parse_structured(text, Schema)` for Pydantic-validated model responses.
- **Tracer** — per-turn token accounting and cost estimation (opt-in via
  `Agent(tracer=Tracer(model_name=...))`).
- **InMemorySessionStore** — in-process session management with `create`,
  `get`, `get_or_create`, `save`, `delete` and `list_sessions`.
- **`create_agent_app()`** — FastAPI server factory with REST (`POST /chat`)
  and WebSocket (`/ws/{session_id}`) endpoints.
- **AWS Bedrock connector** — `BedrockModel` and `AsyncBedrockModel` supporting
  Claude, Titan, Llama and Mistral families; native tool calling for Claude.
- Async model interface and async provider adapters.
- Typed messages, tool calls and model responses.
- Native `send_messages` support and agent system prompts.
- Sync and async tool-calling loops.
- Tool schemas and async tool execution.
- `sejo doctor` CLI health check.
- MkDocs Material documentation site.
- Expanded tests and release validation.
- Anthropic native tool calling: schemas converted to `input_schema`, responses
  parsed from `tool_use` blocks, tool results serialised as `tool_result`
  content blocks.
- `assistant_message_with_tools` helper to preserve tool call context in memory.
- `Message.tool_calls` field for round-tripping assistant tool-use turns.
- `ModelClient.stream_messages` and `AsyncModelClient.stream_messages` — stream
  a response from a list of `Message` objects. Custom adapters that only
  implement `stream_response` inherit a prompt-rendering fallback.
- `Agent.stream` and `astream` now use `stream_messages` so system prompts and
  full conversation history are passed natively.

### Changed

- Provider adapters can preserve OpenAI-compatible tool calls as typed
  `ModelResponse` objects.
- CI validates lint, tests, compile checks, docs, package build and metadata.

### Fixed

- Tool execution now raises SDK-specific errors for missing tools and tool
  failures.

## 0.2.0

- Lightweight provider-agnostic agent runtime.
- Conversation memory.
- Sync provider adapters for OpenAI, Anthropic, Gemini and DeepSeek.
- Optional tool integrations for web search, PostgreSQL and FastAPI WebSocket
  chat.
- PyPI metadata and optional dependency extras.
