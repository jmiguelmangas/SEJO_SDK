# Changelog

All notable changes to SEJO SDK will be documented in this file.

The project follows semantic versioning.

## Unreleased

### Added

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
