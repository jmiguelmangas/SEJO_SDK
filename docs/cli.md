# CLI

SEJO SDK includes a small command line interface.

## Doctor

Run:

```bash
sejo doctor
```

The command reports:

- Python version and executable
- installed SEJO SDK version
- optional provider extras detected in the environment
- development tools detected in the environment

Example:

```text
SEJO SDK Doctor
Python: 3.11.0 (.../python)
Package: sejo-sdk 0.2.0

Optional extras:
- openai: missing
- anthropic: missing

Development tools:
- pytest: ok
- ruff: ok
```

## Strict Mode

Use strict mode when you want missing optional extras or dev tools to fail the
command:

```bash
sejo doctor --strict
```

This is useful in CI or before a release when you want a fully provisioned
environment.
