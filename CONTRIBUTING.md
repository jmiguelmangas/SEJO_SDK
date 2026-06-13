# Contributing

Thanks for considering a contribution to SEJO SDK.

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

## Quality Checks

Run these before opening a pull request:

```bash
python -m ruff check .
python -m pytest --cov=SEJO_SDK --cov-report=term-missing
python -m compileall SEJO_SDK tests examples
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

## Testing Guidelines

- Keep unit tests deterministic.
- Do not require API keys for default tests.
- Inject fake provider clients instead of making network calls.
- Add tests for sync and async behavior when both paths exist.
- Add docs or examples when public APIs change.

## Code Style

- Use Ruff for linting.
- Keep dependencies optional unless they are required by the core package.
- Prefer small interfaces over provider-specific coupling.
- Preserve backward compatibility where practical.

## Pull Requests

Good pull requests include:

- a focused change
- tests for changed behavior
- docs for public behavior
- a short explanation of tradeoffs

## Release Changes

For user-facing changes, update `CHANGELOG.md`.
