## Summary

Describe the change and why it is needed.

## Testing

- [ ] `python -m ruff check .`
- [ ] `python -m pytest --cov=SEJO_SDK --cov-report=term-missing`
- [ ] `python -m compileall SEJO_SDK tests examples`
- [ ] `python -m mkdocs build --strict`
- [ ] `python -m build`
- [ ] `python -m twine check dist/*`

## Checklist

- [ ] Tests added or updated
- [ ] Docs/examples updated for public API changes
- [ ] `CHANGELOG.md` updated for user-facing changes
- [ ] No API keys, secrets or local artifacts committed
