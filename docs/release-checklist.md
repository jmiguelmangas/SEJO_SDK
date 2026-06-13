# Release Checklist

Use this checklist before publishing SEJO SDK to PyPI.

## 1. Clean Working Tree

Check the repository state:

```bash
git status --short
```

Make sure only intended files are modified.

## 2. Version

Update the version in `pyproject.toml`:

```toml
[project]
version = "0.2.1"
```

Use semantic versioning:

- patch for fixes and documentation
- minor for backward-compatible features
- major for breaking changes

## 3. Dependencies

Confirm optional dependencies are still accurate:

```toml
[project.optional-dependencies]
```

The core dependency list should stay small.

## 4. Lint

```bash
python -m ruff check .
```

## 5. Tests

```bash
python -m pytest
```

## 6. Coverage

```bash
python -m pytest --cov=SEJO_SDK --cov-report=term-missing
```

## 7. Compile

```bash
python -m compileall SEJO_SDK tests examples
```

## 8. Build Docs

Build docs before packaging:

```bash
python -m mkdocs build --strict
```

## 9. Clean Previous Build Artifacts

```bash
rm -rf dist build
```

## 10. Build

```bash
python -m build
```

Expected artifacts:

```text
dist/sejo_sdk-<version>.tar.gz
dist/sejo_sdk-<version>-py3-none-any.whl
```

## 11. Twine Check

```bash
python -m twine check dist/*
```

Both artifacts should pass.

## 12. Optional TestPyPI Publish

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI in a clean environment:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  sejo-sdk
```

## 13. PyPI Publish

```bash
python -m twine upload dist/*
```

## 14. Git Tag

```bash
git tag v0.2.1
git push origin v0.2.1
```

## 15. GitHub Release Notes

Include:

- new features
- fixes
- docs changes
- test and coverage summary
- migration notes if needed

## Current Known Release Notes

Recent additions include:

- async model interface
- async provider adapters
- async agent methods
- tool schema support
- tool-calling loops
- `sejo doctor` CLI
- MkDocs documentation site

## Release Workflow

The repository includes a manual GitHub Actions workflow:

```text
.github/workflows/release.yml
```

It validates lint, tests, compile checks, docs, package build and metadata. It
uploads package artifacts on every manual run.

Publishing to PyPI is gated by the `publish_pypi` workflow input and expects
PyPI Trusted Publishing to be configured for the repository.
- expanded test coverage
- package build validation
- extended docs
