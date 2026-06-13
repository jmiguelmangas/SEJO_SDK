# Security Policy

## Supported Versions

SEJO SDK is currently alpha. Security fixes target the latest released version.

## Reporting A Vulnerability

Please do not open a public issue for sensitive security reports.

Send a report to:

```text
jmiguelmangas@gmail.com
```

Include:

- affected version or commit
- description of the issue
- reproduction steps
- potential impact
- suggested fix, if known

## Scope

Security-sensitive areas include:

- tool execution
- database connectors
- server integrations
- provider credential handling
- CLI behavior that inspects local environments

## Dependency Security

Provider SDKs and optional integrations are installed through extras. Keep
optional dependencies updated and avoid adding core dependencies unless truly
needed.
