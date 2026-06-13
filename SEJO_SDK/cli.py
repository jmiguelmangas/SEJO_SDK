"""Command line interface for SEJO SDK."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from collections.abc import Sequence
from importlib import metadata

OPTIONAL_DEPENDENCIES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google.generativeai",
    "websearch": "duckduckgo_search",
    "postgres": "psycopg2",
    "server": "fastapi",
    "redis": "redis",
    "mcp": "mcp",
    "otel": "opentelemetry",
}

DEV_TOOLS = {
    "build": "build",
    "pytest": "pytest",
    "ruff": "ruff",
    "twine": "twine",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sejo", description="SEJO SDK CLI")
    subparsers = parser.add_subparsers(dest="command")

    # -- doctor ---------------------------------------------------------------
    doctor_p = subparsers.add_parser(
        "doctor", help="Check local SEJO SDK installation health."
    )
    doctor_p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when optional extras or dev tools are missing.",
    )

    # -- chat -----------------------------------------------------------------
    chat_p = subparsers.add_parser(
        "chat", help="Start an interactive REPL with an agent."
    )
    _add_model_args(chat_p)
    chat_p.add_argument(
        "--system", default="You are a helpful assistant.", metavar="PROMPT",
        help="System prompt for the agent.",
    )
    chat_p.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colour output."
    )

    # -- serve ----------------------------------------------------------------
    serve_p = subparsers.add_parser(
        "serve", help="Serve an agent via the built-in FastAPI server."
    )
    _add_model_args(serve_p)
    serve_p.add_argument(
        "--system", default="You are a helpful assistant.", metavar="PROMPT"
    )
    serve_p.add_argument("--host", default="0.0.0.0", metavar="HOST")
    serve_p.add_argument("--port", default=8000, type=int, metavar="PORT")
    serve_p.add_argument("--title", default="SEJO Agent", metavar="TITLE")
    serve_p.add_argument("--reload", action="store_true")

    # -- eval -----------------------------------------------------------------
    eval_p = subparsers.add_parser(
        "eval", help="Run an eval suite defined in a Python file."
    )
    eval_p.add_argument("file", help="Python file that defines a 'suite' variable.")
    eval_p.add_argument(
        "--agent", default=None, metavar="VAR",
        help="Name of the Agent variable in the file (default: 'agent').",
    )
    eval_p.add_argument(
        "--fail-under", default=0.0, type=float, metavar="RATE",
        help="Exit non-zero if pass rate is below this threshold (0.0–1.0).",
    )

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return doctor(strict=args.strict)
    if args.command == "chat":
        return cmd_chat(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "eval":
        return cmd_eval(args)

    parser.print_help()
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def doctor(strict: bool = False) -> int:
    version = _package_version()
    print("SEJO SDK Doctor")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Package: sejo-sdk {version}")
    print("")
    print("Optional extras:")
    optional_missing = _print_module_checks(OPTIONAL_DEPENDENCIES)
    print("")
    print("Development tools:")
    dev_missing = _print_module_checks(DEV_TOOLS)
    if strict and (optional_missing or dev_missing):
        return 1
    return 0


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

def cmd_chat(args: argparse.Namespace) -> int:
    model = _build_model(args)
    if model is None:
        return 1
    from SEJO_SDK.agent import Agent
    agent = Agent(model=model, system_prompt=args.system)

    use_color = not args.no_color and sys.stdout.isatty()
    cyan = "\033[36m" if use_color else ""
    reset = "\033[0m" if use_color else ""
    bold = "\033[1m" if use_color else ""

    print(f"{bold}SEJO Chat{reset} — type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            user_input = input(f"{cyan}You:{reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue
        try:
            response = agent.run(user_input)
            print(f"{bold}Agent:{reset} {response}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}\n", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Run: pip install sejo-sdk[server]",
            file=sys.stderr,
        )
        return 1

    model = _build_model(args)
    if model is None:
        return 1

    from SEJO_SDK.server import create_agent_app
    app = create_agent_app(
        model=model,
        system_prompt=args.system,
        title=args.title,
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> int:
    import importlib.util as ilu

    spec = ilu.spec_from_file_location("_sejo_eval_module", args.file)
    if spec is None or spec.loader is None:
        print(f"Cannot load file: {args.file}", file=sys.stderr)
        return 1

    mod = ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading {args.file}: {exc}", file=sys.stderr)
        return 1

    suite = getattr(mod, "suite", None)
    if suite is None:
        print(f"No 'suite' variable found in {args.file}", file=sys.stderr)
        return 1

    agent_var = args.agent or "agent"
    agent = getattr(mod, agent_var, None)
    if agent is None:
        print(f"No '{agent_var}' variable found in {args.file}", file=sys.stderr)
        return 1

    try:
        report = suite.run(agent)
    except Exception as exc:  # noqa: BLE001
        print(f"Eval failed: {exc}", file=sys.stderr)
        return 1

    report.print_report()

    if report.pass_rate() < args.fail_under:
        print(
            f"Pass rate {report.pass_rate():.0%} < "
            f"required {args.fail_under:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "deepseek", "bedrock"],
        default="openai",
        help="Model provider (default: openai).",
    )
    p.add_argument("--model", default=None, metavar="MODEL_NAME")
    p.add_argument("--api-key", default=None, metavar="KEY")


_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini": "gemini-1.5-flash",
    "deepseek": "deepseek-chat",
    "bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",
}


def _build_model(args: argparse.Namespace):
    import os
    provider = args.provider
    model_name = args.model or _DEFAULTS[provider]
    api_key = args.api_key or os.environ.get(
        {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
         "gemini": "GOOGLE_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
         "bedrock": ""}.get(provider, ""), ""
    )

    try:
        if provider == "openai":
            from SEJO_SDK.models import OpenAIModel
            return OpenAIModel(api_key=api_key, model_name=model_name)
        if provider == "anthropic":
            from SEJO_SDK.models import AnthropicModel
            return AnthropicModel(api_key=api_key, model_name=model_name)
        if provider == "gemini":
            from SEJO_SDK.models import GeminiModel
            return GeminiModel(api_key=api_key, model_name=model_name)
        if provider == "deepseek":
            from SEJO_SDK.models.model_deepseek import DeepSeekModel
            return DeepSeekModel(api_key=api_key, model_name=model_name)
        if provider == "bedrock":
            from SEJO_SDK.models import BedrockModel
            return BedrockModel(model_id=model_name)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to initialise {provider} model: {exc}", file=sys.stderr)
        return None

    print(f"Unknown provider: {provider}", file=sys.stderr)
    return None


def _print_module_checks(modules: dict[str, str]) -> list[str]:
    missing = []
    for label, module_name in modules.items():
        available = _module_available(module_name)
        status = "ok" if available else "missing"
        print(f"- {label}: {status}")
        if not available:
            missing.append(label)
    return missing


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _package_version() -> str:
    try:
        return metadata.version("sejo-sdk")
    except metadata.PackageNotFoundError:
        return "editable/local"


if __name__ == "__main__":
    raise SystemExit(main())
