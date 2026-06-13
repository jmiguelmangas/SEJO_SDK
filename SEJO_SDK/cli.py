"""Command line interface for SEJO SDK."""

import argparse
import importlib.util
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from typing import Optional

OPTIONAL_DEPENDENCIES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google.generativeai",
    "websearch": "duckduckgo_search",
    "postgres": "psycopg2",
    "server": "fastapi",
}

DEV_TOOLS = {
    "build": "build",
    "pytest": "pytest",
    "ruff": "ruff",
    "twine": "twine",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sejo", description="SEJO SDK CLI")
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check local SEJO SDK installation health.",
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when optional extras or dev tools are missing.",
    )

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return doctor(strict=args.strict)

    parser.print_help()
    return 0


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
