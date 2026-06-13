"""Auto-tool decorator — turn any Python function into a SEJO Tool.

``@tool`` introspects the function's type annotations and docstring to
build the JSON schema automatically.  No manual schema writing required.

Example::

    from SEJO_SDK.tools import tool

    @tool
    def get_flight_status(flight_id: str, date: str = "today") -> str:
        \"\"\"Get live status for a BA flight.

        Args:
            flight_id: The flight number, e.g. 'BA123'.
            date:      Date in YYYY-MM-DD format or 'today'.
        \"\"\"
        return f"{flight_id} on {date}: on time"

    agent = Agent(model=..., tools=[get_flight_status])
    agent.run_with_tools("What's the status of BA456?")

The returned object is a regular ``Tool`` instance — you can pass it
anywhere a ``Tool`` is expected.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from SEJO_SDK.tools.base import Tool

_PY_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _py_type_to_json(annotation: Any) -> str:
    return _PY_TO_JSON.get(annotation, "string")


def _parse_google_docstring(doc: str) -> dict[str, str]:
    """Extract param descriptions from a Google-style docstring."""
    descriptions: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped[0].isspace() and stripped.endswith(":"):
                # New section header — stop
                in_args = False
                continue
            if ":" in stripped:
                param, _, desc = stripped.partition(":")
                descriptions[param.strip()] = desc.strip()
    return descriptions


def tool(fn: Callable[..., Any]) -> Tool:
    """Decorator that converts a function into a ``Tool``.

    Reads type hints for parameter types and the docstring for
    descriptions.  The function name and first docstring line become
    the tool name and description.

    Supports both sync and async functions.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001
        hints = {}

    doc = inspect.getdoc(fn) or ""
    first_line = doc.splitlines()[0].strip() if doc else fn.__name__
    param_docs = _parse_google_docstring(doc)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        json_type = _py_type_to_json(hints.get(name, str))
        prop: dict[str, Any] = {"type": json_type}
        if name in param_docs:
            prop["description"] = param_docs[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    is_async = inspect.iscoroutinefunction(fn)

    if is_async:
        def _sync_stub(**kwargs: Any) -> Any:
            raise NotImplementedError(f"Call {fn.__name__} via arun_with_tools.")

        return Tool(
            name=fn.__name__,
            description=first_line,
            func=_sync_stub,
            afunc=fn,
            parameters=parameters,
        )

    return Tool(
        name=fn.__name__,
        description=first_line,
        func=fn,
        parameters=parameters,
    )
