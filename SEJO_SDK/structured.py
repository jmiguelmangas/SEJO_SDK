"""Structured output helpers: parse model responses into typed Pydantic models."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass

T = TypeVar("T")

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> str:
    """Extract JSON from a string, stripping markdown fences if present."""
    match = _JSON_FENCE.search(text)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def parse_structured(text: str, schema: type[T]) -> T:
    """Parse a model's text output into a typed Pydantic model.

    Strips markdown code fences and extracts JSON before validating.

    Args:
        text:   Raw text from the model.
        schema: A Pydantic BaseModel subclass.

    Returns:
        An instance of ``schema`` populated from the model output.

    Raises:
        ValueError: if the text cannot be parsed as valid JSON or fails validation.
    """
    raw_json = _extract_json(text)
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model response is not valid JSON.\nRaw output:\n{text}"
        ) from exc

    try:
        return schema(**data)  # type: ignore[return-value]
    except Exception as exc:
        raise ValueError(
            f"JSON does not match schema {schema.__name__}: {exc}\nData: {data}"
        ) from exc


def schema_prompt(schema: Any) -> str:
    """Return a prompt suffix that instructs the model to return JSON matching schema.

    Works with Pydantic v1 and v2.
    """
    try:
        # Pydantic v2
        schema_json = schema.model_json_schema()
    except AttributeError:
        try:
            # Pydantic v1
            schema_json = schema.schema()
        except AttributeError:
            schema_json = {}

    return (
        "\n\nRespond ONLY with a valid JSON object matching this schema "
        "(no extra text, no markdown fences):\n"
        + json.dumps(schema_json, indent=2)
    )
