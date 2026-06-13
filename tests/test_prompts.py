"""Tests for PromptTemplate."""

from __future__ import annotations

import pytest

from SEJO_SDK.prompts import PromptTemplate, dedent_template


def test_render_single_variable():
    t = PromptTemplate("Hello, {name}!")
    assert t.render(name="Alice") == "Hello, Alice!"


def test_render_multiple_variables():
    t = PromptTemplate("You are a {role} speaking {language}.")
    result = t.render(role="pilot", language="Spanish")
    assert result == "You are a pilot speaking Spanish."


def test_variables_property():
    t = PromptTemplate("Fly {airline} from {origin} to {dest}.")
    assert t.variables == {"airline", "origin", "dest"}


def test_render_missing_variable_raises():
    t = PromptTemplate("Hello {name}, welcome to {place}.")
    with pytest.raises(KeyError, match="Missing template variables"):
        t.render(name="Bob")


def test_call_syntax():
    t = PromptTemplate("{a} + {b}")
    assert t(a="1", b="2") == "1 + 2"


def test_no_variables():
    t = PromptTemplate("Static prompt with no vars.")
    assert t.render() == "Static prompt with no vars."
    assert t.variables == frozenset()


def test_repr_truncates():
    t = PromptTemplate("x" * 100)
    assert "..." in repr(t)


def test_dedent_template():
    t = dedent_template("""
        You are a {role}.
        Answer in {language}.
    """)
    result = t.render(role="assistant", language="English")
    assert result == "You are a assistant.\nAnswer in English."
