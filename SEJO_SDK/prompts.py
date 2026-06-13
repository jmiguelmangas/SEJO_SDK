"""Prompt templates with variable substitution."""

from __future__ import annotations

from string import Formatter


class PromptTemplate:
    """A simple prompt template with named variable substitution.

    Example::

        t = PromptTemplate("You are a {role}. Answer in {language}.")
        prompt = t.render(role="pilot", language="Spanish")
    """

    def __init__(self, template: str) -> None:
        self.template = template
        self._variables: frozenset[str] = frozenset(
            name
            for _, name, _, _ in Formatter().parse(template)
            if name is not None
        )

    @property
    def variables(self) -> frozenset[str]:
        """Return the set of variable names declared in this template."""
        return self._variables

    def render(self, **kwargs: object) -> str:
        """Render the template with the given variables.

        Raises:
            KeyError: if a required variable is missing.
        """
        missing = self._variables - kwargs.keys()
        if missing:
            raise KeyError(f"Missing template variables: {sorted(missing)}")
        return self.template.format(**kwargs)

    def __call__(self, **kwargs: object) -> str:
        return self.render(**kwargs)

    def __repr__(self) -> str:
        preview = self.template[:60] + ("..." if len(self.template) > 60 else "")
        return f"PromptTemplate({preview!r})"


def dedent_template(text: str) -> PromptTemplate:
    """Create a PromptTemplate from a triple-quoted string, stripping leading indent."""
    import textwrap

    return PromptTemplate(textwrap.dedent(text).strip())
