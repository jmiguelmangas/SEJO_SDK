"""Agent evaluation framework.

Run a dataset of (input, expected_output) pairs against an Agent and score
the responses using built-in or custom scorers.

Example::

    from SEJO_SDK.evals import EvalCase, EvalSuite, contains

    suite = EvalSuite(
        name="BA assistant smoke test",
        cases=[
            EvalCase("What is the IATA code for Heathrow?", expected="LHR"),
            EvalCase("Hub airport for British Airways?", expected="Heathrow"),
        ],
        scorer=contains,
    )
    report = suite.run(agent)
    report.print_report()
    print(f"Pass rate: {report.pass_rate():.0%}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """A single evaluation case."""

    input: str
    expected: str
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self.input[:40].replace(" ", "_")


@dataclass
class EvalResult:
    """The outcome of running one EvalCase."""

    case: EvalCase
    actual: str
    score: float
    passed: bool
    duration_ms: float = 0.0
    error: str | None = None


class EvalReport:
    """Aggregated results from running an EvalSuite."""

    def __init__(self, name: str, results: list[EvalResult]) -> None:
        self.name = name
        self.results = results

    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def failed(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": len(self.failed()),
            "pass_rate": round(self.pass_rate(), 4),
            "avg_score": round(self.avg_score(), 4),
            "avg_duration_ms": round(
                sum(r.duration_ms for r in self.results) / max(len(self.results), 1),
                1,
            ),
        }

    def print_report(self) -> None:
        s = self.summary()
        print(f"\n{'=' * 60}")
        print(f"  Eval: {s['name']}")
        print(f"  {s['passed']}/{s['total']} passed  "
              f"({s['pass_rate']:.0%})  avg score {s['avg_score']:.2f}")
        print(f"{'=' * 60}")
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            print(f"  {icon}  [{r.case.id}]")
            if not r.passed:
                print(f"       expected: {r.case.expected!r}")
                print(f"       actual:   {r.actual[:120]!r}")
                if r.error:
                    print(f"       error:    {r.error}")
        print()


# ---------------------------------------------------------------------------
# Built-in scorers
# ---------------------------------------------------------------------------

def exact_match(actual: str, expected: str) -> float:
    """1.0 if outputs match exactly (case-insensitive, stripped), else 0.0."""
    return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0


def contains(actual: str, expected: str) -> float:
    """1.0 if the expected string appears anywhere in the actual output."""
    return 1.0 if expected.strip().lower() in actual.lower() else 0.0


def contains_all(actual: str, expected: str) -> float:
    """Score = fraction of comma-separated expected terms found in actual."""
    terms = [t.strip() for t in expected.split(",") if t.strip()]
    if not terms:
        return 1.0
    found = sum(1 for t in terms if t.lower() in actual.lower())
    return found / len(terms)


def llm_judge(
    model: Any,
    prompt_template: str = (
        "Does the following answer correctly address the question?\n"
        "Question: {input}\n"
        "Expected: {expected}\n"
        "Actual: {actual}\n"
        "Reply with exactly one word: YES or NO."
    ),
) -> Callable[[str, str, str], float]:
    """Return a scorer that uses another model to judge correctness.

    The scorer accepts (actual, expected, input) and returns 1.0 or 0.0.
    """
    def _judge(actual: str, expected: str, input_text: str = "") -> float:
        prompt = prompt_template.format(
            input=input_text, expected=expected, actual=actual
        )
        try:
            resp = model.send_prompt(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            return 1.0 if "yes" in text.lower() else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    return _judge


# ---------------------------------------------------------------------------
# EvalSuite
# ---------------------------------------------------------------------------

Scorer = Callable[..., float]


class EvalSuite:
    """A collection of EvalCases with a shared scorer.

    Args:
        cases:   List of EvalCase instances.
        scorer:  Callable(actual, expected[, input]) → float in [0, 1].
                 Defaults to ``contains``.
        name:    Human-readable name for the suite.
        threshold: Minimum score to count as a pass (default 1.0).
    """

    def __init__(
        self,
        cases: list[EvalCase],
        scorer: Scorer = contains,
        name: str = "eval",
        threshold: float = 1.0,
    ) -> None:
        self.cases = list(cases)
        self.scorer = scorer
        self.name = name
        self.threshold = threshold

    def _score(self, actual: str, expected: str, input_text: str) -> float:
        try:
            import inspect
            sig = inspect.signature(self.scorer)
            if len(sig.parameters) >= 3:
                return float(self.scorer(actual, expected, input_text))
            return float(self.scorer(actual, expected))
        except Exception:  # noqa: BLE001
            return 0.0

    def run(self, agent: Any, *, reset_memory: bool = True) -> EvalReport:
        """Run all cases synchronously and return an EvalReport."""
        results: list[EvalResult] = []
        for case in self.cases:
            if reset_memory:
                agent.memory.clear()
            t0 = time.monotonic()
            try:
                actual = agent.run(case.input)
                error = None
            except Exception as exc:  # noqa: BLE001
                actual = ""
                error = str(exc)
            duration_ms = (time.monotonic() - t0) * 1000
            score = self._score(actual, case.expected, case.input)
            results.append(
                EvalResult(
                    case=case,
                    actual=actual,
                    score=score,
                    passed=score >= self.threshold,
                    duration_ms=duration_ms,
                    error=error,
                )
            )
        return EvalReport(name=self.name, results=results)

    async def arun(self, agent: Any, *, reset_memory: bool = True) -> EvalReport:
        """Run all cases asynchronously and return an EvalReport."""
        results: list[EvalResult] = []
        for case in self.cases:
            if reset_memory:
                agent.memory.clear()
            t0 = time.monotonic()
            try:
                actual = await agent.arun(case.input)
                error = None
            except Exception as exc:  # noqa: BLE001
                actual = ""
                error = str(exc)
            duration_ms = (time.monotonic() - t0) * 1000
            score = self._score(actual, case.expected, case.input)
            results.append(
                EvalResult(
                    case=case,
                    actual=actual,
                    score=score,
                    passed=score >= self.threshold,
                    duration_ms=duration_ms,
                    error=error,
                )
            )
        return EvalReport(name=self.name, results=results)
