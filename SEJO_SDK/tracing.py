"""Built-in tracing and cost tracking for agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from SEJO_SDK.messages import Usage

# Cost per million tokens (USD). Extend or override as needed.
# Prices approximate as of mid-2025.
_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # DeepSeek
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


@dataclass
class Turn:
    """A single agent turn (one user→model exchange)."""

    role: str
    input: str
    output: str
    tool_calls: list[str] = field(default_factory=list)
    usage: Usage | None = None
    duration_ms: float = 0.0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "input": self.input,
            "output": self.output,
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.usage:
            d["usage"] = {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        if self.cost_usd:
            d["cost_usd"] = round(self.cost_usd, 6)
        return d


@dataclass
class TraceSummary:
    """Aggregated trace metrics for the full session."""

    turns: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_duration_ms: float
    tools_called: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_tokens,
            },
            "cost_usd": round(self.total_cost_usd, 6),
            "duration_ms": round(self.total_duration_ms, 1),
            "tools_called": self.tools_called,
        }


class Tracer:
    """Records agent turns, token usage and estimated cost."""

    def __init__(
        self,
        model_name: str | None = None,
        prices: dict[str, dict[str, float]] | None = None,
    ):
        self.model_name = model_name
        self._prices = prices or _DEFAULT_PRICES
        self._turns: list[Turn] = []
        self._start: float | None = None

    # -- context manager for timing a turn --

    def start_turn(self) -> None:
        self._start = time.monotonic()

    def end_turn(
        self,
        role: str,
        input_text: str,
        output_text: str,
        tool_calls: list[str] | None = None,
        usage: Usage | None = None,
    ) -> Turn:
        duration_ms = (time.monotonic() - (self._start or time.monotonic())) * 1000
        cost = self._estimate_cost(usage)
        turn = Turn(
            role=role,
            input=input_text,
            output=output_text,
            tool_calls=tool_calls or [],
            usage=usage,
            duration_ms=duration_ms,
            cost_usd=cost,
        )
        self._turns.append(turn)
        self._start = None
        return turn

    # -- public API --

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def summary(self) -> TraceSummary:
        total_in = sum(t.usage.input_tokens for t in self._turns if t.usage)
        total_out = sum(t.usage.output_tokens for t in self._turns if t.usage)
        total_cost = sum(t.cost_usd for t in self._turns)
        total_ms = sum(t.duration_ms for t in self._turns)
        tools: list[str] = []
        for t in self._turns:
            for name in t.tool_calls:
                if name not in tools:
                    tools.append(name)
        return TraceSummary(
            turns=len(self._turns),
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_tokens=total_in + total_out,
            total_cost_usd=total_cost,
            total_duration_ms=total_ms,
            tools_called=tools,
        )

    def reset(self) -> None:
        self._turns.clear()

    def _estimate_cost(self, usage: Usage | None) -> float:
        if usage is None or self.model_name is None:
            return 0.0
        price = self._prices.get(self.model_name)
        if price is None:
            return 0.0
        return (
            usage.input_tokens * price["input"] / 1_000_000
            + usage.output_tokens * price["output"] / 1_000_000
        )
