"""Agent hooks / middleware.

Hooks let you intercept every agent turn without modifying Agent itself.
Register callables for ``on_turn_start`` and ``on_turn_end`` events; they
run synchronously (or are awaited if async) around every ``run`` / ``arun``
call.

Example::

    from SEJO_SDK.hooks import HookRegistry

    hooks = HookRegistry()

    @hooks.on_turn_start
    def log_input(ctx):
        print(f"[turn {ctx['turn']}] → {ctx['input']!r}")

    @hooks.on_turn_end
    def log_output(ctx):
        ms = ctx['duration_ms']
        print(f"[turn {ctx['turn']}] ← {ctx['output']!r}  ({ms:.0f} ms)")

    agent = Agent(model=..., hooks=hooks)
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable

HookFn = Callable[[dict[str, Any]], None]


class HookRegistry:
    """Container for turn-start and turn-end hooks.

    Hooks receive a *context* dict with at least:

    * ``turn``     — 1-based turn counter (incremented by the registry)
    * ``agent``    — the ``Agent`` instance
    * ``input``    — user input string (both events)
    * ``output``   — model output string (``on_turn_end`` only)
    * ``duration_ms`` — wall-clock time for the turn (``on_turn_end`` only)
    * ``error``    — exception if the turn raised, else ``None`` (``on_turn_end``)

    Any extra keys the agent puts in are passed through unchanged.
    """

    def __init__(self) -> None:
        self._start_hooks: list[HookFn] = []
        self._end_hooks: list[HookFn] = []
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Registration helpers (decorator or direct call)
    # ------------------------------------------------------------------

    def on_turn_start(self, fn: HookFn) -> HookFn:
        """Register a hook that fires before each turn. Usable as a decorator."""
        self._start_hooks.append(fn)
        return fn

    def on_turn_end(self, fn: HookFn) -> HookFn:
        """Register a hook that fires after each turn. Usable as a decorator."""
        self._end_hooks.append(fn)
        return fn

    # ------------------------------------------------------------------
    # Internal helpers called by Agent
    # ------------------------------------------------------------------

    def _next_ctx(self, agent: Any, user_input: str) -> dict[str, Any]:
        self._turn_count += 1
        return {
            "turn": self._turn_count,
            "agent": agent,
            "input": user_input,
            "output": None,
            "duration_ms": 0.0,
            "error": None,
        }

    def fire_start(self, ctx: dict[str, Any]) -> None:
        for fn in self._start_hooks:
            fn(ctx)

    def fire_end(self, ctx: dict[str, Any]) -> None:
        for fn in self._end_hooks:
            fn(ctx)

    async def async_fire_start(self, ctx: dict[str, Any]) -> None:
        for fn in self._start_hooks:
            result = fn(ctx)
            if inspect.isawaitable(result):
                await result

    async def async_fire_end(self, ctx: dict[str, Any]) -> None:
        for fn in self._end_hooks:
            result = fn(ctx)
            if inspect.isawaitable(result):
                await result

    # ------------------------------------------------------------------
    # Context-manager helpers for timing (used by Agent internally)
    # ------------------------------------------------------------------

    def start_timer(self) -> float:
        return time.monotonic()

    def elapsed_ms(self, t0: float) -> float:
        return (time.monotonic() - t0) * 1000
