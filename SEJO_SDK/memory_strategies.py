"""Advanced memory strategies for long conversations.

Drop-in replacements for ``Memory`` that keep conversation history
manageable without losing important context.

Strategies
----------
* ``SlidingWindowMemory``  — keeps the last N turns (fast, no model calls)
* ``SummarizationMemory``  — compresses old turns into a rolling summary
* ``VectorMemory``         — retrieves the K most relevant past turns by
                             embedding similarity (RAG over history)

All three share the ``Memory`` public API so they are fully interchangeable::

    from SEJO_SDK.memory_strategies import SummarizationMemory
    from SEJO_SDK.models import OpenAIModel

    model = OpenAIModel(api_key="...", model_name="gpt-4o-mini")
    memory = SummarizationMemory(model, window=6, max_summary_tokens=200)
    agent = Agent(model=model, memory=memory)
"""

from __future__ import annotations

import math
from typing import Any

from SEJO_SDK.memory import Memory
from SEJO_SDK.messages import Message

# ---------------------------------------------------------------------------
# SlidingWindowMemory
# ---------------------------------------------------------------------------

class SlidingWindowMemory(Memory):
    """Keep only the last ``window`` user+assistant exchanges.

    Unlike the base ``Memory`` (which drops messages one by one from the
    front), this strategy always keeps *complete* exchange pairs so the
    model never sees a half-conversation.

    Args:
        window: Number of complete exchanges (user+assistant pairs) to keep.
                Defaults to 10.
    """

    def __init__(self, window: int = 10) -> None:
        super().__init__(max_size=window * 2 + 10)
        self._window = window

    def _trim(self) -> None:
        role_messages = [
            m for m in self.history if m.get("role") in ("user", "assistant")
        ]
        if len(role_messages) > self._window * 2:
            # Find how many messages to drop from the front (full pairs only)
            drop_pairs = (len(role_messages) - self._window * 2) // 2
            drop_count = drop_pairs * 2
            dropped = 0
            new_history = []
            for m in self.history:
                if m.get("role") in ("user", "assistant") and dropped < drop_count:
                    dropped += 1
                    continue
                new_history.append(m)
            self.history = new_history

    def _add(self, message: Any) -> None:
        if isinstance(message, Message):
            message = message.to_dict()
        self.history.append(message)
        self._trim()


# ---------------------------------------------------------------------------
# SummarizationMemory
# ---------------------------------------------------------------------------

class SummarizationMemory(Memory):
    """Compress old turns into a rolling summary when history grows too long.

    When the number of messages exceeds ``window``, the oldest half is
    summarised by calling ``summarizer_model.send_messages(...)``.  The
    summary is injected as a synthetic user message at the start so the
    model always has full context.

    Args:
        summarizer_model: Any ``ModelClient`` used only for summarisation.
        window:           Message count before compression triggers.
        summary_prompt:   Override the default summarisation instruction.
    """

    _DEFAULT_PROMPT = (
        "Summarise the following conversation excerpt concisely, "
        "preserving all important facts, decisions and context:\n\n{history}"
    )

    def __init__(
        self,
        summarizer_model: Any,
        window: int = 20,
        summary_prompt: str = _DEFAULT_PROMPT,
    ) -> None:
        super().__init__(max_size=window + 20)
        self._model = summarizer_model
        self._window = window
        self._summary_prompt = summary_prompt
        self._summary: str | None = None

    def _maybe_summarize(self) -> None:
        if len(self.history) <= self._window:
            return
        half = len(self.history) // 2
        to_compress = self.history[:half]
        self.history = self.history[half:]

        excerpt = "\n".join(
            f"{m.get('role', '?').upper()}: {m.get('content', '')}"
            for m in to_compress
            if m.get("role") in ("user", "assistant")
        )
        prompt = self._summary_prompt.format(history=excerpt)
        try:
            resp = self._model.send_prompt(prompt)
            new_summary = resp.content if hasattr(resp, "content") else str(resp)
        except Exception:  # noqa: BLE001
            new_summary = excerpt[:500]

        if self._summary:
            self._summary = f"{self._summary}\n\n{new_summary}"
        else:
            self._summary = new_summary

    @property
    def summary(self) -> str | None:
        return self._summary

    def _add(self, message: Any) -> None:
        if isinstance(message, Message):
            message = message.to_dict()
        self.history.append(message)
        self._maybe_summarize()

    def get_messages(self) -> list[Message]:
        messages = super().get_messages()
        if self._summary:
            prefix = Message(
                role="user",
                content=f"[Conversation summary so far]\n{self._summary}",
            )
            return [prefix, *messages]
        return messages

    def clear(self) -> None:
        super().clear()
        self._summary = None


# ---------------------------------------------------------------------------
# VectorMemory
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorMemory(Memory):
    """Retrieve the K most relevant past turns by embedding similarity.

    All messages are stored indefinitely.  On ``get_messages()`` the
    ``top_k`` most similar past turns (plus the last ``recency`` messages)
    are returned, giving the model targeted long-term recall without
    exceeding context limits.

    Args:
        embedder:  Callable ``(str) → list[float]``.  Can be any embedding
                   function — the ``openai_embedder`` helper from
                   ``SEJO_SDK.cache`` works out of the box.
        top_k:     Maximum number of retrieved messages to include.
        recency:   Always include the last N messages regardless of score.
    """

    def __init__(
        self,
        embedder: Any,
        top_k: int = 5,
        recency: int = 6,
    ) -> None:
        super().__init__(max_size=999_999)
        self._embedder = embedder
        self._top_k = top_k
        self._recency = recency
        self._embeddings: list[tuple[list[float], int]] = []

    def _add(self, message: Any) -> None:
        if isinstance(message, Message):
            message = message.to_dict()
        idx = len(self.history)
        self.history.append(message)
        content = message.get("content", "")
        if content and message.get("role") in ("user", "assistant"):
            try:
                vec = self._embedder(content)
                self._embeddings.append((vec, idx))
            except Exception:  # noqa: BLE001
                pass

    def get_messages(self, query: str | None = None) -> list[Message]:
        if not self.history:
            return []

        recent_start = max(0, len(self.history) - self._recency)
        recent_idxs = set(range(recent_start, len(self.history)))

        if query and self._embeddings:
            try:
                q_vec = self._embedder(query)
                scored = [
                    (_cosine(q_vec, vec), idx)
                    for vec, idx in self._embeddings
                    if idx not in recent_idxs
                ]
                scored.sort(reverse=True)
                retrieved_idxs = {idx for _, idx in scored[: self._top_k]}
            except Exception:  # noqa: BLE001
                retrieved_idxs = set()
        else:
            retrieved_idxs = set()

        keep = sorted(retrieved_idxs | recent_idxs)
        selected = [self.history[i] for i in keep]
        return [Message.from_dict(m) for m in selected]

    def clear(self) -> None:
        super().clear()
        self._embeddings.clear()
