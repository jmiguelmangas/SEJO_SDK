"""Semantic and exact-match response cache for SEJO SDK.

Wrap any ``ModelClient`` with ``CachedModel`` to avoid redundant API calls.

Two strategies:

* **Exact cache** (default) — caches by the verbatim prompt string.
  Zero dependencies, perfect for deterministic / repeated queries.
* **Semantic cache** — caches by cosine similarity of prompt embeddings.
  Requires an embedding function; returns a cached response when a new
  prompt is within ``threshold`` similarity of a stored one.

Example (exact cache)::

    from SEJO_SDK.cache import CachedModel

    model = CachedModel(OpenAIModel(...))
    agent = Agent(model=model)
    agent.run("What is LHR?")   # hits the API
    agent.run("What is LHR?")   # served from cache

Example (semantic cache with OpenAI embeddings)::

    import openai
    from SEJO_SDK.cache import CachedModel, openai_embedder

    embedder = openai_embedder(openai.OpenAI(api_key="..."))
    model = CachedModel(OpenAIModel(...), embedder=embedder, threshold=0.92)

Example (semantic cache with a custom embedder)::

    def my_embedder(text: str) -> list[float]:
        return some_embedding_model.encode(text).tolist()

    model = CachedModel(base_model, embedder=my_embedder, threshold=0.90)
"""

from __future__ import annotations

import math
from typing import Any, Callable

from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import ModelClient

EmbedFn = Callable[[str], list[float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def openai_embedder(client: Any, model: str = "text-embedding-3-small") -> EmbedFn:
    """Return an embedding function backed by the OpenAI embeddings API."""
    def _embed(text: str) -> list[float]:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    return _embed


# ---------------------------------------------------------------------------
# CachedModel
# ---------------------------------------------------------------------------

class CachedModel(ModelClient):
    """A ``ModelClient`` wrapper that short-circuits repeated prompts.

    Args:
        model:      The underlying ``ModelClient`` to cache.
        embedder:   Optional embedding function ``(str) → list[float]``.
                    If ``None``, exact string matching is used.
        threshold:  Cosine similarity threshold for semantic cache hits
                    (default ``0.95``).  Ignored when ``embedder`` is ``None``.
        max_size:   Maximum number of entries to keep (LRU-style drop of
                    the oldest when exceeded).  ``0`` = unlimited.
    """

    def __init__(
        self,
        model: ModelClient,
        embedder: EmbedFn | None = None,
        threshold: float = 0.95,
        max_size: int = 256,
    ) -> None:
        self._model = model
        self._embedder = embedder
        self._threshold = threshold
        self._max_size = max_size
        # exact cache: prompt → ModelResponse
        self._exact: dict[str, ModelResponse] = {}
        # semantic cache: list of (embedding, ModelResponse)
        self._semantic: list[tuple[list[float], ModelResponse]] = []

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._exact) + len(self._semantic)

    def clear(self) -> None:
        """Evict all cached entries."""
        self._exact.clear()
        self._semantic.clear()

    def _evict_if_needed(self) -> None:
        if self._max_size <= 0:
            return
        while len(self._exact) + len(self._semantic) >= self._max_size:
            if self._exact:
                self._exact.pop(next(iter(self._exact)))
            elif self._semantic:
                self._semantic.pop(0)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _lookup(self, prompt: str) -> ModelResponse | None:
        if prompt in self._exact:
            return self._exact[prompt]
        if self._embedder is not None:
            vec = self._embedder(prompt)
            best_score = -1.0
            best = None
            for stored_vec, stored_resp in self._semantic:
                score = _cosine(vec, stored_vec)
                if score > best_score:
                    best_score = score
                    best = stored_resp
            if best_score >= self._threshold:
                return best
        return None

    def _store(self, prompt: str, response: ModelResponse) -> None:
        self._evict_if_needed()
        if self._embedder is not None:
            vec = self._embedder(prompt)
            self._semantic.append((vec, response))
        else:
            self._exact[prompt] = response

    # ------------------------------------------------------------------
    # ModelClient interface
    # ------------------------------------------------------------------

    def _messages_to_key(self, messages: list[Message]) -> str:
        return "\n".join(f"{m.role}:{m.content}" for m in messages)

    def send_prompt(self, prompt: str, **kwargs: Any) -> ModelResponse:
        cached = self._lookup(prompt)
        if cached is not None:
            return cached
        response = self._model.send_prompt(prompt, **kwargs)
        self._store(prompt, response)
        return response

    def send_messages(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        key = self._messages_to_key(messages)
        cached = self._lookup(key)
        if cached is not None:
            return cached
        response = self._model.send_messages(messages, **kwargs)
        self._store(key, response)
        return response

    def stream_response(self, prompt: str, **kwargs: Any):
        # Streaming bypasses cache — delegate directly.
        yield from self._model.stream_response(prompt, **kwargs)

    def stream_messages(self, messages: list[Message], **kwargs: Any):
        yield from self._model.stream_messages(messages, **kwargs)
