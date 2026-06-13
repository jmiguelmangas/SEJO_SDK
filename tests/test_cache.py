"""Tests for CachedModel (exact and semantic strategies)."""

from __future__ import annotations

from typing import Any

from SEJO_SDK.cache import CachedModel, _cosine
from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import ModelClient


class CountingModel(ModelClient):
    """Records how many times the underlying model was actually called."""

    def __init__(self, reply: str = "answer") -> None:
        self.reply = reply
        self.call_count = 0

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(content=self.reply)

    def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        self.call_count += 1
        last = msgs[-1].content if msgs else ""
        return ModelResponse(content=f"{self.reply}:{last}")

    def stream_response(self, prompt: str, **_: Any):
        self.call_count += 1
        yield self.reply

    def stream_messages(self, msgs: list[Message], **_: Any):
        self.call_count += 1
        yield self.reply


# ---------------------------------------------------------------------------
# Cosine helper
# ---------------------------------------------------------------------------

def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine(a, b)) < 1e-9


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# Exact cache
# ---------------------------------------------------------------------------

def test_exact_cache_hit():
    base = CountingModel()
    model = CachedModel(base)
    model.send_prompt("hello")
    model.send_prompt("hello")
    assert base.call_count == 1


def test_exact_cache_miss_different_prompt():
    base = CountingModel()
    model = CachedModel(base)
    model.send_prompt("hello")
    model.send_prompt("world")
    assert base.call_count == 2


def test_exact_cache_messages():
    base = CountingModel()
    model = CachedModel(base)
    msgs = [Message(role="user", content="test")]
    model.send_messages(msgs)
    model.send_messages(msgs)
    assert base.call_count == 1


def test_exact_cache_clear():
    base = CountingModel()
    model = CachedModel(base)
    model.send_prompt("hi")
    model.clear()
    model.send_prompt("hi")
    assert base.call_count == 2


def test_exact_cache_size():
    base = CountingModel()
    model = CachedModel(base)
    assert model.size == 0
    model.send_prompt("a")
    assert model.size == 1
    model.send_prompt("b")
    assert model.size == 2


def test_exact_cache_max_size_eviction():
    base = CountingModel()
    model = CachedModel(base, max_size=2)
    model.send_prompt("a")
    model.send_prompt("b")
    model.send_prompt("c")
    # max_size=2: one entry was evicted
    assert model.size <= 2


def test_stream_bypasses_cache():
    base = CountingModel()
    model = CachedModel(base)
    list(model.stream_response("hi"))
    list(model.stream_response("hi"))
    assert base.call_count == 2


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

def _make_embedder(mapping: dict[str, list[float]]):
    def embed(text: str) -> list[float]:
        return mapping.get(text, [0.0, 0.0, 1.0])
    return embed


def test_semantic_cache_hit():
    base = CountingModel()
    v = [1.0, 0.0, 0.0]
    embedder = _make_embedder({"q1": v, "q2": v})
    model = CachedModel(base, embedder=embedder, threshold=0.99)
    model.send_prompt("q1")
    model.send_prompt("q2")  # same vector → cache hit
    assert base.call_count == 1


def test_semantic_cache_miss():
    base = CountingModel()
    embedder = _make_embedder({
        "q1": [1.0, 0.0, 0.0],
        "q2": [0.0, 1.0, 0.0],  # orthogonal → miss
    })
    model = CachedModel(base, embedder=embedder, threshold=0.99)
    model.send_prompt("q1")
    model.send_prompt("q2")
    assert base.call_count == 2


def test_semantic_cache_threshold():
    base = CountingModel()
    import math
    # Two vectors at ~45° → cosine = 0.707
    v1 = [1.0, 0.0]
    v2 = [1.0, 1.0]
    norm = math.sqrt(2)
    v2_norm = [x / norm for x in v2]
    embedder = _make_embedder({"q1": v1, "q2": v2_norm})

    # High threshold → miss
    model = CachedModel(base, embedder=embedder, threshold=0.99)
    model.send_prompt("q1")
    model.send_prompt("q2")
    assert base.call_count == 2

    # Low threshold → hit
    base2 = CountingModel()
    model2 = CachedModel(base2, embedder=embedder, threshold=0.50)
    model2.send_prompt("q1")
    model2.send_prompt("q2")
    assert base2.call_count == 1
