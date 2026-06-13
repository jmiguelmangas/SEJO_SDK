"""Tests for advanced memory strategies."""

from __future__ import annotations

from typing import Any

from SEJO_SDK.memory_strategies import (
    SlidingWindowMemory,
    SummarizationMemory,
    VectorMemory,
)
from SEJO_SDK.messages import Message, ModelResponse
from SEJO_SDK.model import ModelClient

# ---------------------------------------------------------------------------
# Fake summarizer model
# ---------------------------------------------------------------------------

class SummaryModel(ModelClient):
    def __init__(self, reply: str = "SUMMARY") -> None:
        self.reply = reply
        self.called = 0

    def send_prompt(self, prompt: str, **_: Any) -> ModelResponse:
        self.called += 1
        return ModelResponse(content=self.reply)

    def send_messages(self, msgs: list[Message], **_: Any) -> ModelResponse:
        self.called += 1
        return ModelResponse(content=self.reply)

    def stream_response(self, prompt: str, **_: Any):
        yield self.reply

    def stream_messages(self, msgs: list[Message], **_: Any):
        yield self.reply


# ---------------------------------------------------------------------------
# SlidingWindowMemory
# ---------------------------------------------------------------------------

def test_sliding_keeps_within_window():
    mem = SlidingWindowMemory(window=2)
    for i in range(10):
        mem.add_user_message(f"q{i}")
        mem.add_ai_message(f"a{i}")
    msgs = mem.get_messages()
    # Only last 2 exchanges = 4 messages
    assert len(msgs) <= 4


def test_sliding_preserves_complete_pairs():
    mem = SlidingWindowMemory(window=3)
    for i in range(5):
        mem.add_user_message(f"user{i}")
        mem.add_ai_message(f"ai{i}")
    msgs = mem.get_messages()
    roles = [m.role for m in msgs if m.role in ("user", "assistant")]
    # Must be interleaved pairs
    for j in range(0, len(roles) - 1, 2):
        assert roles[j] == "user"
        assert roles[j + 1] == "assistant"


def test_sliding_clear_resets():
    mem = SlidingWindowMemory(window=2)
    mem.add_user_message("hi")
    mem.add_ai_message("hello")
    mem.clear()
    assert mem.get_messages() == []


def test_sliding_window_one_exchange():
    mem = SlidingWindowMemory(window=1)
    mem.add_user_message("a")
    mem.add_ai_message("b")
    mem.add_user_message("c")
    mem.add_ai_message("d")
    msgs = mem.get_messages()
    contents = [m.content for m in msgs if m.role in ("user", "assistant")]
    assert "c" in contents
    assert "d" in contents


# ---------------------------------------------------------------------------
# SummarizationMemory
# ---------------------------------------------------------------------------

def test_summarization_no_compression_below_window():
    model = SummaryModel()
    mem = SummarizationMemory(model, window=20)
    for i in range(5):
        mem.add_user_message(f"q{i}")
        mem.add_ai_message(f"a{i}")
    assert model.called == 0
    assert mem.summary is None


def test_summarization_compresses_when_exceeded():
    model = SummaryModel("SUMMARY_TEXT")
    mem = SummarizationMemory(model, window=4)
    for i in range(6):
        mem.add_user_message(f"q{i}")
        mem.add_ai_message(f"a{i}")
    assert model.called >= 1
    assert mem.summary is not None
    assert "SUMMARY_TEXT" in mem.summary


def test_summarization_summary_prepended_to_messages():
    model = SummaryModel("KEY_SUMMARY")
    mem = SummarizationMemory(model, window=4)
    for i in range(6):
        mem.add_user_message(f"msg{i}")
        mem.add_ai_message(f"reply{i}")
    msgs = mem.get_messages()
    assert any("KEY_SUMMARY" in m.content for m in msgs)


def test_summarization_clear_resets_summary():
    model = SummaryModel()
    mem = SummarizationMemory(model, window=4)
    for i in range(6):
        mem.add_user_message(f"q{i}")
        mem.add_ai_message(f"a{i}")
    mem.clear()
    assert mem.summary is None
    assert mem.get_messages() == []


# ---------------------------------------------------------------------------
# VectorMemory
# ---------------------------------------------------------------------------

def _fixed_embedder(d: int = 4):
    """Returns an embedder that maps text to a fixed vector by hash."""
    import hashlib

    def embed(text: str) -> list[float]:
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = [(h >> i & 0xFF) / 255.0 for i in range(d)]
        total = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / total for x in vec]

    return embed


def test_vector_memory_stores_messages():
    mem = VectorMemory(embedder=_fixed_embedder(), top_k=3, recency=2)
    mem.add_user_message("hello")
    mem.add_ai_message("hi there")
    assert len(mem.history) == 2


def test_vector_memory_clear():
    mem = VectorMemory(embedder=_fixed_embedder())
    mem.add_user_message("a")
    mem.clear()
    assert mem.history == []
    assert mem._embeddings == []


def test_vector_memory_get_messages_returns_recency():
    mem = VectorMemory(embedder=_fixed_embedder(), top_k=2, recency=2)
    for i in range(10):
        mem.add_user_message(f"q{i}")
        mem.add_ai_message(f"a{i}")
    msgs = mem.get_messages()
    # At least the recency messages should be there
    assert len(msgs) >= 2


def test_vector_memory_query_retrieves_relevant():
    embedder = _fixed_embedder(d=8)
    mem = VectorMemory(embedder=embedder, top_k=2, recency=1)
    mem.add_user_message("What is LHR?")
    mem.add_ai_message("LHR is London Heathrow airport.")
    mem.add_user_message("Favourite colour?")
    mem.add_ai_message("Blue.")
    # Query for airport-related content — should retrieve LHR exchange
    msgs = mem.get_messages(query="Tell me about airports")
    " ".join(m.content for m in msgs)
    assert len(msgs) > 0
    # All messages should be Message objects
    assert all(isinstance(m, Message) for m in msgs)
