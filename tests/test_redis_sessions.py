"""Tests for RedisSessionStore using a fake redis client."""

from __future__ import annotations

from SEJO_SDK.sessions import Session
from SEJO_SDK.sessions_redis import RedisSessionStore

# ---------------------------------------------------------------------------
# Fake redis client
# ---------------------------------------------------------------------------

class FakeRedis:
    """Minimal in-memory fake for redis-py."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value  # ignore TTL in tests

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def delete(self, *keys: str) -> int:
        deleted = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                deleted += 1
        return deleted

    def scan_iter(self, pattern: str = "*"):
        prefix = pattern.rstrip("*")
        return [k for k in self._store if k.startswith(prefix)]


def _make_store(ttl: int | None = None) -> tuple[RedisSessionStore, FakeRedis]:
    fake = FakeRedis()
    store = RedisSessionStore(_client_factory=lambda: fake, ttl=ttl)
    return store, fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_returns_session():
    store, _ = _make_store()
    s = store.create("u1")
    assert s.id == "u1"
    assert isinstance(s, Session)


def test_create_auto_id():
    store, _ = _make_store()
    s = store.create()
    assert len(s.id) == 36


def test_get_existing():
    store, _ = _make_store()
    store.create("s1", metadata={"env": "test"})
    s = store.get("s1")
    assert s is not None
    assert s.id == "s1"
    assert s.metadata == {"env": "test"}


def test_get_missing():
    store, _ = _make_store()
    assert store.get("ghost") is None


def test_get_or_create_existing():
    store, _ = _make_store()
    store.create("s2")
    s = store.get_or_create("s2")
    assert s.id == "s2"


def test_get_or_create_new():
    store, _ = _make_store()
    s = store.get_or_create("s3")
    assert s.id == "s3"


def test_save_persists_history():
    store, _ = _make_store()
    s = store.create("s4")
    s.memory.add_user_message("hi")
    s.memory.add_ai_message("hello")
    store.save(s)

    restored = store.get("s4")
    assert restored is not None
    msgs = restored.memory.get_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "hi"
    assert msgs[1].content == "hello"


def test_delete():
    store, _ = _make_store()
    store.create("s5")
    assert store.delete("s5") is True
    assert store.get("s5") is None


def test_delete_missing():
    store, _ = _make_store()
    assert store.delete("ghost") is False


def test_list_sessions():
    store, _ = _make_store()
    store.create("a")
    store.create("b")
    listing = store.list_sessions()
    ids = {s["id"] for s in listing}
    assert {"a", "b"} == ids


def test_len():
    store, _ = _make_store()
    store.create("x")
    store.create("y")
    assert len(store) == 2


def test_ttl_uses_setex():
    fake = FakeRedis()
    store = RedisSessionStore(_client_factory=lambda: fake, ttl=3600)
    # Monkey-patch to detect setex calls
    setex_calls = []
    orig = fake.setex
    fake.setex = lambda k, t, v: (setex_calls.append(t), orig(k, t, v))
    store.create("ttl-session")
    assert setex_calls == [3600]


def test_agent_with_redis_store():
    from SEJO_SDK.agent import Agent
    from SEJO_SDK.messages import Message, ModelResponse
    from SEJO_SDK.model import ModelClient

    class EchoModel(ModelClient):
        def send_prompt(self, prompt: str, **_) -> ModelResponse:
            return ModelResponse(content=f"echo:{prompt}")

        def send_messages(self, msgs: list[Message], **_) -> ModelResponse:
            last = msgs[-1].content if msgs else ""
            return ModelResponse(content=f"echo:{last}")

        def stream_response(self, prompt: str, **_):
            yield f"echo:{prompt}"

        def stream_messages(self, msgs: list[Message], **_):
            yield "echo"

    store, _ = _make_store()
    session = store.create("agent-redis")
    agent = Agent(model=EchoModel(), memory=session.memory)
    agent.run("ping")
    store.save(session)

    restored = store.get("agent-redis")
    assert restored is not None
    assert any(m.content == "ping" for m in restored.memory.get_messages())
