"""Tests for PostgresSessionStore using a fake psycopg2 connection."""

from __future__ import annotations

from SEJO_SDK.sessions import PostgresSessionStore, Session

# ---------------------------------------------------------------------------
# Fake psycopg2 infrastructure
# ---------------------------------------------------------------------------

class FakeCursor:
    """Minimal cursor that executes SQL against an in-memory dict store."""

    def __init__(self, store: dict[str, list]) -> None:
        self._store = store
        self._results: list[tuple] = []
        self.rowcount = 0

    def execute(self, sql: str, params: tuple = ()) -> None:
        sql_upper = " ".join(sql.upper().split())
        if "CREATE TABLE" in sql_upper:
            return
        if sql_upper.startswith("INSERT INTO SEJO_SESSIONS"):
            if "ON CONFLICT" in sql_upper:
                # UPSERT
                sid = params[0]
                self._store[sid] = list(params)
                self.rowcount = 1
            else:
                sid = params[0]
                self._store[sid] = list(params)
                self.rowcount = 1
        elif sql_upper.startswith("SELECT") and "WHERE ID" in sql_upper:
            sid = params[0]
            row = self._store.get(sid)
            self._results = [tuple(row)] if row else []
        elif sql_upper.startswith("SELECT") and "COUNT" in sql_upper:
            self._results = [(len(self._store),)]
        elif sql_upper.startswith("SELECT"):
            self._results = [tuple(v) for v in self._store.values()]
        elif sql_upper.startswith("DELETE"):
            sid = params[0]
            if sid in self._store:
                del self._store[sid]
                self.rowcount = 1
            else:
                self.rowcount = 0

    def fetchone(self) -> tuple | None:
        return self._results[0] if self._results else None

    def fetchall(self) -> list[tuple]:
        return self._results

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class FakeConnection:
    """Fake psycopg2 connection backed by a shared dict."""

    def __init__(self, store: dict[str, list]) -> None:
        self._store = store

    def cursor(self):
        return FakeCursor(self._store)

    def commit(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _make_store() -> tuple[PostgresSessionStore, dict]:
    """Return a PostgresSessionStore wired to an in-memory fake."""
    db: dict[str, list] = {}

    def conn_factory() -> FakeConnection:
        return FakeConnection(db)

    store = PostgresSessionStore("fake://", _conn_factory=conn_factory)
    return store, db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_returns_session():
    store, _ = _make_store()
    session = store.create("user-1")
    assert session.id == "user-1"
    assert isinstance(session, Session)


def test_create_auto_generates_id():
    store, _ = _make_store()
    session = store.create()
    assert len(session.id) == 36  # UUID


def test_get_returns_created_session():
    store, _ = _make_store()
    store.create("s1", metadata={"user": "alice"})
    fetched = store.get("s1")
    assert fetched is not None
    assert fetched.id == "s1"


def test_get_missing_returns_none():
    store, _ = _make_store()
    assert store.get("does-not-exist") is None


def test_get_or_create_existing():
    store, _ = _make_store()
    store.create("s2")
    session = store.get_or_create("s2")
    assert session.id == "s2"


def test_get_or_create_new():
    store, _ = _make_store()
    session = store.get_or_create("s3")
    assert session.id == "s3"


def test_save_persists_history():
    store, db = _make_store()
    session = store.create("s4")
    session.memory.add_user_message("hello")
    session.memory.add_ai_message("world")
    store.save(session)

    # Re-fetch and verify history was stored
    fetched = store.get("s4")
    assert fetched is not None
    messages = fetched.memory.get_messages()
    assert len(messages) == 2
    assert messages[0].content == "hello"
    assert messages[1].content == "world"


def test_delete_removes_session():
    store, _ = _make_store()
    store.create("s5")
    assert store.delete("s5") is True
    assert store.get("s5") is None


def test_delete_missing_returns_false():
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


def test_agent_with_postgres_store():
    """Agent uses a session from the store; history persists after save."""
    from SEJO_SDK.agent import Agent
    from SEJO_SDK.messages import Message, ModelResponse
    from SEJO_SDK.model import ModelClient

    class EchoModel(ModelClient):
        def send_prompt(self, prompt: str, **_) -> ModelResponse:
            return ModelResponse(content=f"echo: {prompt}")

        def send_messages(self, messages: list[Message], **_) -> ModelResponse:
            last = messages[-1].content if messages else ""
            return ModelResponse(content=f"echo: {last}")

        def stream_response(self, prompt: str, **_):
            yield f"echo: {prompt}"

        def stream_messages(self, messages: list[Message], **_):
            yield "echo"

    store, _ = _make_store()
    session = store.create("agent-session")
    agent = Agent(model=EchoModel(), memory=session.memory)
    agent.run("hi there")
    store.save(session)

    restored = store.get("agent-session")
    assert restored is not None
    msgs = restored.memory.get_messages()
    assert any(m.content == "hi there" for m in msgs)
