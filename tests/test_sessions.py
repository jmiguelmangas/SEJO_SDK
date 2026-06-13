"""Tests for session store."""

from SEJO_SDK.sessions import InMemorySessionStore


def test_create_session_generates_id():
    store = InMemorySessionStore()
    session = store.create()

    assert session.id
    assert len(store) == 1


def test_create_session_with_explicit_id():
    store = InMemorySessionStore()
    session = store.create(session_id="abc-123")

    assert session.id == "abc-123"


def test_get_returns_none_for_missing_session():
    store = InMemorySessionStore()

    assert store.get("nonexistent") is None


def test_get_or_create_creates_new_session():
    store = InMemorySessionStore()
    session = store.get_or_create("s1")

    assert session.id == "s1"
    assert len(store) == 1


def test_get_or_create_returns_existing_session():
    store = InMemorySessionStore()
    store.create(session_id="s1")
    store.get_or_create("s1").memory.add_user_message("Hello")

    session = store.get_or_create("s1")

    assert len(session.memory.history) == 1


def test_session_memory_persists_across_agent_runs():
    from SEJO_SDK.agent import Agent
    from SEJO_SDK.model import ModelClient

    class EchoModel(ModelClient):
        def send_prompt(self, prompt, **kwargs):
            return "echo"

        def send_messages(self, messages, **kwargs):
            return "echo"

        def stream_response(self, prompt, **kwargs):
            yield "echo"

    store = InMemorySessionStore()
    session = store.get_or_create("user-42")

    agent1 = Agent(model=EchoModel(), memory=session.memory)
    agent1.run("First message")

    agent2 = Agent(model=EchoModel(), memory=store.get("user-42").memory)
    agent2.run("Second message")

    assert len(store.get("user-42").memory.history) == 4  # 2 user + 2 assistant


def test_delete_session():
    store = InMemorySessionStore()
    store.create(session_id="to-delete")

    deleted = store.delete("to-delete")

    assert deleted is True
    assert store.get("to-delete") is None
    assert len(store) == 0


def test_delete_missing_session_returns_false():
    store = InMemorySessionStore()

    assert store.delete("ghost") is False


def test_list_sessions():
    store = InMemorySessionStore()
    store.create(session_id="a")
    store.create(session_id="b")

    sessions = store.list_sessions()

    ids = {s["id"] for s in sessions}
    assert ids == {"a", "b"}


def test_session_to_dict_includes_turn_count():
    store = InMemorySessionStore()
    session = store.create(session_id="s1")
    session.memory.add_user_message("hi")

    d = session.to_dict()

    assert d["id"] == "s1"
    assert d["turn_count"] == 1
    assert "created_at" in d
    assert "updated_at" in d
