"""Tests for AsyncPostgresSessionStore and AsyncRedisSessionStore."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from SEJO_SDK.sessions import Session
from SEJO_SDK.sessions_async import AsyncPostgresSessionStore, AsyncRedisSessionStore

# ---------------------------------------------------------------------------
# Fake asyncpg connection
# ---------------------------------------------------------------------------

class FakeAsyncCursor:
    def __init__(self, store: dict) -> None:
        self._store = store
        self._rows: list = []

    async def execute(self, sql: str, *args: Any) -> str:
        sql = sql.strip()
        if sql.startswith("CREATE TABLE"):
            return "CREATE TABLE"
        if sql.startswith("INSERT INTO sejo_sessions"):
            sid = args[0] if args else args
            history = json.loads(args[1]) if isinstance(args[1], str) else []
            metadata = json.loads(args[2]) if isinstance(args[2], str) else {}
            created_at = args[3]
            updated_at = args[4]
            self._store[sid] = {
                "id": sid,
                "history": history,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
            }
            return "INSERT 1"
        if sql.startswith("DELETE FROM sejo_sessions"):
            sid = args[0]
            existed = sid in self._store
            self._store.pop(sid, None)
            return "DELETE 1" if existed else "DELETE 0"
        return ""

    async def fetchrow(self, sql: str, *args: Any) -> dict | None:
        sid = args[0] if args else None
        return self._store.get(sid)

    async def fetch(self, sql: str, *args: Any) -> list[dict]:
        rows = list(self._store.values())
        return sorted(rows, key=lambda r: str(r.get("updated_at", "")), reverse=True)


class FakeAsyncConn:
    def __init__(self, store: dict) -> None:
        self._store = store
        self._cur = FakeAsyncCursor(store)

    async def execute(self, sql: str, *args: Any) -> str:
        return await self._cur.execute(sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> dict | None:
        return await self._cur.fetchrow(sql, *args)

    async def fetch(self, sql: str, *args: Any) -> list[dict]:
        return await self._cur.fetch(sql, *args)

    async def close(self) -> None:
        pass

    def __getitem__(self, key: str):
        return self._store.get(key)


def _make_pg_store() -> tuple[AsyncPostgresSessionStore, dict]:
    db: dict = {}

    async def factory():
        return FakeAsyncConn(db)

    store = AsyncPostgresSessionStore(dsn="fake://", _conn_factory=factory)
    return store, db


# ---------------------------------------------------------------------------
# AsyncPostgresSessionStore tests
# ---------------------------------------------------------------------------

def test_pg_create_returns_session():
    async def run():
        store, _ = _make_pg_store()
        s = await store.create("u1")
        assert s.id == "u1"
        assert isinstance(s, Session)

    asyncio.run(run())


def test_pg_create_auto_id():
    async def run():
        store, _ = _make_pg_store()
        s = await store.create()
        assert len(s.id) == 36

    asyncio.run(run())


def test_pg_get_existing():
    async def run():
        store, _ = _make_pg_store()
        await store.create("s1", metadata={"k": "v"})
        s = await store.get("s1")
        assert s is not None
        assert s.id == "s1"

    asyncio.run(run())


def test_pg_get_missing():
    async def run():
        store, _ = _make_pg_store()
        assert await store.get("ghost") is None

    asyncio.run(run())


def test_pg_get_or_create_new():
    async def run():
        store, _ = _make_pg_store()
        s = await store.get_or_create("new-session")
        assert s.id == "new-session"

    asyncio.run(run())


def test_pg_save_persists_history():
    async def run():
        store, _ = _make_pg_store()
        s = await store.create("s2")
        s.memory.add_user_message("hi")
        s.memory.add_ai_message("hello")
        await store.save(s)

        restored = await store.get("s2")
        assert restored is not None
        assert len(restored.memory.get_messages()) == 2

    asyncio.run(run())


def test_pg_delete_existing():
    async def run():
        store, _ = _make_pg_store()
        await store.create("s3")
        result = await store.delete("s3")
        assert result is True

    asyncio.run(run())


def test_pg_delete_missing():
    async def run():
        store, _ = _make_pg_store()
        result = await store.delete("ghost")
        assert result is False

    asyncio.run(run())


def test_pg_list_sessions():
    async def run():
        store, _ = _make_pg_store()
        await store.create("a")
        await store.create("b")
        listing = await store.list_sessions()
        ids = {s["id"] for s in listing}
        assert {"a", "b"} == ids

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Fake async redis
# ---------------------------------------------------------------------------

class FakeAsyncRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                deleted += 1
        return deleted

    async def scan_iter(self, pattern: str = "*"):
        prefix = pattern.rstrip("*")
        for k in list(self._store):
            if k.startswith(prefix):
                yield k

    async def aclose(self) -> None:
        pass


def _make_redis_store(
    ttl: int | None = None,
) -> tuple[AsyncRedisSessionStore, FakeAsyncRedis]:
    fake = FakeAsyncRedis()
    store = AsyncRedisSessionStore(_client_factory=lambda: fake, ttl=ttl)
    store._client = fake
    return store, fake


# ---------------------------------------------------------------------------
# AsyncRedisSessionStore tests
# ---------------------------------------------------------------------------

def test_redis_create_returns_session():
    async def run():
        store, _ = _make_redis_store()
        s = await store.create("u1")
        assert s.id == "u1"

    asyncio.run(run())


def test_redis_get_existing():
    async def run():
        store, _ = _make_redis_store()
        await store.create("s1", metadata={"env": "test"})
        s = await store.get("s1")
        assert s is not None
        assert s.metadata == {"env": "test"}

    asyncio.run(run())


def test_redis_get_missing():
    async def run():
        store, _ = _make_redis_store()
        assert await store.get("ghost") is None

    asyncio.run(run())


def test_redis_save_persists_history():
    async def run():
        store, _ = _make_redis_store()
        s = await store.create("s2")
        s.memory.add_user_message("ping")
        await store.save(s)

        restored = await store.get("s2")
        assert restored is not None
        assert restored.memory.get_messages()[0].content == "ping"

    asyncio.run(run())


def test_redis_delete():
    async def run():
        store, _ = _make_redis_store()
        await store.create("s3")
        assert await store.delete("s3") is True
        assert await store.get("s3") is None

    asyncio.run(run())


def test_redis_delete_missing():
    async def run():
        store, _ = _make_redis_store()
        assert await store.delete("ghost") is False

    asyncio.run(run())


def test_redis_list_sessions():
    async def run():
        store, _ = _make_redis_store()
        await store.create("a")
        await store.create("b")
        listing = await store.list_sessions()
        ids = {s["id"] for s in listing}
        assert {"a", "b"} == ids

    asyncio.run(run())


def test_redis_ttl_uses_setex():
    async def run():
        fake = FakeAsyncRedis()
        setex_calls: list[int] = []
        orig = fake.setex

        async def patched(k, t, v):
            setex_calls.append(t)
            await orig(k, t, v)

        fake.setex = patched
        store = AsyncRedisSessionStore(_client_factory=lambda: fake, ttl=3600)
        store._client = fake
        await store.create("ttl-session")
        assert setex_calls == [3600]

    asyncio.run(run())
