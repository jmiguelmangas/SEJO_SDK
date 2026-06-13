"""Async session stores for SEJO SDK.

Drop-in async counterparts to ``PostgresSessionStore`` and
``RedisSessionStore``.  Use these in async FastAPI apps or any
``asyncio``-based context where you need non-blocking I/O.

Example (asyncpg)::

    from SEJO_SDK.sessions_async import AsyncPostgresSessionStore

    store = AsyncPostgresSessionStore("postgresql://user:pass@localhost/db")
    await store.init()
    session = await store.get_or_create("user-42")
    agent = Agent(model=..., memory=session.memory)
    await agent.arun("Hello")
    await store.save(session)

Example (async redis)::

    from SEJO_SDK.sessions_async import AsyncRedisSessionStore

    store = AsyncRedisSessionStore("redis://localhost:6379", ttl=86400)
    session = await store.get_or_create("user-42")
    await store.save(session)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from SEJO_SDK.memory import Memory
from SEJO_SDK.sessions import Session

# ---------------------------------------------------------------------------
# Shared helpers (reuse serialisation from sessions_redis)
# ---------------------------------------------------------------------------

def _session_to_dict(session: Session) -> dict[str, Any]:
    return {
        "id": session.id,
        "history": session.memory.history,
        "metadata": session.metadata,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


def _dict_to_session(data: dict[str, Any], max_memory_size: int) -> Session:
    mem = Memory(max_size=max_memory_size)
    mem.history = data.get("history", [])
    return Session(
        id=data["id"],
        memory=mem,
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        metadata=data.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# AsyncPostgresSessionStore
# ---------------------------------------------------------------------------

_PG_DDL = """
    CREATE TABLE IF NOT EXISTS sejo_sessions (
        id          TEXT        PRIMARY KEY,
        history     JSONB       NOT NULL DEFAULT '[]',
        metadata    JSONB       NOT NULL DEFAULT '{}',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
"""


class AsyncPostgresSessionStore:
    """Async session store backed by PostgreSQL via asyncpg.

    Requires ``pip install asyncpg``.

    Call ``await store.init()`` once before use to create the table.
    """

    def __init__(
        self,
        dsn: str,
        max_memory_size: int = 100,
        _conn_factory: Callable | None = None,
    ) -> None:
        self._dsn = dsn
        self._max_memory_size = max_memory_size
        self._conn_factory = _conn_factory

    async def _connect(self):
        if self._conn_factory is not None:
            return await self._conn_factory()
        try:
            import asyncpg
            return await asyncpg.connect(self._dsn)
        except ImportError as exc:
            from SEJO_SDK.errors import ProviderDependencyError
            raise ProviderDependencyError(
                "Install asyncpg with `pip install asyncpg`."
            ) from exc

    async def init(self) -> None:
        """Create the sejo_sessions table if it doesn't exist."""
        conn = await self._connect()
        try:
            await conn.execute(_PG_DDL)
        finally:
            await conn.close()

    def _row_to_session(self, row: Any) -> Session:
        import json as _json
        raw_h = row["history"]
        history = raw_h if isinstance(raw_h, list) else _json.loads(raw_h)
        raw_m = row["metadata"]
        metadata = raw_m if isinstance(raw_m, dict) else _json.loads(raw_m)
        mem = Memory(max_size=self._max_memory_size)
        mem.history = history
        return Session(
            id=row["id"],
            memory=mem,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    async def create(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        sid = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        session = Session(
            id=sid,
            memory=Memory(max_size=self._max_memory_size),
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        conn = await self._connect()
        try:
            await conn.execute(
                "INSERT INTO sejo_sessions"
                " (id, history, metadata, created_at, updated_at)"
                " VALUES ($1, $2::jsonb, $3::jsonb, $4, $5)",
                sid, json.dumps([]), json.dumps({}), now, now,
            )
        finally:
            await conn.close()
        return session

    async def get(self, session_id: str) -> Session | None:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                "SELECT id, history, metadata, created_at, updated_at "
                "FROM sejo_sessions WHERE id = $1",
                session_id,
            )
        finally:
            await conn.close()
        return self._row_to_session(row) if row else None

    async def get_or_create(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = await self.get(session_id)
        if session is None:
            session = await self.create(session_id=session_id, metadata=metadata)
        return session

    async def save(self, session: Session) -> None:
        session.touch()
        conn = await self._connect()
        try:
            await conn.execute(
                "INSERT INTO sejo_sessions"
                " (id, history, metadata, created_at, updated_at)"
                " VALUES ($1, $2::jsonb, $3::jsonb, $4, $5)"
                " ON CONFLICT (id) DO UPDATE SET"
                "  history = EXCLUDED.history,"
                "  metadata = EXCLUDED.metadata,"
                "  updated_at = EXCLUDED.updated_at",
                session.id,
                json.dumps(session.memory.history),
                json.dumps(session.metadata),
                session.created_at,
                session.updated_at,
            )
        finally:
            await conn.close()

    async def delete(self, session_id: str) -> bool:
        conn = await self._connect()
        try:
            result = await conn.execute(
                "DELETE FROM sejo_sessions WHERE id = $1", session_id
            )
        finally:
            await conn.close()
        return result.endswith("1")

    async def list_sessions(self) -> list[dict[str, Any]]:
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                "SELECT id, history, metadata, created_at, updated_at "
                "FROM sejo_sessions ORDER BY updated_at DESC"
            )
        finally:
            await conn.close()
        return [self._row_to_session(r).to_dict() for r in rows]

    async def __alen__(self) -> int:
        conn = await self._connect()
        try:
            row = await conn.fetchrow("SELECT COUNT(*) AS n FROM sejo_sessions")
        finally:
            await conn.close()
        return row["n"]


# ---------------------------------------------------------------------------
# AsyncRedisSessionStore
# ---------------------------------------------------------------------------

_KEY_PREFIX = "sejo:session:"


def _rkey(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


class AsyncRedisSessionStore:
    """Async session store backed by Redis via redis.asyncio.

    Requires ``pip install sejo-sdk[redis]``.

    Example::

        store = AsyncRedisSessionStore("redis://localhost:6379", ttl=86400)
        session = await store.get_or_create("user-42")
        await store.save(session)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        ttl: int | None = None,
        max_memory_size: int = 100,
        _client_factory: Callable | None = None,
    ) -> None:
        self._url = url
        self._ttl = ttl
        self._max_memory_size = max_memory_size
        self._client_factory = _client_factory
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            # Factories may be sync (return client) or async (return awaitable)
            result = self._client_factory()
            if hasattr(result, "__await__"):
                self._client = await result
            else:
                self._client = result
            return self._client
        try:
            from redis.asyncio import from_url
            self._client = from_url(self._url, decode_responses=True)
            return self._client
        except ImportError as exc:
            from SEJO_SDK.errors import ProviderDependencyError
            raise ProviderDependencyError(
                "Install Redis support with `pip install sejo-sdk[redis]`."
            ) from exc

    async def _write(self, session: Session) -> None:
        client = await self._get_client()
        key = _rkey(session.id)
        payload = json.dumps(_session_to_dict(session), ensure_ascii=False)
        if self._ttl is not None:
            await client.setex(key, self._ttl, payload)
        else:
            await client.set(key, payload)

    async def create(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        sid = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        session = Session(
            id=sid,
            memory=Memory(max_size=self._max_memory_size),
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        await self._write(session)
        return session

    async def get(self, session_id: str) -> Session | None:
        client = await self._get_client()
        raw = await client.get(_rkey(session_id))
        if raw is None:
            return None
        return _dict_to_session(json.loads(raw), self._max_memory_size)

    async def get_or_create(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = await self.get(session_id)
        if session is None:
            session = await self.create(session_id=session_id, metadata=metadata)
        return session

    async def save(self, session: Session) -> None:
        session.touch()
        await self._write(session)

    async def delete(self, session_id: str) -> bool:
        client = await self._get_client()
        deleted = await client.delete(_rkey(session_id))
        return bool(deleted)

    async def list_sessions(self) -> list[dict[str, Any]]:
        client = await self._get_client()
        pattern = f"{_KEY_PREFIX}*"
        keys = [k async for k in client.scan_iter(pattern)]
        sessions = []
        for key in keys:
            raw = await client.get(key)
            if raw:
                s = _dict_to_session(json.loads(raw), self._max_memory_size)
                sessions.append(s.to_dict())
        return sorted(sessions, key=lambda s: s["updated_at"], reverse=True)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
