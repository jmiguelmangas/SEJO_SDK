"""Session store for persistent agent conversations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from SEJO_SDK.memory import Memory


@dataclass
class Session:
    """A named conversation session backed by a Memory instance."""

    id: str
    memory: Memory
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "turn_count": len(self.memory.history),
            "metadata": self.metadata,
        }


class InMemorySessionStore:
    """In-process session store.  Drop-in base for persistent backends."""

    def __init__(self, max_memory_size: int = 100):
        self._sessions: dict[str, Session] = {}
        self._max_memory_size = max_memory_size

    def create(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        sid = session_id or str(uuid.uuid4())
        session = Session(
            id=sid,
            memory=Memory(max_size=self._max_memory_size),
            metadata=metadata or {},
        )
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = self.get(session_id)
        if session is None:
            session = self.create(session_id=session_id, metadata=metadata)
        return session

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    def __len__(self) -> int:
        return len(self._sessions)


class PostgresSessionStore:
    """Session store backed by PostgreSQL.

    Requires ``pip install sejo-sdk[postgres]``.

    The ``sejo_sessions`` table is created automatically on first use.

    Example::

        store = PostgresSessionStore("postgresql://user:pass@localhost/mydb")
        session = store.get_or_create("user-42")
        agent = Agent(model=..., memory=session.memory)
        agent.run("Hello")
        store.save(session)   # persist the updated history
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS sejo_sessions (
            id          TEXT        PRIMARY KEY,
            history     JSONB       NOT NULL DEFAULT '[]',
            metadata    JSONB       NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """

    def __init__(
        self,
        dsn: str,
        max_memory_size: int = 100,
        _conn_factory: Callable | None = None,
    ) -> None:
        self._dsn = dsn
        self._max_memory_size = max_memory_size
        if _conn_factory is not None:
            self._make_conn = _conn_factory
        else:
            try:
                import psycopg2

                self._psycopg2 = psycopg2
                self._Json = psycopg2.extras.Json
            except ImportError as exc:
                from SEJO_SDK.errors import ProviderDependencyError

                raise ProviderDependencyError(
                    "Install Postgres support with `pip install sejo-sdk[postgres]`."
                ) from exc
            self._make_conn = lambda: psycopg2.connect(dsn)
        self._init_table()

    def _init_table(self) -> None:
        with self._make_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(self._DDL)
            conn.commit()

    def _row_to_session(self, row: tuple) -> Session:
        sid, history, metadata, created_at, updated_at = row
        mem = Memory(max_size=self._max_memory_size)
        mem.history = list(history)
        return Session(
            id=sid,
            memory=mem,
            created_at=created_at,
            updated_at=updated_at,
            metadata=dict(metadata),
        )

    def _serialize(self, value: Any) -> Any:
        """Wrap value for JSONB; uses psycopg2.extras.Json if available."""
        if hasattr(self, "_Json"):
            return self._Json(value)
        return value  # test double passes plain Python objects

    def create(
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
        with self._make_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sejo_sessions
                        (id, history, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (sid, self._serialize([]), self._serialize({}), now, now),
                )
            conn.commit()
        return session

    def get(self, session_id: str) -> Session | None:
        with self._make_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, history, metadata, created_at, updated_at "
                    "FROM sejo_sessions WHERE id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
        return self._row_to_session(row) if row else None

    def get_or_create(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = self.get(session_id)
        if session is None:
            session = self.create(session_id=session_id, metadata=metadata)
        return session

    def save(self, session: Session) -> None:
        """Persist the session's current memory state back to the database."""
        session.touch()
        with self._make_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sejo_sessions
                        (id, history, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        history    = EXCLUDED.history,
                        metadata   = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        session.id,
                        self._serialize(session.memory.history),
                        self._serialize(session.metadata),
                        session.created_at,
                        session.updated_at,
                    ),
                )
            conn.commit()

    def delete(self, session_id: str) -> bool:
        with self._make_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM sejo_sessions WHERE id = %s", (session_id,)
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._make_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, history, metadata, created_at, updated_at "
                    "FROM sejo_sessions ORDER BY updated_at DESC"
                )
                rows = cur.fetchall()
        return [self._row_to_session(r).to_dict() for r in rows]

    def __len__(self) -> int:
        with self._make_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM sejo_sessions")
                (count,) = cur.fetchone()
        return count


# Default store alias for convenience.
SessionStore = InMemorySessionStore
