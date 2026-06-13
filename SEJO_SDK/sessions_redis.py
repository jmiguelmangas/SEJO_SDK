"""Redis-backed session store for SEJO SDK.

Requires ``pip install sejo-sdk[redis]``.

Sessions are stored as JSON strings under keys ``sejo:session:{id}``.
An optional TTL lets sessions expire automatically.

Example::

    from SEJO_SDK.sessions_redis import RedisSessionStore

    store = RedisSessionStore("redis://localhost:6379", ttl=86400)
    session = store.get_or_create("user-42")
    agent = Agent(model=..., memory=session.memory)
    agent.run("Hello")
    store.save(session)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from SEJO_SDK.memory import Memory
from SEJO_SDK.sessions import Session

_KEY_PREFIX = "sejo:session:"


def _session_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _session_to_payload(session: Session) -> str:
    return json.dumps(
        {
            "id": session.id,
            "history": session.memory.history,
            "metadata": session.metadata,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        },
        ensure_ascii=False,
    )


def _payload_to_session(
    raw: str | bytes, max_memory_size: int
) -> Session:
    data = json.loads(raw)
    mem = Memory(max_size=max_memory_size)
    mem.history = data.get("history", [])
    return Session(
        id=data["id"],
        memory=mem,
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        metadata=data.get("metadata", {}),
    )


class RedisSessionStore:
    """Session store backed by Redis.

    Args:
        url:             Redis URL (e.g. ``"redis://localhost:6379"``).
        ttl:             Optional TTL in seconds. Sessions expire after this
                         long without a ``save()`` call.  ``None`` = no expiry.
        max_memory_size: Passed to ``Memory`` instances created by this store.
        _client_factory: Injectable factory for testing — called with no args,
                         must return an object with ``get/set/delete/keys/scan``
                         compatible with redis-py.
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
        if _client_factory is not None:
            self._redis = _client_factory()
        else:
            try:
                import redis

                self._redis = redis.from_url(url, decode_responses=True)
            except ImportError as exc:
                from SEJO_SDK.errors import ProviderDependencyError

                raise ProviderDependencyError(
                    "Install Redis support with `pip install sejo-sdk[redis]`."
                ) from exc

    def _write(self, session: Session) -> None:
        key = _session_key(session.id)
        payload = _session_to_payload(session)
        if self._ttl is not None:
            self._redis.setex(key, self._ttl, payload)
        else:
            self._redis.set(key, payload)

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
        self._write(session)
        return session

    def get(self, session_id: str) -> Session | None:
        raw = self._redis.get(_session_key(session_id))
        if raw is None:
            return None
        return _payload_to_session(raw, self._max_memory_size)

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
        """Persist the session's current state (and refresh TTL if set)."""
        session.touch()
        self._write(session)

    def delete(self, session_id: str) -> bool:
        deleted = self._redis.delete(_session_key(session_id))
        return bool(deleted)

    def list_sessions(self) -> list[dict[str, Any]]:
        pattern = f"{_KEY_PREFIX}*"
        keys = list(self._redis.scan_iter(pattern))
        sessions = []
        for key in keys:
            raw = self._redis.get(key)
            if raw:
                s = _payload_to_session(raw, self._max_memory_size)
                sessions.append(s.to_dict())
        return sorted(sessions, key=lambda s: s["updated_at"], reverse=True)

    def __len__(self) -> int:
        return len(list(self._redis.scan_iter(f"{_KEY_PREFIX}*")))
