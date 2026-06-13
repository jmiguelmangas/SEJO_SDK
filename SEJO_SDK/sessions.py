"""Session store for persistent agent conversations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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


# Default store alias for convenience.
SessionStore = InMemorySessionStore
