"""FastAPI app factory — expose any agent as a REST + WebSocket service."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from SEJO_SDK.errors import ProviderDependencyError

if TYPE_CHECKING:
    from SEJO_SDK.model import AsyncModelClient, ModelClient
    from SEJO_SDK.sessions import InMemorySessionStore
    from SEJO_SDK.tools import Tool
    from SEJO_SDK.tracing import Tracer


def create_agent_app(
    model: ModelClient | AsyncModelClient,
    tools: dict[str, Tool] | Iterable[Tool] | None = None,
    session_store: InMemorySessionStore | None = None,
    system_prompt: str | None = None,
    tracer: Tracer | None = None,
    title: str = "SEJO Agent",
) -> Any:
    """Return a FastAPI application that exposes the agent over HTTP and WebSocket.

    Endpoints:
        POST   /chat                         — single-turn (session_id optional)
        POST   /chat/{session_id}            — multi-turn with persistent memory
        GET    /sessions                     — list all sessions
        GET    /sessions/{session_id}        — session metadata + history
        DELETE /sessions/{session_id}        — delete a session
        GET    /trace                        — current tracer summary
        WS     /ws/{session_id}             — streaming via WebSocket
    """
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from pydantic import BaseModel
    except ImportError as exc:
        raise ProviderDependencyError(
            "Install server support with `pip install sejo-sdk[server]`."
        ) from exc

    from SEJO_SDK.agent import Agent
    from SEJO_SDK.sessions import InMemorySessionStore

    store = session_store or InMemorySessionStore()

    app = FastAPI(title=title)

    class ChatRequest(BaseModel):
        message: str
        session_id: str | None = None

    def _agent_for_session(session_id: str) -> Agent:
        session = store.get_or_create(session_id)
        return Agent(
            model=model,
            memory=session.memory,
            tools=tools,
            system_prompt=system_prompt,
            tracer=tracer,
        )

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        import uuid

        session_id = request.session_id or str(uuid.uuid4())
        agent = _agent_for_session(session_id)
        session = store.get_or_create(session_id)
        if agent.tools:
            response = agent.run_with_tools(request.message)
        else:
            response = agent.run(request.message)
        session.touch()
        result: dict[str, Any] = {
            "session_id": session_id,
            "response": response,
        }
        if tracer:
            turns = tracer.turns
            if turns:
                result["usage"] = turns[-1].to_dict().get("usage")
                result["cost_usd"] = turns[-1].cost_usd
        return result

    @app.post("/chat/{session_id}")
    def chat_session(session_id: str, request: ChatRequest) -> dict[str, Any]:
        request.session_id = session_id
        return chat(request)

    @app.get("/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return store.list_sessions()

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        data = session.to_dict()
        data["history"] = [m.to_dict() for m in session.memory.get_messages()]
        return data

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        deleted = store.delete(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": session_id}

    @app.get("/trace")
    def get_trace() -> dict[str, Any]:
        if tracer is None:
            return {"tracing": False}
        return tracer.summary().to_dict()

    @app.websocket("/ws/{session_id}")
    async def ws_chat(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            while True:
                user_input = await websocket.receive_text()
                session = store.get_or_create(session_id)
                from SEJO_SDK.model import AsyncModelClient as _Async

                if isinstance(model, _Async):
                    agent = Agent(
                        model=model,
                        memory=session.memory,
                        tools=tools,
                        system_prompt=system_prompt,
                        tracer=tracer,
                    )
                    chunks = []
                    async for chunk in agent.astream(user_input):
                        await websocket.send_text(chunk)
                        chunks.append(chunk)
                    session.touch()
                else:
                    agent = _agent_for_session(session_id)
                    for chunk in agent.stream(user_input):
                        await websocket.send_text(chunk)
                    session.touch()
        except WebSocketDisconnect:
            pass

    return app
