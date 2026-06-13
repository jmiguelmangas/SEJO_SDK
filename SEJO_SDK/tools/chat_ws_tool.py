"""FastAPI WebSocket helper for exposing an agent over chat."""

from SEJO_SDK.agent import Agent
from SEJO_SDK.errors import ProviderDependencyError


def create_chat_app(agent: Agent, path: str = "/chat"):
    """Create a FastAPI app that responds to WebSocket messages with an Agent."""

    try:
        from fastapi import FastAPI, WebSocket
    except ImportError as exc:
        raise ProviderDependencyError(
            "Install server support with `pip install sejo-sdk[server]`."
        ) from exc

    app = FastAPI(title="SEJO SDK Chat")

    @app.websocket(path)
    async def chat(websocket: WebSocket):
        await websocket.accept()
        while True:
            message = await websocket.receive_text()
            response = agent.run(message)
            await websocket.send_text(response)

    return app
