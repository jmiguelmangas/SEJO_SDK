import asyncio

from SEJO_SDK.agent import Agent
from SEJO_SDK.model import AsyncModelClient


class AsyncEchoModel(AsyncModelClient):
    async def send_prompt(self, prompt: str, **kwargs) -> str:
        return f"Async echo: {prompt}"

    async def stream_response(self, prompt: str, **kwargs):
        yield "Async echo: "
        yield prompt


async def main():
    agent = Agent(model=AsyncEchoModel())

    print(await agent.arun("Hello async SEJO"))

    async for chunk in agent.astream("Stream this"):
        print(chunk, end="")


asyncio.run(main())
