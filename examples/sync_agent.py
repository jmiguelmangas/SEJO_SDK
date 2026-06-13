from SEJO_SDK.agent import Agent
from SEJO_SDK.model import ModelClient


class EchoModel(ModelClient):
    def send_prompt(self, prompt: str, **kwargs) -> str:
        return f"Echo: {prompt}"

    def stream_response(self, prompt: str, **kwargs):
        yield "Echo: "
        yield prompt


agent = Agent(model=EchoModel())

print(agent.run("Hello SEJO"))
