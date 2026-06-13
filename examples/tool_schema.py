from SEJO_SDK.agent import Agent
from SEJO_SDK.model import ModelClient
from SEJO_SDK.tools import Tool


class EchoModel(ModelClient):
    def send_prompt(self, prompt: str, **kwargs) -> str:
        return "Use run_tool to execute local tools."

    def stream_response(self, prompt: str, **kwargs):
        yield self.send_prompt(prompt, **kwargs)


def add(left: int, right: int) -> int:
    return left + right


add_tool = Tool(
    name="add",
    description="Add two integers.",
    func=add,
    parameters={
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
    },
)

agent = Agent(model=EchoModel(), tools=[add_tool])

print(agent.tool_schemas())
print(agent.run_tool("add", left=2, right=3))
