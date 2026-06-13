from SEJO_SDK.agent import Agent
from SEJO_SDK.messages import ModelResponse, ToolCall
from SEJO_SDK.model import ModelClient
from SEJO_SDK.tools import Tool


class ToolCallingModel(ModelClient):
    def __init__(self):
        self.calls = 0

    def send_prompt(self, prompt: str, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="I will calculate that.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="add",
                        arguments={"left": 2, "right": 3},
                    )
                ],
            )
        return "2 + 3 is 5."

    def stream_response(self, prompt: str, **kwargs):
        yield self.send_prompt(prompt, **kwargs)


def add(left: int, right: int) -> int:
    return left + right


agent = Agent(
    model=ToolCallingModel(),
    tools=[
        Tool(
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
    ],
)

print(agent.run_with_tools("What is 2 + 3?"))
