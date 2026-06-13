from SEJO_SDK.agent import Agent
from SEJO_SDK.memory import Memory
from SEJO_SDK.model import ModelClient


class FakeModel(ModelClient):
    def __init__(self, response="ok"):
        self.response = response
        self.prompts = []

    def send_prompt(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        return self.response

    def stream_response(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        yield "o"
        yield "k"


def test_agent_run_builds_prompt_and_updates_memory():
    model = FakeModel(response="Hola")
    agent = Agent(model=model)

    response = agent.run("Que tal?")

    assert response == "Hola"
    assert model.prompts == ["user: Que tal?\nAgent:"]
    assert agent.memory.history == [
        {"role": "user", "content": "Que tal?"},
        {"role": "assistant", "content": "Hola"},
    ]


def test_agent_stream_persists_joined_response():
    model = FakeModel()
    agent = Agent(model=model)

    chunks = list(agent.stream("Ping"))

    assert chunks == ["o", "k"]
    assert agent.memory.history[-1] == {"role": "assistant", "content": "ok"}


def test_memory_respects_max_size():
    memory = Memory(max_size=2)

    memory.add_user_message("one")
    memory.add_ai_message("two")
    memory.add_user_message("three")

    assert memory.history == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
