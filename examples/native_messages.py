from SEJO_SDK.agent import Agent
from SEJO_SDK.messages import system_message, user_message
from SEJO_SDK.model import ModelClient


class NativeMessageModel(ModelClient):
    def __init__(self):
        self.messages = []

    def send_prompt(self, prompt: str, **kwargs):
        return f"Prompt fallback: {prompt}"

    def send_messages(self, messages, **kwargs):
        self.messages.append(messages)
        roles = ", ".join(message.role for message in messages)
        return f"Received native messages: {roles}"

    def stream_response(self, prompt: str, **kwargs):
        yield self.send_prompt(prompt, **kwargs)


model = NativeMessageModel()
agent = Agent(model=model, system_prompt="Be concise.")

print(agent.run("Hello"))
print(model.send_messages([system_message("Rules"), user_message("Direct call")]))
