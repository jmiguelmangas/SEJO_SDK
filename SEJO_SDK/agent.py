from SEJO_SDK.model import Model_client
from memory import Memory
from typing import Iterator, Dict, Any

"""Agent implementation."""

class Agent:
    def __init__(self, model: Model_client, memory: Memory = None, tools: dict = None):
        """Initialize the agent with model, memory and additional keyword arguments."""
        self.model = model
        self.memory = memory or Memory()
        self.tools = tools or {}
    
    def run(self, user_input: str) -> str:
        """Run the agent with a prompt and return the response."""
        self.memory._add_user_message(user_input)
        prompt = f"{self.memory.get_context()}\nUser: {user_input}\nAgent:"
        response = self.model.send_prompt(prompt)
        self.memory._add_ai_message(response)
        return response