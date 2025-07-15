from SEJO_SDK.model import Model_client
from openai import OpenAI
from typing import Any, Iterator

"""OpenAI model implementation."""

class OpenAIModel(Model_client):
    def __init__(self, api_key: str, model_name: str):
        """Initialize the OpenAI model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
    
    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the OpenAI model and return the response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )
        return response.choices[0].message.content or ""
    
    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the OpenAI model and return the response as a stream."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            **kwargs,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            yield content