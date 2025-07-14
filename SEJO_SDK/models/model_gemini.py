from SEJO_SDK.model import Model_client
import google.generativeai as genai
from typing import Iterator, Any

"""Gemini model implementation."""

class GeminiModel(Model_client):
    def __init__(self, api_key: str, model_name: str):
        """Initialize the Gemini model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
    
    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the Gemini model and return the response."""
        response = genai.GenerativeModel(self.model_name).generate_content(prompt)
        return response.text
    
    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the Gemini model and return the response as a stream."""
        for chunk in genai.GenerativeModel(self.model_name).generate_content(prompt):
            yield chunk.text
