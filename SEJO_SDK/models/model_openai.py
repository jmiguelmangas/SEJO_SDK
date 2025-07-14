from SEJO_SDK.model import Model
import openai

"""OpenAI model implementation."""

class OpenAIModel(Model):
    def __init__(self, api_key: str, model_name: str):
        """Initialize the OpenAI model with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = api_key
    
    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the OpenAI model and return the response."""
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )
        return response.choices[0].message.content
    
    def stream_response(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Send a prompt to the OpenAI model and return the response as a stream."""
        for chunk in openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            **kwargs,
        ):
            yield chunk.choices[0].delta.get("content", "")