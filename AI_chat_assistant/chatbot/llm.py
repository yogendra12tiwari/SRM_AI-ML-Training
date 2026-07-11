from groq import Groq

from config.settings import (
    GROQ_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
)


class LLM:

    def __init__(self):
        self.client = Groq(
            api_key=GROQ_API_KEY
        )

    def chat(self, messages):

        completion = self.client.chat.completions.create(

            model=MODEL_NAME,

            messages=messages,

            temperature=TEMPERATURE,

            max_tokens=MAX_TOKENS,
        )

        return completion.choices[0].message.content.strip()