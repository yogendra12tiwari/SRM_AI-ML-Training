from chatbot.llm import LLM
from config.prompts import SYSTEM_PROMPT


class ChatEngine:

    def __init__(self):

        self.llm = LLM()

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    def chat(self, user_message):

        self.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        try:
            response = self.llm.chat(self.messages)

        except Exception as e:
            response = f"❌ Error: {e}"

        self.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        return response

    def reset(self):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]