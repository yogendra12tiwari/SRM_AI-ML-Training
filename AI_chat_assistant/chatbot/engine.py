from chatbot.llm import LLM
from chatbot.rag import PDFChat
from config.prompts import SYSTEM_PROMPT


class ChatEngine:

    def __init__(self):

        self.llm = LLM()

        self.pdf = PDFChat()

        self.pdf_loaded = False

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    def chat(self, user_message):

        # -------------------------
        # RAG Mode
        # -------------------------

        if self.pdf_loaded:

            context, pages = self.pdf.search(user_message)

            

            prompt = f"""
You are answering from the uploaded PDF.

Context:

{context}

Question:

{user_message}

If the answer is not found inside the context,
say:

'I couldn't find that information in the uploaded PDF.'
"""

            temp_messages = [

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ]

            answer = self.llm.chat(temp_messages)

            if pages:

                answer += "\n\n---\n📄 **Sources:** "

                answer += ", ".join(
                    f"Page {page}"
                    for page in sorted(set(pages))
                )

            return answer

        # -------------------------
        # Normal Chat
        # -------------------------

        self.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        response = self.llm.chat(self.messages)

        self.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        return response

    def load_pdf(self, pdf_path):

        self.pdf.load_pdf(pdf_path)

        self.pdf_loaded = True

    def reset(self):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        self.pdf_loaded = False