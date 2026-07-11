from chatbot.engine import ChatEngine

engine = ChatEngine()

print("=" * 50)
print("🤖 AI Chat Assistant")
print("=" * 50)
print("Type 'exit' to quit")
print("Type 'reset' to clear conversation\n")

while True:

    user_input = input("You : ")

    if user_input.lower() == "exit":
        print("\nGoodbye 👋")
        break

    if user_input.lower() == "reset":

        engine.reset()

        print("\nConversation Cleared\n")

        continue

    response = engine.chat(user_input)

    print("\nAI :\n")

    print(response)

    print()