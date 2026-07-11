import streamlit as st
from datetime import datetime

from chatbot.engine import ChatEngine
from chatbot.stream import stream_text


# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------
# CSS
# ----------------------------

def load_css():

    try:

        with open("assets/styles.css") as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

    except FileNotFoundError:
        pass


load_css()


# ----------------------------
# Session State
# ----------------------------

if "engine" not in st.session_state:

    st.session_state.engine = ChatEngine()

if "messages" not in st.session_state:

    st.session_state.messages = []


# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.title("🤖 AI Assistant")

    st.success("🟢 Connected")

    st.divider()

    if st.button("🆕 New Chat"):

        st.session_state.engine.reset()

        st.session_state.messages = []

        st.rerun()

    st.divider()

    user_count = sum(
        1
        for m in st.session_state.messages
        if m["role"] == "user"
    )

    ai_count = sum(
        1
        for m in st.session_state.messages
        if m["role"] == "assistant"
    )

    st.metric("User Messages", user_count)

    st.metric("AI Messages", ai_count)

    st.divider()

    st.caption("Powered by Groq + Llama")


# ----------------------------
# Header
# ----------------------------

st.title("🤖 AI Chat Assistant")

st.caption("Your Personal AI Assistant")


# ----------------------------
# Welcome
# ----------------------------

if len(st.session_state.messages) == 0:

    st.markdown("""
## 👋 Welcome!

I'm your AI assistant.
                
    How can I  help you


""")


# ----------------------------
# Chat History
# ----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        st.caption(message["time"])


# ----------------------------
# User Input
# ----------------------------

prompt = st.chat_input("Message AI Assistant...")


# ----------------------------
# Chat Logic
# ----------------------------

if prompt:

    current_time = datetime.now().strftime("%H:%M")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "time": current_time
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

        st.caption(current_time)

    with st.spinner("🤖 Thinking..."):

        response = st.session_state.engine.chat(prompt)

    response_time = datetime.now().strftime("%H:%M")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "time": response_time
        }
    )

    with st.chat_message("assistant"):

        st.write_stream(stream_text(response))

        st.caption(response_time)