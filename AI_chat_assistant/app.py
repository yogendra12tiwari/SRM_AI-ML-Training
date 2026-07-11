
import streamlit as st
from datetime import datetime

from chatbot.engine import ChatEngine
from chatbot.stream import stream_text


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Load CSS
# -----------------------------
def load_css():
    try:
        with open("assets/styles.css") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        pass


load_css()


# -----------------------------
# Session State
# -----------------------------
if "engine" not in st.session_state:
    st.session_state.engine = ChatEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("🤖 AI Knowledge Assistant")

    st.success("🟢 Connected")

    st.divider()

    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.engine.reset()
        st.session_state.messages = []
        st.rerun()

    st.divider()

    user_messages = sum(
        1 for msg in st.session_state.messages
        if msg["role"] == "user"
    )

    ai_messages = sum(
        1 for msg in st.session_state.messages
        if msg["role"] == "assistant"
    )

    st.metric("User Messages", user_messages)
    st.metric("AI Messages", ai_messages)

    st.divider()

    st.info(
        """
Model

Llama 3.3

Provider

Groq
"""
    )


# -----------------------------
# Header
# -----------------------------
st.title("🤖 AI Knowledge Assistant")

st.caption("Powered by Groq + Llama 3")


# -----------------------------
# Welcome
# -----------------------------
if len(st.session_state.messages) == 0:

    st.markdown("""
# 👋 Welcome

You can ask me anything.


""")


# -----------------------------
# Display Chat History
# -----------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        st.caption(msg["time"])


# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input("Ask anything...")


# -----------------------------
# Chat Logic
# -----------------------------
if prompt:

    current_time = datetime.now().strftime("%H:%M")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "time": current_time,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(current_time)

    with st.spinner("Generating response..."):
        response = st.session_state.engine.chat(prompt)

    response_time = datetime.now().strftime("%H:%M")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "time": response_time,
        }
    )

    with st.chat_message("assistant"):
        st.write_stream(stream_text(response))
        st.caption(response_time)

