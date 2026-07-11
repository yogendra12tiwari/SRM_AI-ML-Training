import streamlit as st
from datetime import datetime

from chatbot.engine import ChatEngine
from chatbot.stream import stream_text


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="AI Chat Assistant",
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
                unsafe_allow_html=True
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

    st.title("🤖 AI Chat Assistant")

    st.success("🟢 Connected")
    st.divider()

chat_mode = st.radio(
    "Select Chat Mode",
    [
        "🌐 AI Chat",
        "📄 PDF Chat"
    ],
    index=0
)

st.divider()
    

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_pdf:

    pdf_path = f"uploads/{uploaded_pdf.name}"

    with open(pdf_path, "wb") as f:

        f.write(uploaded_pdf.getbuffer())

    with st.spinner("Reading PDF..."):

        st.session_state.engine.load_pdf(pdf_path)

    st.success("PDF Loaded Successfully!")

    st.divider()

    if st.button("🆕 New Chat", use_container_width=True):

        st.session_state.engine.reset()

        st.session_state.messages = []

        st.rerun()

    st.divider()

    user_count = sum(
        1 for msg in st.session_state.messages
        if msg["role"] == "user"
    )

    ai_count = sum(
        1 for msg in st.session_state.messages
        if msg["role"] == "assistant"
    )

    st.metric("User", user_count)
    st.metric("Assistant", ai_count)

    st.divider()

    st.caption("Powered by Groq")


# -----------------------------
# Header
# -----------------------------

st.title("🤖 AI Chat Assistant")

st.caption("Built with Streamlit + Groq")


# -----------------------------
# Welcome
# -----------------------------

if len(st.session_state.messages) == 0:

    st.info(
        """
👋 Welcome!

Try asking:

How can I help You 

"""
    )


# -----------------------------
# Chat History
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        st.caption(msg["time"])


# -----------------------------
# Chat Input
# -----------------------------

prompt = st.chat_input("Message AI Assistant...")


# -----------------------------
# Chat Logic
# -----------------------------

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

        if chat_mode == "📄 PDF Chat":

            if st.session_state.engine.pdf_loaded:

                response = st.session_state.engine.chat(prompt)

            else:

                response = "⚠ Please upload a PDF first."

        else:

            pdf_state = st.session_state.engine.pdf_loaded

            st.session_state.engine.pdf_loaded = False

            response = st.session_state.engine.chat(prompt)

            st.session_state.engine.pdf_loaded = pdf_state

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