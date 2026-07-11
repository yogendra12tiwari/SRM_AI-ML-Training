# 🤖 AI Knowledge Assistant

An intelligent AI-powered chatbot built with **Streamlit** and **Groq Llama 3**, featuring conversational memory and PDF-based question answering (RAG).

---

## 🚀 Features

* 💬 AI-powered chat using Groq Llama 3
* 🧠 Conversation memory during the session
* 📄 Upload a PDF and ask questions about its content
* ⚡ Streaming responses for a ChatGPT-like experience
* 🎨 Clean and responsive Streamlit interface
* 🔄 Start a new conversation with one click
* 📝 Markdown support for formatted responses

---


## 🛠️ Tech Stack

| Category              | Technology                        |
| --------------------- | --------------------------------- |
| Language              | Python                            |
| Frontend              | Streamlit                         |
| LLM                   | Groq (Llama 3)                    |
| AI Framework          | LangChain                         |
| Vector Store          | FAISS                             |
| Embeddings            | HuggingFace Sentence Transformers |
| PDF Processing        | PyPDF                             |
| Environment Variables | python-dotenv                     |

---

## 📂 Project Structure

```text
AI-Chat-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── chatbot/
│   ├── engine.py
│   ├── llm.py
│   ├── rag.py
│   └── stream.py
│
├── config/
│   ├── settings.py
│   └── prompts.py
│
├── assets/
│   └── styles.css
│
└── uploads/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yogendra12tiwari/SRM_AI-ML-Training.git
cd AI-Chat-Assistant
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## 💡 How It Works

### AI Chat Mode

1. User enters a prompt.
2. The prompt is sent to the Groq Llama 3 model.
3. The assistant generates a response.
4. Conversation history is maintained during the session.

### PDF Chat Mode

1. Upload a PDF document.
2. The PDF is split into smaller chunks.
3. Each chunk is converted into embeddings.
4. FAISS retrieves the most relevant chunks.
5. The retrieved context is sent to the LLM.
6. The assistant answers questions based on the uploaded document.

---

## 📌 Example Questions

### AI Chat

* Explain Machine Learning.
* Write Python code for Bubble Sort.
* Explain SQL JOIN with examples.
* What is Retrieval-Augmented Generation (RAG)?

### PDF Chat

* Summarize this document.
* What is the main objective?
* Explain Chapter 2.
* What conclusions are presented?

---

## 🎯 Learning Outcomes

This project demonstrates practical skills in:

* Large Language Model (LLM) integration
* Prompt engineering
* Retrieval-Augmented Generation (RAG)
* Semantic search with FAISS
* Streamlit application development
* Modular Python project architecture
* Environment variable management
* Building AI-powered user interfaces

---

## 🔮 Future Improvements

* Support multiple PDF documents
* Conversation export
* Chat history persistence
* Voice input and output
* Image understanding
* Web search integration

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Yogendra Tiwari**

Engineering Student | AI & Machine Learning Enthusiast

If you found this project useful, consider giving it a  on GitHub.
