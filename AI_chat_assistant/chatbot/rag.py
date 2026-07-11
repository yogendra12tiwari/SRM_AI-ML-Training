from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class PDFChat:

    def __init__(self):
        self.vector_db = None

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_pdf(self, pdf_path):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        self.vector_db = FAISS.from_documents(
            chunks,
            self.embedding_model
        )

    def search(self, question, k=3):

        if self.vector_db is None:
            return "", []

        docs = self.vector_db.similarity_search(question, k=k)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        pages = []

        for doc in docs:

            page = doc.metadata.get("page")

            if page is not None:
                pages.append(page + 1)

        return context, pages