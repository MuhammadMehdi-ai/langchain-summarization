"""
Retriever service.
"""

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from app.core.embeddings_factory import EmbeddingsFactory


class RetrieverService:
    """Handles document loading, splitting, and retrieval."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.embeddings = EmbeddingsFactory.create_embeddings()

        self._build()

    def _build(self):
        """Build vector store."""
        loader = TextLoader(self.file_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

    def retrieve(self, query: str) -> str:
        """Retrieve relevant text."""
        docs = self.retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])