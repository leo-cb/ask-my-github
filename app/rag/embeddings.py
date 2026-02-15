"""Module for building vector stores from repository documents."""

from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


def build_vector_store(repo_documents: list[dict]) -> FAISS:
    """Build a FAISS vector store from repository documents. """

    documents = [
        Document(
            page_content=repo["content"],
            metadata=repo["metadata"],
        )
        for repo in repo_documents
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store
