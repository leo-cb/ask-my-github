"""Module for building vector stores from repository documents."""

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def build_vector_store(repo_documents: list[dict[str,any]]) -> FAISS:
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
