"""Retriever logic for GitHub RAG."""

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever


def build_retriever(vector_store: FAISS, k: int = 3) -> BaseRetriever:
    """
    Build a retriever from a FAISS vector store.

    Args:
        vector_store: FAISS vector store
        k: number of relevant documents to retrieve

    Returns:
        LangChain retriever
    """
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )
