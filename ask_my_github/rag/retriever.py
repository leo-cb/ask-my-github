"""Retriever construction."""

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from ask_my_github.config import get_settings


def build_retriever(vector_store: FAISS, k: int | None = None) -> BaseRetriever:
    """Build a retriever over the given vector store."""
    search_kwargs = {"k": k or get_settings().retriever_k}
    return vector_store.as_retriever(search_kwargs=search_kwargs)
