"""Retriever construction."""

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from ask_my_github.config import get_settings
from ask_my_github.logging_config import get_logger

logger = get_logger(__name__)


def build_retriever(vector_store: FAISS, k: int | None = None) -> BaseRetriever:
    """Build a retriever over the given vector store."""
    top_k = k or get_settings().retriever_k
    logger.info("Building retriever with k=%d", top_k)
    search_kwargs = {"k": top_k}
    return vector_store.as_retriever(search_kwargs=search_kwargs)
