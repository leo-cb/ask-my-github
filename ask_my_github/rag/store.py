"""FAISS vector store construction and persistence."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ask_my_github.config import get_settings
from ask_my_github.logging_config import get_logger
from ask_my_github.rag.embeddings import get_embeddings
from ask_my_github.rag.splitter import split_documents

logger = get_logger(__name__)


def build_vector_store(documents: list[Document]) -> FAISS:
    """Build a FAISS vector store from repository documents."""
    chunks = split_documents(documents)
    logger.info("Embedding %d chunks into FAISS", len(chunks))
    return FAISS.from_documents(chunks, get_embeddings())


def save_vector_store(vector_store: FAISS, username: str) -> None:
    """Persist the vector store to disk for the given username."""
    path = store_path(username)
    vector_store.save_local(str(path))
    logger.info("Saved vector store to '%s'", path)


def load_vector_store(username: str) -> FAISS | None:
    """Load a persisted vector store, or None if it does not exist."""
    path = store_path(username)
    if not path.exists():
        return None
    logger.info("Loading persisted vector store from '%s'", path)
    # Local FAISS indexes are pickled; this flag is required to load them.
    return FAISS.load_local(
        str(path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def store_path(username: str) -> Path:
    """Return the on-disk path for a user's vector store."""
    return Path(get_settings().faiss_dir) / username


def clear_vector_store(username: str) -> None:
    """Delete a user's persisted vector store so it can be rebuilt."""
    path = store_path(username)
    if path.exists():
        import shutil

        shutil.rmtree(path, ignore_errors=True)
        logger.info("Cleared vector store at '%s'", path)
