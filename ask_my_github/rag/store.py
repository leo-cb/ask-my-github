"""FAISS vector store construction and persistence."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ask_my_github.config import get_settings
from ask_my_github.rag.embeddings import get_embeddings
from ask_my_github.rag.splitter import split_documents


def build_vector_store(documents: list[Document]) -> FAISS:
    """Build a FAISS vector store from repository documents."""
    chunks = split_documents(documents)
    return FAISS.from_documents(chunks, get_embeddings())


def save_vector_store(vector_store: FAISS, username: str) -> None:
    """Persist the vector store to disk for the given username."""
    vector_store.save_local(str(store_path(username)))


def load_vector_store(username: str) -> FAISS | None:
    """Load a persisted vector store, or None if it does not exist."""
    path = store_path(username)
    if not path.exists():
        return None
    # Local FAISS indexes are pickled; this flag is required to load them.
    return FAISS.load_local(
        str(path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def store_path(username: str) -> Path:
    """Return the on-disk path for a user's vector store."""
    return Path(get_settings().faiss_dir) / username
