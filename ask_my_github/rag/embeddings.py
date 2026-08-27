"""Embedding model factory."""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from ask_my_github.config import get_settings


def get_embeddings() -> Embeddings:
    """Return the configured embedding model."""
    return HuggingFaceEmbeddings(model_name=get_settings().embedding_model)
