"""Embedding model factory supporting local, cloud, and Ollama providers."""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from ask_my_github.config import Settings, get_settings


EMBEDDING_PROVIDERS = frozenset({"huggingface", "openai", "ollama"})


def get_embeddings() -> Embeddings:
    """Return the embedding model for the configured provider."""
    settings = get_settings()
    provider = settings.embedding_provider
    if not provider:
        raise ValueError(
            "EMBEDDING_PROVIDER is not set. Set it in .env to choose the "
            f"embedding provider: {', '.join(sorted(EMBEDDING_PROVIDERS))}"
        )
    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    if provider == "openai":
        return _openai_embeddings(settings)
    if provider == "ollama":
        return _ollama_embeddings(settings)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. "
        f"Expected one of: {', '.join(sorted(EMBEDDING_PROVIDERS))}"
    )


def _openai_embeddings(settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def _ollama_embeddings(settings: Settings) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
