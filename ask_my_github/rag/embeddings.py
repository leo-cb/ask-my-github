"""Embedding model factory supporting FastEmbed (local) and OpenAI (cloud) providers."""

from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import PrivateAttr

from ask_my_github.config import Settings, get_settings
from ask_my_github.logging_config import get_logger

logger = get_logger(__name__)


EMBEDDING_PROVIDERS = frozenset({"fastembed", "openai"})


def get_embeddings() -> Embeddings:
    """Return the embedding model for the configured provider."""
    settings = get_settings()
    provider = settings.embedding_provider
    if not provider:
        raise ValueError(
            "EMBEDDING_PROVIDER is not set. Set it in .env to choose the "
            f"embedding provider: {', '.join(sorted(EMBEDDING_PROVIDERS))}"
        )
    logger.info("Initializing embeddings: provider=%s model=%s", provider, settings.embedding_model)
    if provider == "fastembed":
        return FastEmbedAdapter(model_name=settings.embedding_model)
    if provider == "openai":
        return _openai_embeddings(settings)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. "
        f"Expected one of: {', '.join(sorted(EMBEDDING_PROVIDERS))}"
    )


class FastEmbedAdapter(Embeddings):
    """LangChain-compatible wrapper around fastembed's TextEmbedding.

    Runs ONNX-optimized models locally via ONNX Runtime, which is
    significantly faster than the sentence-transformers backend it replaces.
    """

    model_name: str
    cache_dir: str | None = None
    _model: Any = PrivateAttr()

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        super().__init__()
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.cache_dir = cache_dir
        logger.info("Loading FastEmbed model '%s' (first run may download)", model_name)
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        logger.info("FastEmbed model '%s' ready", model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return dense embeddings for a batch of documents."""
        return [embedding.tolist() for embedding in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Return the dense embedding for a single query."""
        return next(self._model.embed([text])).tolist()


def _openai_embeddings(settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
