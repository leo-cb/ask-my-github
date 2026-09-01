"""Embedding model factory supporting FastEmbed (local) and OpenAI (cloud) providers."""

from typing import Any

from langchain_core.embeddings import Embeddings
from langsmith import traceable
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


def _summarize_doc_inputs(inputs: dict[str, Any]) -> dict[str, int]:
    """Summarize batch embedding inputs for LangSmith, omitting text bodies and `self`."""
    texts = inputs.get("texts", [])
    return {"num_texts": len(texts)}


def _summarize_query_inputs(inputs: dict[str, Any]) -> dict[str, str]:
    """Summarize query embedding input for LangSmith, omitting `self`."""
    return {"query": inputs.get("text", "")}


def _summarize_doc_outputs(vectors: list[list[float]]) -> dict[str, int]:
    """Summarize batch embedding output, avoiding logging raw vectors to LangSmith."""
    return {
        "num_vectors": len(vectors),
        "dimensions": len(vectors[0]) if vectors else 0,
    }


def _summarize_query_output(vector: list[float]) -> dict[str, int]:
    """Summarize query embedding output, avoiding logging the raw vector."""
    return {"dimensions": len(vector)}


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

    @traceable(
        run_type="embedding",
        name="fastembed_embed_documents",
        process_inputs=_summarize_doc_inputs,
        process_outputs=_summarize_doc_outputs,
    )
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return dense embeddings for a batch of documents."""
        return [embedding.tolist() for embedding in self._model.embed(texts)]

    @traceable(
        run_type="embedding",
        name="fastembed_embed_query",
        process_inputs=_summarize_query_inputs,
        process_outputs=_summarize_query_output,
    )
    def embed_query(self, text: str) -> list[float]:
        """Return the dense embedding for a single query."""
        return next(self._model.embed([text])).tolist()


def _openai_embeddings(settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    # Wrap the raw OpenAI client so every embeddings.create() call is traced
    # with token usage and latency, even outside a LangChain callback context.
    client = wrap_openai(OpenAI(api_key=settings.openai_api_key))
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        client=client,
    )
