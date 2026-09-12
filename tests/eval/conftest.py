"""Shared fixtures for the DeepEval RAG evaluation suite."""

from pathlib import Path

import pytest

from ask_my_github.config import get_settings
from ask_my_github.eval.dataset import load_goldens, resolve_eval_user
from ask_my_github.eval.judge import get_judge_model
from ask_my_github.eval.pipeline import RagPipeline
from ask_my_github.rag.store import load_vector_store


def _cache_path(path: str) -> Path:
    """Return the on-disk answer cache file for the given answer path."""
    settings = get_settings()
    return Path(settings.eval_cache_dir) / resolve_eval_user() / f"{path}.json"


@pytest.fixture(scope="session")
def judge():
    """The shared DeepSeek LLM judge used by every metric."""
    return get_judge_model()


@pytest.fixture(scope="session")
def vector_store():
    """The persisted FAISS index for the configured eval user."""
    username = resolve_eval_user()
    store = load_vector_store(username)
    if store is None:
        pytest.skip(f"No persisted index for '{username}'. Ingest it first.")
    return store


@pytest.fixture(scope="session")
def fast_pipeline(vector_store):
    """One-shot RAG pipeline (retriever + QA prompt), with cached answers."""
    return RagPipeline(vector_store, path="fast", cache_path=_cache_path("fast"))


@pytest.fixture(scope="session")
def agentic_pipeline(vector_store):
    """Corrective-RAG graph pipeline, with cached answers."""
    return RagPipeline(vector_store, path="agentic", cache_path=_cache_path("agentic"))


@pytest.fixture(scope="session")
def goldens():
    """Golden dataset shared by all evaluation tests."""
    data = load_goldens()
    if not data:
        pytest.skip("No goldens found. Run utils/generate_goldens.py or add goldens.json.")
    return data
