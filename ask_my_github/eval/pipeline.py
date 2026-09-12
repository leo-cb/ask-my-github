"""Thin adapter exposing the RAG answer paths to DeepEval.

DeepEval test cases need an ``actual_output`` (the generated answer) and a
``retrieval_context`` (the chunks actually fed to the LLM). This module wraps
the existing retrieval and generation code from ``api/query.py`` so the
evaluation reuses the exact same building blocks as production, instead of
reimplementing them.
"""

import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from ask_my_github.agentic.graph import build_agentic_graph
from ask_my_github.logging_config import get_logger
from ask_my_github.rag.llm import get_fast_chat_model
from ask_my_github.rag.prompt import QA_PROMPT
from ask_my_github.rag.retriever import build_retriever

logger = get_logger(__name__)


class RagPipeline:
    """Runs the fast or agentic answer path over a single vector store.

    The ``path`` only changes answer generation: both paths share the same
    FAISS retriever, so retrieval-only scoring is identical across them.
    """

    def __init__(
        self,
        vector_store: FAISS,
        path: str = "fast",
        temperature: float = 0.0,
        cache_path: Path | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.path = path
        self.temperature = temperature
        self.cache_path = cache_path
        self._cache = self._load_cache(cache_path) if cache_path else None

    def retrieve(self, question: str) -> list[str]:
        """Return the top-k chunks the retriever selects for a question."""
        documents = build_retriever(self.vector_store).invoke(question)
        return [document.page_content for document in documents]

    def answer(self, question: str) -> dict[str, Any]:
        """Return ``answer`` and ``retrieval_context`` for a question.

        The returned dict matches the keys DeepEval metrics expect, so callers
        can pass the values straight into an ``LLMTestCase``. When ``cache_path``
        is set, results are memoized on disk keyed by (path, question) so repeat
        runs skip the answer LLM call; only a fresh question spends tokens.
        """
        key = self._cache_key(question)
        if self._cache is not None and key in self._cache:
            return dict(self._cache[key])
        result = (
            self._answer_agentic(question)
            if self.path == "agentic"
            else self._answer_fast(question)
        )
        if self._cache is not None:
            self._cache[key] = result
            self._save_cache()
        return result

    def _cache_key(self, question: str) -> str:
        return f"{self.path}::{question}"

    @staticmethod
    def _load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        # Single-process JSON cache; fine for sequential pytest, not safe for
        # concurrent writers.
        assert self.cache_path is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _answer_fast(self, question: str) -> dict[str, Any]:
        documents = build_retriever(self.vector_store).invoke(question)
        context = "\n\n".join(document.page_content for document in documents)
        llm = get_fast_chat_model(temperature=self.temperature)
        answer = (QA_PROMPT | llm).invoke(
            {"context": context, "question": question}
        ).content
        return {
            "answer": answer,
            "retrieval_context": [document.page_content for document in documents],
        }

    def _answer_agentic(self, question: str) -> dict[str, Any]:
        graph = build_agentic_graph(self.vector_store, temperature=self.temperature)
        result = graph.invoke({"question": question})
        documents = result.get("documents", [])
        return {
            "answer": result.get("generation", ""),
            "retrieval_context": [document.page_content for document in documents],
        }
