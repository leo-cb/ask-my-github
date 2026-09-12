"""Thin adapter exposing the RAG answer paths to DeepEval.

DeepEval test cases need an ``actual_output`` (the generated answer) and a
``retrieval_context`` (the chunks actually fed to the LLM). This module wraps
the existing retrieval and generation code from ``api/query.py`` so the
evaluation reuses the exact same building blocks as production, instead of
reimplementing them.
"""

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

    def __init__(self, vector_store: FAISS, path: str = "fast") -> None:
        self.vector_store = vector_store
        self.path = path

    def retrieve(self, question: str) -> list[str]:
        """Return the top-k chunks the retriever selects for a question."""
        documents = build_retriever(self.vector_store).invoke(question)
        return [document.page_content for document in documents]

    def answer(self, question: str) -> dict[str, Any]:
        """Return ``answer`` and ``retrieval_context`` for a question.

        The returned dict matches the keys DeepEval metrics expect, so callers
        can pass the values straight into an ``LLMTestCase``.
        """
        if self.path == "agentic":
            return self._answer_agentic(question)
        return self._answer_fast(question)

    def _answer_fast(self, question: str) -> dict[str, Any]:
        documents = build_retriever(self.vector_store).invoke(question)
        context = "\n\n".join(document.page_content for document in documents)
        llm = get_fast_chat_model()
        answer = (QA_PROMPT | llm).invoke(
            {"context": context, "question": question}
        ).content
        return {
            "answer": answer,
            "retrieval_context": [document.page_content for document in documents],
        }

    def _answer_agentic(self, question: str) -> dict[str, Any]:
        graph = build_agentic_graph(self.vector_store)
        result = graph.invoke({"question": question})
        documents = result.get("documents", [])
        return {
            "answer": result.get("generation", ""),
            "retrieval_context": [document.page_content for document in documents],
        }
