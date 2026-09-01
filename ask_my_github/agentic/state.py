"""Shared state for the agentic RAG graph."""

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    question: str
    query: str
    route: str
    documents: list[Document]
    generation: str
    iteration: int
