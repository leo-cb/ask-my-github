"""Nodes for the corrective RAG graph."""

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever

from ask_my_github.agentic.state import AgentState
from ask_my_github.rag.prompt import (
    GENERATE_SYSTEM_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
)


def retrieve_node(state: AgentState, retriever: BaseRetriever) -> dict:
    """Retrieve documents for the current query."""
    query = state.get("query") or state["question"]
    return {"query": query, "documents": retriever.invoke(query)}


def grade_node(state: AgentState, llm: BaseChatModel) -> dict:
    """Filter retrieved documents to those relevant to the question."""
    question = state["question"]
    relevant = [
        document
        for document in state.get("documents", [])
        if _is_relevant(llm, question, document)
    ]
    return {"documents": relevant}


def transform_query_node(state: AgentState, llm: BaseChatModel) -> dict:
    """Rewrite the query to improve retrieval."""
    response = llm.invoke(REWRITE_PROMPT.format(question=state["question"]))
    return {
        "query": response.content,
        "iteration": state.get("iteration", 0) + 1,
    }


def generate_node(state: AgentState, llm: BaseChatModel) -> dict:
    """Generate an answer from the relevant documents."""
    context = "\n\n".join(
        document.page_content for document in state.get("documents", [])
    )
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=_format_prompt(state["question"], context)),
    ]
    return {"generation": llm.invoke(messages).content}


def tool_fallback_node(state: AgentState, agent) -> dict:
    """Fall back to a ReAct tool agent when retrieval is insufficient."""
    result = agent.invoke({"messages": [("user", state["question"])]})
    final = result["messages"][-1]
    answer = final.content if isinstance(final.content, str) else str(final.content)
    return {"generation": answer}


def _is_relevant(llm: BaseChatModel, question: str, document: Document) -> bool:
    response = llm.invoke(
        GRADE_PROMPT.format(question=question, document=document.page_content)
    )
    return response.content.strip().lower().startswith("yes")


def _format_prompt(question: str, context: str) -> str:
    return f"Question: {question}\n\nContext:\n{context}"
