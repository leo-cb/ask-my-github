"""Compile the corrective RAG graph with a tool-agent fallback."""

from functools import partial

from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from ask_my_github.agentic.nodes import (
    generate_node,
    grade_node,
    retrieve_node,
    route_node,
    stats_node,
    tool_fallback_node,
    transform_query_node,
)
from ask_my_github.agentic.state import AgentState
from ask_my_github.agentic.tools import (
    get_repo_stats,
    list_repo_files,
    read_github_file,
    search_github_code,
)
from ask_my_github.logging_config import get_logger
from ask_my_github.rag.llm import get_agentic_chat_model
from ask_my_github.rag.retriever import build_retriever

logger = get_logger(__name__)

MAX_ITERATIONS = 3

_ROUTES = {
    "generate": "generate",
    "transform_query": "transform_query",
    "tool_fallback": "tool_fallback",
}

_CLASSIFICATION_ROUTES = {
    "stats": "stats",
    "code": "retrieve",
}


def build_agentic_graph(vector_store: FAISS) -> CompiledStateGraph:
    """Build the agentic RAG graph for the given vector store."""
    retriever = build_retriever(vector_store)
    llm = get_agentic_chat_model()
    tools = [search_github_code, read_github_file, list_repo_files, get_repo_stats]
    return _compile_graph(retriever, llm, tools)


def _compile_graph(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    tools: list[BaseTool],
) -> CompiledStateGraph:
    logger.info("Compiling agentic RAG graph with %d tools", len(tools))
    workflow = StateGraph(AgentState)

    workflow.add_node("route", partial(route_node, llm=llm))
    workflow.add_node("stats", stats_node)
    workflow.add_node("retrieve", partial(retrieve_node, retriever=retriever))
    workflow.add_node("grade", partial(grade_node, llm=llm))
    workflow.add_node("transform_query", partial(transform_query_node, llm=llm))
    workflow.add_node("generate", partial(generate_node, llm=llm))

    tool_agent = create_react_agent(llm, tools)
    workflow.add_node("tool_fallback", partial(tool_fallback_node, agent=tool_agent))

    workflow.add_edge(START, "route")
    workflow.add_conditional_edges(
        "route", _route_after_classification, _CLASSIFICATION_ROUTES
    )
    workflow.add_edge("stats", "generate")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", _route_after_grade, _ROUTES)
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("tool_fallback", END)

    return workflow.compile()


def _route_after_classification(state: AgentState) -> str:
    return state.get("route", "code")


def _route_after_grade(state: AgentState) -> str:
    if state.get("documents"):
        return "generate"
    if state.get("iteration", 0) < MAX_ITERATIONS:
        return "transform_query"
    return "tool_fallback"
