"""API endpoint for querying ingested GitHub data."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ask_my_github.agentic.graph import build_agentic_graph
from ask_my_github.config import get_settings
from ask_my_github.logging_config import get_logger
from ask_my_github.rag.llm import get_fast_chat_model
from ask_my_github.rag.prompt import QA_PROMPT
from ask_my_github.rag.retriever import build_retriever

logger = get_logger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
def query_github(request: QueryRequest) -> dict:
    vector_store = _get_vector_store()
    path = "fast" if get_settings().use_fast_rag else "agentic"
    logger.info("Query received (path=%s): %s", path, request.question)
    try:
        if get_settings().use_fast_rag:
            return _answer_fast(vector_store, request.question)
        return _answer_agentic(vector_store, request.question)
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _get_vector_store():
    from ask_my_github.main import app

    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No GitHub data ingested yet. Call /ingest/github first.",
        )
    return vector_store


def _answer_fast(vector_store, question: str) -> dict:
    logger.info("Retrieving documents for fast path")
    documents = build_retriever(vector_store).invoke(question)
    logger.info("Retrieved %d documents; generating answer", len(documents))
    llm = get_fast_chat_model()
    answer = (QA_PROMPT | llm).invoke(
        {"context": _join_documents(documents), "question": question}
    ).content
    return _format_response(question, answer, documents)


def _answer_agentic(vector_store, question: str) -> dict:
    logger.info("Building agentic graph")
    graph = build_agentic_graph(vector_store)
    result = graph.invoke({"question": question})
    logger.info("Agentic graph completed with %d source documents", len(result.get("documents", [])))
    return _format_response(
        question,
        result.get("generation", ""),
        result.get("documents", []),
    )


def _join_documents(documents) -> str:
    return "\n\n".join(document.page_content for document in documents)


def _format_response(question: str, answer: str, documents) -> dict:
    sources = [
        {
            "repo": document.metadata.get("repo"),
            "path": document.metadata.get("path"),
            "language": document.metadata.get("language"),
            "url": document.metadata.get("url"),
            "last_commit_date": document.metadata.get("last_commit_date"),
            "repo_pushed_at": document.metadata.get("repo_pushed_at"),
        }
        for document in documents
        if document.metadata.get("doc_type") != "repo_stats"
    ]
    return {"question": question, "answer": answer, "sources": sources}
