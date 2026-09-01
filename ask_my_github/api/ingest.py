"""API endpoint for ingesting GitHub repositories."""

from fastapi import APIRouter, HTTPException
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from ask_my_github.github.loader import GitHubLoader
from ask_my_github.github.repo_stats import save_repo_stats
from ask_my_github.logging_config import get_logger
from ask_my_github.rag.store import (
    build_vector_store,
    load_vector_store,
    save_vector_store,
)

logger = get_logger(__name__)

router = APIRouter()


class GitHubIngestRequest(BaseModel):
    username: str


@router.post("/ingest/github")
async def ingest_github(request: GitHubIngestRequest) -> dict:
    logger.info("Ingest requested for user '%s'", request.username)
    try:
        vector_store = load_vector_store(request.username)
        if vector_store is None:
            vector_store = await _ingest_fresh(request.username)
        else:
            logger.info("Reusing persisted vector store for user '%s'", request.username)
        _set_app_store(request.username, vector_store)
        logger.info("Ingest completed for user '%s'", request.username)
        return {"status": "success", "username": request.username}
    except Exception as e:
        logger.exception("Ingest failed for user '%s': %s", request.username, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _ingest_fresh(username: str) -> FAISS:
    logger.info("No persisted store for '%s'; loading repositories", username)
    documents, repo_stats = await GitHubLoader().load_repos(username)
    if not documents:
        raise ValueError(f"No repositories found for user '{username}'.")
    save_repo_stats(username, repo_stats)
    logger.info("Loaded %d documents; building vector store", len(documents))
    vector_store = build_vector_store(documents)
    save_vector_store(vector_store, username)
    logger.info("Vector store built and saved for user '%s'", username)
    return vector_store


def _set_app_store(username: str, vector_store: FAISS) -> None:
    from ask_my_github.main import app

    app.state.vector_store = vector_store
    app.state.username = username
