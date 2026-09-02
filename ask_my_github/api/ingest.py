"""API endpoint for ingesting GitHub repositories."""

from fastapi import APIRouter, HTTPException
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from ask_my_github.github.ingest import ingest_user
from ask_my_github.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class GitHubIngestRequest(BaseModel):
    username: str


@router.post("/ingest/github")
async def ingest_github(request: GitHubIngestRequest) -> dict:
    logger.info("Ingest requested for user '%s'", request.username)
    try:
        vector_store = await ingest_user(request.username)
        _set_app_store(request.username, vector_store)
        logger.info("Ingest completed for user '%s'", request.username)
        return {"status": "success", "username": request.username}
    except Exception as e:
        logger.exception("Ingest failed for user '%s': %s", request.username, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _set_app_store(username: str, vector_store: FAISS) -> None:
    from ask_my_github.main import app

    app.state.vector_store = vector_store
    app.state.username = username
