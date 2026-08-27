"""API endpoint for ingesting GitHub repositories."""

from fastapi import APIRouter, HTTPException
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from ask_my_github.github.loader import GitHubLoader
from ask_my_github.rag.store import (
    build_vector_store,
    load_vector_store,
    save_vector_store,
)

router = APIRouter()


class GitHubIngestRequest(BaseModel):
    username: str


@router.post("/ingest/github")
async def ingest_github(request: GitHubIngestRequest) -> dict:
    try:
        vector_store = load_vector_store(request.username)
        if vector_store is None:
            vector_store = await _ingest_fresh(request.username)
        _set_app_store(request.username, vector_store)
        return {"status": "success", "username": request.username}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _ingest_fresh(username: str) -> FAISS:
    documents = await GitHubLoader().load_repos(username)
    if not documents:
        raise ValueError(f"No repositories found for user '{username}'.")
    vector_store = build_vector_store(documents)
    save_vector_store(vector_store, username)
    return vector_store


def _set_app_store(username: str, vector_store: FAISS) -> None:
    from ask_my_github.main import app

    app.state.vector_store = vector_store
    app.state.username = username
