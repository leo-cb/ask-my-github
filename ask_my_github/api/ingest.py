"""API endpoint for ingesting GitHub repositories."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ask_my_github.github.loader import GitHubLoader
from ask_my_github.rag.embeddings import build_vector_store

router = APIRouter()


class GitHubIngestRequest(BaseModel):
    username: str


@router.post("/ingest/github")
def ingest_github(request: GitHubIngestRequest):
    try:
        loader = GitHubLoader()
        repo_documents = loader.load_user_repos(request.username)

        if not repo_documents:
            raise ValueError("No repositories found for this user.")

        vector_store = build_vector_store(repo_documents)

        # TEMP: store in global app state (we'll improve this later)
        from ask_my_github.main import app
        app.state.vector_store = vector_store

        return {
            "status": "success",
            "repos_indexed": len(repo_documents)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
