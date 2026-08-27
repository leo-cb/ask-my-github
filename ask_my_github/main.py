"""Main API application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ask_my_github.api.ingest import router as ingest_router
from ask_my_github.api.query import router as query_router
from ask_my_github.config import configure_tracing, get_settings
from ask_my_github.rag.store import load_vector_store


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    configure_tracing(settings)
    if settings.github_username:
        application.state.vector_store = load_vector_store(settings.github_username)
    yield


app = FastAPI(title="Ask My GitHub", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(query_router)
