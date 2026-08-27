"""Main API application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ask_my_github.api.ingest import router as ingest_router
from ask_my_github.api.query import router as query_router
from ask_my_github.config import configure_tracing, get_settings
from ask_my_github.logging_config import get_logger, setup_logging
from ask_my_github.rag.store import load_vector_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    configure_tracing(settings)
    logger.info("Ask My GitHub starting up")
    if settings.github_username:
        logger.info("Loading persisted vector store for user '%s'", settings.github_username)
        application.state.vector_store = load_vector_store(settings.github_username)
    yield
    logger.info("Ask My GitHub shutting down")


app = FastAPI(title="Ask My GitHub", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(query_router)
