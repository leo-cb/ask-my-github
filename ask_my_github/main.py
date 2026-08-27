"""Main API application file."""

from fastapi import FastAPI
from ask_my_github.api.ingest import router as ingest_router
from ask_my_github.api.query import router as query_router

app = FastAPI(title="Ask My GitHub")

app.include_router(ingest_router)
app.include_router(query_router)
