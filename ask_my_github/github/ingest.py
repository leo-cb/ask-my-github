"""Shared GitHub ingestion orchestration.

Both the FastAPI ingest endpoint and the Streamlit dashboard need the same
"load the persisted vector store, or scrape + build it" logic. This module
centralizes that so the two entry points stay consistent.
"""

from langchain_community.vectorstores import FAISS

from ask_my_github.github.loader import GitHubLoader
from ask_my_github.github.repo_stats import clear_repo_stats, save_repo_stats
from ask_my_github.logging_config import get_logger
from ask_my_github.rag.store import (
    build_vector_store,
    clear_vector_store,
    load_vector_store,
    save_vector_store,
)

logger = get_logger(__name__)


async def ingest_user(username: str) -> FAISS:
    """Load the persisted vector store for a user, or build and persist it.

    Returns the existing on-disk store when present so repeated launches are
    fast and require no GitHub API access. Otherwise it scrapes the user's
    repositories, saves the repo-stats table, builds the FAISS index, and
    persists both before returning the store.
    """
    vector_store = load_vector_store(username)
    if vector_store is not None:
        logger.info("Reusing persisted vector store for user '%s'", username)
        return vector_store

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


async def force_reindex(username: str) -> FAISS:
    """Delete and rebuild a user's vector store and repo-stats table.

    Used by the dashboard's Reindex action when the persisted data is stale
    (e.g. missing newly introduced author metadata). It clears both on-disk
    artifacts before delegating to the normal ingest flow.
    """
    logger.info("Forcing reindex for user '%s'", username)
    clear_vector_store(username)
    clear_repo_stats(username)
    return await ingest_user(username)
