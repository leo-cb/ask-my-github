"""Service layer for the dashboard: summaries, technology extraction, rankings."""

import json
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel

from ask_my_github.config import get_settings, require_cloud_llm
from ask_my_github.dashboard.prompts import REPO_SUMMARY_PROMPT, TECHNOLOGIES_PROMPT
from ask_my_github.github.repo_stats import load_repo_stats
from ask_my_github.logging_config import get_logger
from ask_my_github.rag.llm import get_fast_chat_model

logger = get_logger(__name__)

SUMMARY_K = 8
TECH_SEARCH_K = 200
MAX_TECH_CHUNKS = 40
MAX_TECH_PER_LANGUAGE = 15

# Dependency/manifest files whose content reveals libraries and frameworks.
_DEPENDENCY_FILES = frozenset({
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "pipfile", "package.json", "cargo.toml", "go.mod", "go.sum",
    "makefile", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "gemfile", "composer.json", "build.gradle", "build.gradle.kts",
    "pom.xml", "project.clj", "stack.yaml", "mix.exs",
})


def get_cloud_llm() -> BaseChatModel:
    """Return the dashboard's chat model, enforcing a cloud provider."""
    require_cloud_llm(get_settings())
    return get_fast_chat_model()


def summarize_repo(
    vector_store: FAISS,
    llm: BaseChatModel,
    repo_name: str,
    k: int = SUMMARY_K,
) -> str:
    """Generate a short natural-language summary of one repository.

    Retrieves the top chunks belonging to the repository from FAISS (filtered
    by the ``repo`` metadata key) and asks the LLM to summarize them.
    """
    query = f"What does the repository {repo_name} do? Purpose and main features."
    documents = vector_store.similarity_search(query, k=k, filter={"repo": repo_name})
    if not documents:
        return "No indexed content available for this repository."
    context = "\n\n".join(document.page_content for document in documents)
    return _invoke_text(llm, REPO_SUMMARY_PROMPT.format(repo_name=repo_name, context=context))


def extract_technologies(
    vector_store: FAISS,
    llm: BaseChatModel,
    username: str | None = None,
) -> dict[str, list[str]]:
    """Extract languages and libraries from dependency/manifest chunks in FAISS.

    Searches the vector store for dependency-related content, keeps only the
    manifest-file chunks authored by ``username`` (so libraries from forks
    written by others are excluded), and asks the LLM to return a JSON mapping
    of language to libraries, capped at ``MAX_TECH_PER_LANGUAGE`` each.
    """
    query = "dependencies libraries frameworks packages requirements"
    documents = vector_store.similarity_search(query, k=TECH_SEARCH_K)
    manifest_docs = [
        document
        for document in documents
        if _is_dependency_file(document.metadata.get("path"))
        and _is_author_match(document, username)
    ][:MAX_TECH_CHUNKS]
    if not manifest_docs:
        logger.info("No author manifest chunks found; technology extraction skipped")
        return {}
    context = "\n\n".join(
        f"--- {document.metadata.get('repo', '?')} / "
        f"{document.metadata.get('path', '?')} ---\n{document.page_content}"
        for document in manifest_docs
    )
    raw = _invoke_text(llm, TECHNOLOGIES_PROMPT.format(context=context))
    technologies = _parse_json_object(raw)
    return {
        language: libraries[:MAX_TECH_PER_LANGUAGE]
        for language, libraries in technologies.items()
    }


def repo_stats_frame(username: str) -> pd.DataFrame:
    """Return the persisted repo-stats table for a user as a DataFrame."""
    rows = load_repo_stats(username)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in ("commit_count", "author_commit_count"):
        if column in frame.columns:
            frame[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(0)
                .astype(int)
            )
    for column in ("pushed_at", "author_pushed_at", "created_at", "updated_at"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def top_by_commits(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top ``n`` repositories by the user's authored commit count."""
    if frame.empty:
        return frame
    return frame.sort_values("author_commit_count", ascending=False).head(n)


def top_by_recent(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the ``n`` repositories most recently pushed to by the user."""
    if frame.empty:
        return frame
    return frame.sort_values(
        "author_pushed_at", ascending=False, na_position="last"
    ).head(n)


def format_date(value, missing: str = "—") -> str:
    """Format a datetime-like value as a date-only string (no time component)."""
    if value is None:
        return missing
    try:
        if pd.isna(value):
            return missing
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value[:10]
    try:
        return value.strftime("%Y-%m-%d")
    except (AttributeError, ValueError):
        return missing


def _is_dependency_file(path: str | None) -> bool:
    """Return True when a document path is a recognized dependency manifest."""
    if not path:
        return False
    return Path(path).name.lower() in _DEPENDENCY_FILES


def _is_author_match(document, username: str | None) -> bool:
    """Return True when a document was authored by ``username``.

    When no username filter is requested, all documents match. When the
    document's author is unknown, it is retained so dependency files without
    author metadata are not silently dropped.
    """
    if not username:
        return True
    author = document.metadata.get("author")
    if not author:
        return True
    return author.lower() == username.lower()


def _invoke_text(llm: BaseChatModel, text: str) -> str:
    """Invoke the chat model and return its text content as a string."""
    response = llm.invoke(text)
    content = response.content
    return content if isinstance(content, str) else str(content)


def _parse_json_object(raw: str) -> dict[str, list[str]]:
    """Parse a language→libraries JSON object from an LLM response.

    Tolerates markdown code fences and surrounding prose by extracting the
    first balanced JSON object, falling back to an empty mapping on failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        logger.warning("Could not parse technologies JSON from LLM response")
        return {}
    result: dict[str, list[str]] = {}
    for language, libraries in data.items():
        if isinstance(libraries, list):
            result[str(language)] = [str(item) for item in libraries]
        elif isinstance(libraries, str):
            result[str(language)] = [libraries]
        else:
            result[str(language)] = []
    return result
