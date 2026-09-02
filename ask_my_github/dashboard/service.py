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
TECH_SEARCH_K = 30
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

# Canonical display names for technologies the LLM may spell differently.
_TECH_ALIASES = {
    "sklearn": "scikit-learn",
    "pil": "Pillow",
    "langchain_core": "langchain-core",
    "langchain core": "langchain-core",
    "flaskform": "flask-wtf",
    "flask_wtf": "flask-wtf",
    "opencv": "opencv-python",
    "github actions": "GitHub Actions",
    "github-actions": "GitHub Actions",
    "docker compose": "Docker Compose",
    "docker-compose": "Docker Compose",
}

# Low-signal tokens that are not meaningful "technologies" to surface.
_IGNORED_TECH = frozenset({"pip", "uv", "git", "sqlite3", "docker hub"})


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
    documents = vector_store.similarity_search(
        query,
        k=k,
        filter={"repo": repo_name},
        fetch_k=vector_store.index.ntotal,
    )
    if not documents:
        return "No indexed content available for this repository."
    context = "\n\n".join(document.page_content for document in documents)
    return _invoke_text(llm, REPO_SUMMARY_PROMPT.format(repo_name=repo_name, context=context))


def extract_technologies(
    vector_store: FAISS,
    llm: BaseChatModel,
) -> dict[str, dict[str, list[str]]]:
    """Extract languages and libraries from each repository's source code.

    For every repository in the store, retrieves a representative sample of
    source-code chunks (excluding dependency manifests) and asks the LLM, one
    repository at a time, for the languages and third-party libraries actually
    imported or used there. The LLM also filters out low-level and
    standard-library modules. Each language is capped at
    ``MAX_TECH_PER_LANGUAGE`` libraries.
    """
    per_repo: dict[str, dict[str, list[str]]] = {}
    for repo_name in _distinct_repos(vector_store):
        libraries = _extract_repo_technologies(vector_store, llm, repo_name)
        if libraries:
            per_repo[repo_name] = {
                language: _dedupe_libraries(libs)[:MAX_TECH_PER_LANGUAGE]
                for language, libs in libraries.items()
            }
    return per_repo


def _distinct_repos(vector_store: FAISS) -> list[str]:
    """Return the sorted list of repository names present in the store."""
    repos = {
        document.metadata.get("repo")
        for document in vector_store.docstore._dict.values()
    }
    return sorted(repo for repo in repos if repo)


def _extract_repo_technologies(
    vector_store: FAISS,
    llm: BaseChatModel,
    repo_name: str,
) -> dict[str, list[str]]:
    """Return one repository's language→libraries mapping via the LLM."""
    query = (
        "import, require, or usage of third-party libraries, frameworks, "
        "packages, and modules in source code"
    )
    documents = vector_store.similarity_search(
        query,
        k=TECH_SEARCH_K,
        filter={"repo": repo_name},
        fetch_k=vector_store.index.ntotal,
    )
    code_docs = [
        document
        for document in documents
        if not _is_dependency_file(document.metadata.get("path"))
    ][:MAX_TECH_CHUNKS]
    if not code_docs:
        return {}
    context = "\n\n".join(
        f"--- {document.metadata.get('path', '?')} "
        f"({document.metadata.get('language') or 'unknown'}) ---\n"
        f"{document.page_content}"
        for document in code_docs
    )
    raw = _invoke_text(
        llm, TECHNOLOGIES_PROMPT.format(repo_name=repo_name, context=context)
    )
    return _parse_language_json(raw)


def aggregate_technologies(
    per_repo: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    """Merge per-repo technologies into a single language→libraries mapping.

    Preserves first-seen order while deduplicating libraries across repos, so
    more commonly seen libraries surface first. Each language stays capped at
    ``MAX_TECH_PER_LANGUAGE``.
    """
    by_language: dict[str, list[str]] = {}
    for languages in per_repo.values():
        for language, libraries in languages.items():
            by_language.setdefault(language, []).extend(libraries)
    return {
        language: _dedupe_libraries(libraries)[:MAX_TECH_PER_LANGUAGE]
        for language, libraries in by_language.items()
    }


def technology_frequencies(
    per_repo: dict[str, dict[str, list[str]]],
) -> list[tuple[str, int]]:
    """Count how many repositories use each technology.

    Returns ``(technology, count)`` tuples ordered from most to least used.
    A technology is counted once per repository even if it appears under
    multiple languages, so the count reflects repo-level adoption.
    """
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    for languages in per_repo.values():
        seen_in_repo: set[str] = set()
        for library in {library for libs in languages.values() for library in libs}:
            canonical = _canonical_tech(library)
            if canonical is None:
                continue
            key = canonical.lower()
            if key in seen_in_repo:
                continue
            seen_in_repo.add(key)
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, canonical)
    return sorted(
        ((display[key], counts[key]) for key in counts),
        key=lambda item: (-item[1], item[0].lower()),
    )


def format_technologies(languages: dict[str, list[str]]) -> str:
    """Render a repository's language→libraries mapping as a short string."""
    parts = [
        f"{language}: {', '.join(libraries)}"
        for language, libraries in languages.items()
        if libraries
    ]
    return " · ".join(parts) if parts else "—"


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


def _canonical_tech(name: str) -> str | None:
    """Return a technology's canonical display name, or None to drop it."""
    key = name.strip().lower()
    if key in _IGNORED_TECH:
        return None
    return _TECH_ALIASES.get(key, name.strip())


def _dedupe_libraries(libraries: list[str]) -> list[str]:
    """Canonicalize and de-duplicate a list of technology names, in order."""
    result: list[str] = []
    seen: set[str] = set()
    for library in libraries:
        canonical = _canonical_tech(library)
        if canonical is None:
            continue
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(canonical)
    return result


def _invoke_text(llm: BaseChatModel, text: str) -> str:
    """Invoke the chat model and return its text content as a string."""
    response = llm.invoke(text)
    content = response.content
    return content if isinstance(content, str) else str(content)


def _parse_language_json(raw: str) -> dict[str, list[str]]:
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
