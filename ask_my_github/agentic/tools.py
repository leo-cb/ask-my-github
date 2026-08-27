"""GitHub tools available to the agentic fallback."""

import base64

import httpx
from langchain_core.tools import tool

from ask_my_github.config import get_settings
from ask_my_github.logging_config import get_logger

logger = get_logger(__name__)

GITHUB_API_BASE = "https://api.github.com"
MAX_SEARCH_RESULTS = 10
MAX_LISTED_FILES = 200
MAX_FILE_CHARS = 8000


@tool
def search_github_code(query: str) -> str:
    """Search GitHub for code matching the query. Returns matching repo paths."""
    logger.info("Tool 'search_github_code': query='%s'", query)
    response = httpx.get(
        f"{GITHUB_API_BASE}/search/code",
        params={"q": query, "per_page": MAX_SEARCH_RESULTS},
        headers=_headers(),
    )
    if response.status_code != 200:
        return f"Code search failed with status {response.status_code}."
    items = response.json().get("items", [])
    if not items:
        return "No matching code found."
    return "\n".join(
        f"{item['repository']['full_name']} -> {item['path']}" for item in items
    )


@tool
def list_repo_files(owner: str, repo: str) -> str:
    """List file paths in a GitHub repository using the git tree API."""
    logger.info("Tool 'list_repo_files': owner='%s' repo='%s'", owner, repo)
    response = httpx.get(
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/HEAD",
        params={"recursive": "1"},
        headers=_headers(),
    )
    if response.status_code != 200:
        return f"Listing files failed with status {response.status_code}."
    tree = response.json().get("tree", [])
    paths = [entry["path"] for entry in tree if entry.get("type") == "blob"]
    return "\n".join(paths[:MAX_LISTED_FILES])


@tool
def read_github_file(owner: str, repo: str, path: str) -> str:
    """Read a single file from a GitHub repository using the contents API."""
    logger.info("Tool 'read_github_file': owner='%s' repo='%s' path='%s'", owner, repo, path)
    response = httpx.get(
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}",
        headers=_headers(),
    )
    if response.status_code != 200:
        return f"Reading file failed with status {response.status_code}."
    data = response.json()
    content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
    return content[:MAX_FILE_CHARS]


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = get_settings().github_token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
