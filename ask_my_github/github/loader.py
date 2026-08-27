"""Asynchronous GitHub repository loader."""

import asyncio
from pathlib import Path
from urllib.parse import quote

import httpx
from langchain_core.documents import Document

from ask_my_github.config import get_settings
from ask_my_github.logging_config import get_logger

logger = get_logger(__name__)

GITHUB_API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"

RATE_LIMIT_FLOOR = 50
RATE_LIMIT_BACKOFF_SECONDS = 60.0
REPOS_PER_PAGE = 100

_HIGH_SIGNAL_FILES = frozenset({
    "readme", "readme.md", "readme.txt", "readme.rst",
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "cargo.toml", "go.mod", "makefile",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "changelog", "changelog.md", "license", "license.md", "license.txt",
})

_EXCLUDED_DIRS = frozenset({
    ".git", "node_modules", "vendor", "dist", "build", "out",
    "__pycache__", ".venv", "venv", ".next", "target", ".gradle",
    ".idea", ".vscode", "coverage", ".pytest_cache", ".mypy_cache",
})

_EXCLUDED_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".pdf",
    ".zip", ".tar", ".gz", ".tgz", ".whl", ".so", ".dll", ".exe", ".bin",
    ".pyc", ".pyo", ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4",
    ".mov", ".avi", ".lock", ".min.js", ".min.css", ".map",
})

_EXCLUDED_FILENAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "cargo.lock", "go.sum", "uv.lock", "pipfile.lock", "gemfile.lock",
})

_LANGUAGE_KEY_BY_EXTENSION = {
    ".py": "python", ".pyi": "python",
    ".js": "js", ".mjs": "js", ".cjs": "js", ".jsx": "js",
    ".ts": "ts", ".tsx": "ts",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cxx": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".md": "markdown",
    ".rst": "rst",
    ".html": "html", ".htm": "html",
    ".sol": "sol",
    ".lua": "lua",
    ".proto": "proto",
    ".tex": "latex",
    ".cbl": "cobol", ".cob": "cobol",
    ".pl": "perl", ".pm": "perl",
    ".hs": "haskell",
    ".ex": "elixir", ".exs": "elixir",
    ".ps1": "powershell",
}


class GitHubLoader:
    """Loads repository files for a GitHub user using async HTTP."""

    def __init__(self, token: str | None = None):
        self._token = token or get_settings().github_token

    async def load_repos(self, username: str) -> list[Document]:
        """Load all indexable files across a user's repositories."""
        async with self._build_client() as client:
            repos = await self._list_user_repos(client, username)
            logger.info("Found %d repositories for user '%s'", len(repos), username)
            semaphore = asyncio.Semaphore(get_settings().max_concurrency)
            results = await asyncio.gather(
                *(self._load_repo(client, semaphore, username, repo) for repo in repos)
            )
        documents = [document for documents in results for document in documents]
        logger.info("Loaded %d documents total for user '%s'", len(documents), username)
        return documents

    def _build_client(self) -> httpx.AsyncClient:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return httpx.AsyncClient(headers=headers, timeout=30.0, follow_redirects=True)

    async def _list_user_repos(self, client: httpx.AsyncClient, username: str) -> list[dict]:
        repos: list[dict] = []
        page = 1
        while True:
            response = await client.get(
                f"{GITHUB_API_BASE}/users/{username}/repos",
                params={"per_page": REPOS_PER_PAGE, "page": page},
            )
            response.raise_for_status()
            await self._respect_rate_limit(response)
            batch = response.json()
            if not batch:
                break
            repos.extend(batch)
            if len(batch) < REPOS_PER_PAGE:
                break
            page += 1
        return repos

    async def _load_repo(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        username: str,
        repo: dict,
    ) -> list[Document]:
        repo_name = repo["name"]
        paths = await self._list_repo_files(client, semaphore, username, repo)
        logger.info("Repository '%s': %d indexable files", repo_name, len(paths))
        documents = await asyncio.gather(
            *(self._fetch_file(client, semaphore, username, repo, path) for path in paths)
        )
        loaded = [document for document in documents if document is not None]
        logger.info("Repository '%s': fetched %d/%d files", repo_name, len(loaded), len(paths))
        return loaded

    async def _list_repo_files(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        username: str,
        repo: dict,
    ) -> list[str]:
        branch = repo.get("default_branch") or "main"
        url = f"{GITHUB_API_BASE}/repos/{username}/{repo['name']}/git/trees/{branch}"
        async with semaphore:
            response = await client.get(url, params={"recursive": "1"})
        response.raise_for_status()
        await self._respect_rate_limit(response)
        tree = response.json().get("tree", [])
        return [
            entry["path"]
            for entry in tree
            if entry.get("type") == "blob" and _should_index(entry)
        ]

    async def _fetch_file(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        username: str,
        repo: dict,
        path: str,
    ) -> Document | None:
        branch = repo.get("default_branch") or "main"
        url = f"{RAW_BASE}/{username}/{repo['name']}/{branch}/{quote(path)}"
        async with semaphore:
            response = await client.get(url)
        if response.status_code != 200:
            return None
        content = response.text
        if not content.strip():
            return None
        return Document(
            page_content=content,
            metadata={
                "repo": repo["name"],
                "path": path,
                "language": _language_key_for(path),
                "url": f"{repo['html_url']}/blob/{branch}/{path}",
                "stars": repo.get("stargazers_count") or 0,
                "description": repo.get("description") or "",
            },
        )

    async def _respect_rate_limit(self, response: httpx.Response) -> None:
        remaining = int(response.headers.get("x-ratelimit-remaining", "0"))
        if remaining <= RATE_LIMIT_FLOOR:
            logger.warning("GitHub rate limit low (%d remaining); backing off %.0fs", remaining, RATE_LIMIT_BACKOFF_SECONDS)
            await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)


def _should_index(entry: dict) -> bool:
    path = entry["path"]
    size = entry.get("size") or 0
    return _is_high_signal(path) or _is_important(path, size)


def _is_high_signal(path: str) -> bool:
    name = Path(path).name.lower()
    if name in _HIGH_SIGNAL_FILES:
        return True
    return path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml"))


def _is_important(path: str, size: int) -> bool:
    lower = path.lower()
    if any(part in _EXCLUDED_DIRS for part in Path(path).parts):
        return False
    if Path(path).name.lower() in _EXCLUDED_FILENAMES:
        return False
    if any(lower.endswith(suffix) for suffix in _EXCLUDED_SUFFIXES):
        return False
    if size == 0 or size > get_settings().max_file_bytes:
        return False
    return True


def _language_key_for(path: str) -> str | None:
    return _LANGUAGE_KEY_BY_EXTENSION.get(Path(path).suffix.lower())
