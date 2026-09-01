"""Asynchronous GitHub repository loader."""

import asyncio
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

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

    async def load_repos(self, username: str) -> tuple[list[Document], list[dict]]:
        """Load all indexable files and per-repository statistics for a user."""
        async with self._build_client() as client:
            repos = await self._list_user_repos(client, username)
            logger.info("Found %d repositories for user '%s'", len(repos), username)
            semaphore = asyncio.Semaphore(get_settings().max_concurrency)
            results = await asyncio.gather(
                *(self._load_repo(client, semaphore, username, repo) for repo in repos)
            )
        documents = [document for docs, _ in results for document in docs]
        stats = [stats for _, stats in results]
        logger.info(
            "Loaded %d documents and %d repo stats for user '%s'",
            len(documents),
            len(stats),
            username,
        )
        return documents, stats

    def _build_client(self) -> httpx.AsyncClient:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return httpx.AsyncClient(headers=headers, timeout=120.0, follow_redirects=True)

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
    ) -> tuple[list[Document], dict]:
        repo_name = repo["name"]
        paths = await self._list_repo_files(client, semaphore, username, repo)
        commit_count, parent = await asyncio.gather(
            self._fetch_commit_count(client, semaphore, username, repo),
            self._fetch_parent(client, semaphore, username, repo),
        )
        logger.info(
            "Repository '%s': %d indexable files, %s commits",
            repo_name,
            len(paths),
            commit_count,
        )
        documents = await asyncio.gather(
            *(self._fetch_file(client, semaphore, username, repo, path) for path in paths)
        )
        loaded = [document for document in documents if document is not None]
        stats = self._repo_stats_dict(repo, commit_count, len(loaded), parent)
        logger.info(
            "Repository '%s': fetched %d/%d files",
            repo_name,
            len(loaded),
            len(paths),
        )
        return loaded, stats

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
        last_commit_date = await self._fetch_last_commit_date(
            client, semaphore, username, repo, path
        )
        return Document(
            page_content=content,
            metadata={
                "repo": repo["name"],
                "path": path,
                "language": _language_key_for(path),
                "url": f"{repo['html_url']}/blob/{branch}/{path}",
                "stars": repo.get("stargazers_count") or 0,
                "description": repo.get("description") or "",
                "last_commit_date": last_commit_date,
                "repo_pushed_at": repo.get("pushed_at"),
                "repo_updated_at": repo.get("updated_at"),
            },
        )

    async def _fetch_last_commit_date(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        username: str,
        repo: dict,
        path: str,
    ) -> str | None:
        """Return the ISO-8601 date of the last commit touching a file, or None."""
        url = f"{GITHUB_API_BASE}/repos/{username}/{repo['name']}/commits"
        params = {"path": path, "per_page": "1"}
        async with semaphore:
            response = await client.get(url, params=params)
        if response.status_code != 200:
            return None
        await self._respect_rate_limit(response)
        commits = response.json()
        if not commits:
            return None
        commit = commits[0].get("commit", {})
        for key in ("committer", "author"):
            date = (commit.get(key) or {}).get("date")
            if date:
                return date
        return None

    async def _fetch_commit_count(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        username: str,
        repo: dict,
    ) -> int | None:
        """Return the total number of commits in a repository, or None if unknown."""
        url = f"{GITHUB_API_BASE}/repos/{username}/{repo['name']}/commits"
        params = {"per_page": "1"}
        async with semaphore:
            response = await client.get(url, params=params)
        if response.status_code != 200:
            return None
        await self._respect_rate_limit(response)
        last_page = _last_page_from_link(response.headers.get("link", ""))
        if last_page is not None:
            return last_page
        commits = response.json()
        return len(commits) if isinstance(commits, list) else None

    async def _fetch_parent(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        username: str,
        repo: dict,
    ) -> str | None:
        """Return the repository this one was forked from, or None if not a fork.

        The ``/users/{username}/repos`` list endpoint omits the ``parent`` and
        ``source`` fields, so we fetch the single-repo endpoint for forks only.
        """
        if not repo.get("fork"):
            return None
        url = f"{GITHUB_API_BASE}/repos/{username}/{repo['name']}"
        async with semaphore:
            response = await client.get(url)
        if response.status_code != 200:
            return None
        await self._respect_rate_limit(response)
        payload = response.json()
        return (payload.get("parent") or payload.get("source") or {}).get("full_name")

    def _repo_stats_dict(
        self,
        repo: dict,
        commit_count: int | None,
        file_count: int,
        parent: str | None = None,
    ) -> dict:
        """Flatten a repo API object into the fields persisted in the stats table."""
        license_info = repo.get("license") or {}
        return {
            "name": repo.get("name", ""),
            "description": repo.get("description") or "",
            "language": repo.get("language") or "",
            "stars": repo.get("stargazers_count") or 0,
            "forks": repo.get("forks_count") or 0,
            "open_issues": repo.get("open_issues_count") or 0,
            "commit_count": commit_count,
            "size_kb": repo.get("size") or 0,
            "license": license_info.get("spdx_id") or license_info.get("name"),
            "topics": ",".join(repo.get("topics") or []),
            "default_branch": repo.get("default_branch") or "",
            "created_at": repo.get("created_at"),
            "pushed_at": repo.get("pushed_at"),
            "updated_at": repo.get("updated_at"),
            "html_url": repo.get("html_url") or "",
            "file_count": file_count,
            "is_fork": bool(repo.get("fork", False)),
            "parent": parent,
        }

    async def _respect_rate_limit(self, response: httpx.Response) -> None:
        remaining = int(response.headers.get("x-ratelimit-remaining", "0"))
        if remaining <= RATE_LIMIT_FLOOR:
            logger.warning("GitHub rate limit low (%d remaining); backing off %.0fs", remaining, RATE_LIMIT_BACKOFF_SECONDS)
            await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)


def _last_page_from_link(link_header: str) -> int | None:
    """Parse a GitHub pagination Link header and return the last page number."""
    for part in link_header.split(","):
        segment = part.split(";")
        if len(segment) < 2:
            continue
        url, *rels = segment
        if not any('rel="last"' in rel for rel in rels):
            continue
        pages = parse_qs(urlparse(url.strip().strip("<>")).query).get("page")
        if pages:
            try:
                return int(pages[0])
            except ValueError:
                return None
    return None


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
