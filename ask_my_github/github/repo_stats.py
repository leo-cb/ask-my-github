"""Repository-level statistics persisted in a SQLite table, separate from FAISS.

The FAISS vector store only indexes ``page_content`` (code), so repo-level facts
like commit counts and stars are not searchable there. This module keeps those
facts in a small relational table, one row per repository, that a dedicated
tool can query directly to answer aggregate questions.
"""

import sqlite3
from pathlib import Path
from typing import Any

from ask_my_github.config import get_settings
from ask_my_github.logging_config import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    username       TEXT NOT NULL,
    name           TEXT NOT NULL,
    description    TEXT,
    language       TEXT,
    stars          INTEGER NOT NULL DEFAULT 0,
    forks          INTEGER NOT NULL DEFAULT 0,
    open_issues    INTEGER NOT NULL DEFAULT 0,
    commit_count   INTEGER,
    size_kb        INTEGER,
    license        TEXT,
    topics         TEXT,
    default_branch TEXT,
    created_at     TEXT,
    pushed_at      TEXT,
    updated_at     TEXT,
    html_url       TEXT,
    file_count     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (username, name)
)
"""

# Non-username columns, matching the dict keys produced by
# ``GitHubLoader._repo_stats_dict``.
_COLUMNS = (
    "name",
    "description",
    "language",
    "stars",
    "forks",
    "open_issues",
    "commit_count",
    "size_kb",
    "license",
    "topics",
    "default_branch",
    "created_at",
    "pushed_at",
    "updated_at",
    "html_url",
    "file_count",
)

_INSERT_SQL = (
    "INSERT OR REPLACE INTO repos (username, "
    + ", ".join(_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in range(len(_COLUMNS) + 1))
    + ")"
)


def repo_stats_db_path() -> Path:
    """Return the on-disk path of the repository statistics database."""
    return Path(get_settings().faiss_dir).parent / "repo_stats.db"


def save_repo_stats(username: str, repos: list[dict[str, Any]]) -> None:
    """Persist one row per repository for a user, upserting on (username, name)."""
    path = repo_stats_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(username, *(repo.get(col) for col in _COLUMNS)) for repo in repos]
    with sqlite3.connect(path) as connection:
        connection.execute(_SCHEMA)
        connection.executemany(_INSERT_SQL, rows)
    logger.info("Saved %d repo stats rows for user '%s'", len(repos), username)


def load_repo_stats(username: str) -> list[dict[str, Any]]:
    """Return the persisted repo statistics for a user, most recently pushed first."""
    path = repo_stats_db_path()
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM repos WHERE username = ? ORDER BY pushed_at DESC",
            (username,),
        ).fetchall()
    return [dict(row) for row in rows]


def current_username() -> str:
    """Resolve the active username from app state, falling back to settings."""
    from ask_my_github.main import app

    username = getattr(app.state, "username", None)
    if username:
        return username
    return get_settings().github_username or ""
