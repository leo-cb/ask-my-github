"""Container entrypoint: load persisted indexes, then launch the dashboard."""

import asyncio
import os
import sys

from ask_my_github.config import get_settings, require_cloud_llm
from ask_my_github.github.ingest import load_user_index
from ask_my_github.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

_APP_PATH = "ask_my_github/dashboard/app.py"


def _parse_users(settings) -> list[str]:
    """Split DASHBOARD_USERS into a deduplicated list of trimmed usernames."""
    seen: set[str] = set()
    users: list[str] = []
    for name in (settings.dashboard_users or "").split(","):
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            users.append(name)
    return users


def _ingest_all(users: list[str]) -> None:
    """Load each user's persisted index up front so the dashboard is instant.

    This is a demo-only path: it reads from the persisted FAISS vector store
    and SQLite repo-stats table and never scrapes GitHub in real time. If the
    persisted data is missing, it fails loudly instead of triggering a scrape.
    """
    for username in users:
        logger.info("Loading persisted index for '%s'", username)
        asyncio.run(load_user_index(username))


def _launch_streamlit(port: str | None) -> None:
    """Run the Streamlit server in-process.

    The port is taken from Streamlit's own ``STREAMLIT_SERVER_PORT`` env var
    when set, falling back to Streamlit's default (8501) otherwise.
    """
    from streamlit.web import cli as stcli

    argv = [
        "streamlit",
        "run",
        _APP_PATH,
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    if port:
        argv += ["--server.port", port]
    sys.argv = argv
    sys.exit(stcli.main())


def main() -> None:
    """Assert configuration, load persisted indexes, and start the dashboard."""
    setup_logging()
    settings = get_settings()
    require_cloud_llm(settings)
    users = _parse_users(settings)
    if not users:
        raise SystemExit(
            "DASHBOARD_USERS is not set. Set it in .env to a comma-separated "
            "list of GitHub usernames to display in the dashboard."
        )
    _ingest_all(users)
    port = os.environ.get("STREAMLIT_SERVER_PORT")
    _launch_streamlit(port)


if __name__ == "__main__":
    main()
