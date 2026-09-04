"""CLI utility to ingest a list of GitHub users into the local store."""

import asyncio
import sys

from ask_my_github.github.ingest import ingest_user
from ask_my_github.logging_config import get_logger, setup_logging

logger = get_logger("ingest_users")


async def main() -> None:
    for username in sys.argv[1:]:
        logger.info("INGESTING %s", username)
        await ingest_user(username)
        logger.info("DONE %s", username)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
