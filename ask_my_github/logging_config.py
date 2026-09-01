"""Centralized logging configuration for Ask My GitHub.

Configures a single root logger that writes to both the console (stderr) and a
rotating log file on disk, so runtime progress is visible in the terminal and
persisted for later inspection. Individual modules obtain a namespaced logger
via :func:`get_logger` and inherit this root configuration.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ask_my_github.config import Settings, get_settings

# Human-readable line format shared by the console and file handlers.
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024  # Rotate the log file after 10 MB.
_BACKUP_COUNT = 5  # Keep up to five rotated log files.

# Third-party loggers that are noisy at INFO (per-request HTTP logs, runtime
# internals, model loader chatter). Their INFO output is suppressed and only
# WARNING and above surfaces, keeping the console readable during ingestion.
_NOISY_LOGGERS = frozenset({
    "httpx",
    "httpcore",
    "urllib3",
    "fastembed",
    "onnxruntime",
    "faiss",
    "asyncio",
})


def setup_logging(settings: Settings | None = None) -> None:
    """Configure the root logger once for the whole process.

    The function is idempotent: repeated calls leave existing handlers intact,
    so it is safe to invoke from both the application lifespan and tests.

    :param settings: Optional application settings; defaults to the cached
        :class:`Settings` instance when omitted.
    """
    settings = settings or get_settings()

    root = logging.getLogger()
    root.setLevel(_level(settings.log_level))

    # Silence noisy third-party loggers so their INFO chatter does not drown
    # out the application's own log lines.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # A second call (e.g. reload during development) must not stack handlers.
    if root.handlers:
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(_level(settings.log_level))
    console.setFormatter(formatter)
    root.addHandler(console)

    log_path = Path(settings.log_dir) / settings.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(_level(settings.log_level))
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger that inherits the root configuration."""
    return logging.getLogger(name)


def _level(level_name: str | None) -> int:
    """Resolve a case-insensitive level name to its logging constant."""
    if not level_name:
        return logging.INFO
    return getattr(logging, level_name.upper(), logging.INFO)
