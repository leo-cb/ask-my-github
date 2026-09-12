"""Golden-dataset loading and eval-user resolution.

Golden files are plain JSON lists where each entry is::

    {
        "input": "question asked to the RAG system",
        "expected_output": "ground-truth answer the retrieved context should support",
        "context": ["optional reference chunks", "used by contextual-recall metrics"]
    }

``context`` is optional; ``input`` and ``expected_output`` are required. The
hand-curated ``goldens.json`` and the auto-generated ``goldens_synthetic.json``
are merged so both contribute to the same evaluation run.
"""

import json
from pathlib import Path
from typing import Any

from ask_my_github.config import get_settings

_GOLDEN_FILES = ("goldens.json", "goldens_synthetic.json")


def resolve_eval_user() -> str:
    """Return the GitHub username whose index the evaluation should load."""
    settings = get_settings()
    if settings.eval_user:
        return settings.eval_user
    if settings.github_username:
        return settings.github_username
    if settings.dashboard_users:
        return settings.dashboard_users.split(",")[0].strip()
    raise ValueError("Set EVAL_USER or GITHUB_USERNAME in .env to pick a user to evaluate")


def load_goldens() -> list[dict[str, Any]]:
    """Load and merge all golden entries from the configured dataset directory."""
    data_dir = Path(get_settings().eval_dataset_dir)
    goldens: list[dict[str, Any]] = []
    for filename in _GOLDEN_FILES:
        path = data_dir / filename
        if path.exists():
            goldens.extend(json.loads(path.read_text(encoding="utf-8")))
    return goldens
