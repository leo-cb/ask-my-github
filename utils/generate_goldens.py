"""Generate synthetic evaluation goldens from ingested repository chunks.

Samples chunks from a user's FAISS store and asks an LLM to write a question
plus an expected answer that each chunk supports. The output is written next to
the hand-curated ``goldens.json`` so both files feed the same evaluation run.

Usage:
    python utils/generate_goldens.py [username] [--count 40]
"""

import argparse
import json
import random
from pathlib import Path

from ask_my_github.config import get_settings
from ask_my_github.eval.dataset import resolve_eval_user
from ask_my_github.logging_config import get_logger
from ask_my_github.rag.llm import get_fast_chat_model
from ask_my_github.rag.store import load_vector_store

logger = get_logger(__name__)

_GENERATE_PROMPT = (
    "Given the repository chunk below, write one question a user might ask that "
    "this chunk answers, and the expected answer grounded only in the chunk.\n"
    'Return a JSON object with exactly the keys "input" and "expected_output" '
    "and nothing else.\n\n"
    "Repository: {repo}\nFile: {path}\n\nChunk:\n{chunk}"
)


def _parse_json(text: str) -> dict | None:
    """Extract the first JSON object from an LLM reply, tolerating fences."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _generate_for_chunk(llm, document) -> dict | None:
    prompt = _GENERATE_PROMPT.format(
        repo=document.metadata.get("repo", ""),
        path=document.metadata.get("path", ""),
        chunk=document.page_content,
    )
    reply = llm.invoke(prompt).content
    golden = _parse_json(reply)
    if golden and golden.get("input") and golden.get("expected_output"):
        return golden
    logger.warning("Skipped chunk: could not parse golden from %s", document.metadata.get("path"))
    return None


def main() -> None:
    """Sample chunks and write synthetic goldens to the configured dataset dir."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", nargs="?", default=None, help="GitHub username to evaluate")
    parser.add_argument("--count", type=int, default=40, help="number of goldens to generate")
    args = parser.parse_args()

    username = args.username or resolve_eval_user()
    vector_store = load_vector_store(username)
    if vector_store is None:
        raise SystemExit(f"No persisted index for '{username}'. Ingest it first.")

    ids = vector_store.index_to_docstore_id.values()
    documents = [vector_store.docstore.search(doc_id) for doc_id in ids]
    sample = random.sample(documents, min(args.count, len(documents)))

    llm = get_fast_chat_model()
    goldens = [g for doc in sample if (g := _generate_for_chunk(llm, doc))]

    out_path = Path(get_settings().eval_dataset_dir) / "goldens_synthetic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(goldens, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(goldens)} goldens to {out_path}")


if __name__ == "__main__":
    main()
