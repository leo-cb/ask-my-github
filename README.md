# Ask My GitHub

Fetches GitHub repositories for a user, loads README content, and builds a FAISS vector store with sentence-transformer embeddings. Intended as a small building block for RAG-style exploration of GitHub repos.

## Features

- Load public repositories for a GitHub username
- Fetch and include repository README content
- Build a FAISS vector store with Hugging Face embeddings
- Query indexed repos via a FastAPI API

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- GitHub token (optional, recommended for higher rate limits)

## Install

```bash
uv sync
```

This installs all dependencies into a virtual environment managed by `uv`.

## Secret scanning (GitHub Actions)

Secrets (API keys, tokens, passwords) are detected by
[gitleaks-action](https://github.com/gitleaks/gitleaks-action), defined in
[.github/workflows/gitleaks.yml](.github/workflows/gitleaks.yml). It runs on
every push and pull request, and once daily on a schedule.

There is nothing to install locally — the scan runs in GitHub Actions.

- For personal accounts, no setup is required.
- For organization accounts, a free `GITLEAKS_LICENSE` secret is required (see
  [gitleaks.io](https://gitleaks.io/)).

## Usage

### Load repositories

```python
from ask_my_github.github.loader import GitHubLoader

loader = GitHubLoader(token="YOUR_GITHUB_TOKEN")
repos = loader.load_user_repos("octocat")

print(repos[0]["content"])
```

### Build a vector store

```python
from ask_my_github.github.loader import GitHubLoader
from ask_my_github.rag.embeddings import build_vector_store

loader = GitHubLoader(token="YOUR_GITHUB_TOKEN")
repos = loader.load_user_repos("octocat")

vector_store = build_vector_store(repos)
```

### Run the API

```bash
uvicorn ask_my_github.main:app --reload
```

### API endpoints

- `POST /ingest/github` with JSON body `{"username": "octocat"}`
- `POST /query` with JSON body `{"question": "What does this repo do?"}`

## Notes

- If you do not provide a token, GitHub API rate limits are much lower.
- README content is fetched via the GitHub REST API and decoded as UTF-8.
- The API stores the vector store in memory; call `/ingest/github` before `/query`.
- The `/query` endpoint uses Ollama at `http://localhost:11434`.

## Project Structure

```
ask_my_github/
  __init__.py
  main.py
  api/
    __init__.py
    ingest.py
    query.py
  github/
    __init__.py
    loader.py
  rag/
    __init__.py
    embeddings.py
    prompt.py
    retriever.py
.github/
  workflows/
    gitleaks.yml
pyproject.toml
uv.lock
```
