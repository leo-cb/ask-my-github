# Ask My GitHub

Fetches GitHub repositories for a user, loads README content, and builds a FAISS vector store with sentence-transformer embeddings. Intended as a small building block for RAG-style exploration of GitHub repos.

## Features

- Load public repositories for a GitHub username
- Fetch and include repository README content
- Build a FAISS vector store with Hugging Face embeddings
- Query indexed repos via a FastAPI API

## Requirements

- Python 3.12
- GitHub token (optional, recommended for higher rate limits)

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Load repositories

```python
from app.github.loader import GitHubLoader

loader = GitHubLoader(token="YOUR_GITHUB_TOKEN")
repos = loader.load_user_repos("octocat")

print(repos[0]["content"])
```

### Build a vector store

```python
from app.github.loader import GitHubLoader
from app.rag.embeddings import build_vector_store

loader = GitHubLoader(token="YOUR_GITHUB_TOKEN")
repos = loader.load_user_repos("octocat")

vector_store = build_vector_store(repos)
```

### Run the API

```bash
uvicorn app.main:app --reload
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
app/
  github/
    loader.py
  rag/
    embeddings.py
requirements.in
requirements.txt
```
