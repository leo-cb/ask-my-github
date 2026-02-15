# Ask My GitHub

Fetches GitHub repositories for a user, loads README content, and builds a FAISS vector store with sentence-transformer embeddings. Intended as a small building block for RAG-style exploration of GitHub repos.

## Features

- Load public repositories for a GitHub username
- Fetch and include repository README content
- Build a FAISS vector store with Hugging Face embeddings

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

## Notes

- If you do not provide a token, GitHub API rate limits are much lower.
- README content is fetched via the GitHub REST API and decoded as UTF-8.
- After building the vector store, you can query the repo using the API created by FastAPI.

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
