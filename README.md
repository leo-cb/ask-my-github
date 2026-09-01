# Ask My GitHub

A general-purpose retrieval-augmented generation (RAG) system specialized for
GitHub and code. It scrapes a user's public repositories (all important source
files, not just the README), builds a FAISS index, and answers questions with a
fast one-shot RAG path or a slower agentic path built on LangGraph — both traced
with LangSmith.

## Features

- Async, parallel scrape of whole repositories (source files + high-signal
  metadata files), with language-aware chunking.
- **Fast path** — one-shot RAG using LangChain LCEL (cloud or local Ollama LLM).

  ```mermaid
  flowchart LR
      A[User question] --> B[FAISS retriever]
      B --> C[Top-k relevant chunks]
      C --> D[QA prompt + LLM]
      D --> E[Answer + sources]
  ```

- **Agentic path** — a Corrective RAG LangGraph with an LLM router, query
  rewriting, document grading, and a ReAct tool-agent fallback (GitHub code
  search / file read / repo stats).

  ```mermaid
  flowchart TD
      A[START] --> B[Route — LLM router]
      B -->|stats| C[Stats node — SQLite repo-stats table]
      B -->|code| D[Retrieve — FAISS retriever]
      C --> E[Generate]
      D --> F[Grade — document grader]
      F -->|relevant| E
      F -->|irrelevant, iterations < 3| G[Transform query — rewrite]
      G --> D
      F -->|irrelevant, iterations = 3| H[Tool fallback — ReAct agent]
      H --> I[GitHub code search / file read / repo stats]
      E --> J[END]
      I --> J
  ```
- **Repo-level stats table** — per-repository facts (commits, stars, forks,
  language, dates, fork status) stored in a SQLite table separate from the
  vector store. An LLM router classifies each question as `stats` (answered
  from the table) or `code` (answered from the FAISS index).
- Path selected via `IS_FAST_RAG`.
- Cloud (OpenAI/Anthropic/DeepSeek) and local (Ollama) LLMs, switchable per path.
- LangSmith tracing for the agentic graph and chains.
- FAISS index and repo-stats DB persisted to disk per user.

## Requirements

- Python 3.12+


**Optional:**
- [Ollama](https://ollama.com/download) installed (if you wish to run local LLMs)
- One of these: OpenAI/Anthropic/Deepseek API key (you can use local models with Ollama instead)
- [uv](https://docs.astral.sh/uv/) (for dependency management; pip also works)
- GitHub token (recommended for higher rate limits)
- Langsmith API key (cloud LLMs tracing)

## Install

### With pip

1. Clone this repository to your local machine:
```bash
git clone https://github.com/leo-cb/ask-my-github.git
```

2. Install dependencies:
```bash
pip install -e .
```

### With uv

1. Clone this repository to your local machine:
```bash
git clone https://github.com/leo-cb/ask-my-github.git
```

2. If you don't have `uv` yet, install it first (it's a single binary / PyPI
package):

```bash
pip install uv
```

3. Then sync the project dependencies:

```bash
uv sync
```

## Configuration

Copy `.env.example` to `.env` and fill in what you need. Key variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `IS_FAST_RAG` | `1` = fast one-shot; anything else = agentic | unset (agentic) |
| `LLM_PROVIDER` | `openai` \| `anthropic` \| `deepseek` \| `ollama` | **required** (no default) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | cloud credentials | — |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` | DeepSeek (OpenAI-compatible, tool-calling) | `deepseek-chat` |
| `OLLAMA_MODEL` | any locally installed Ollama model (e.g. `qwen2.5-coder`) | `llama3.2` |
| `EMBEDDING_PROVIDER` | `fastembed` (local, ONNX) \| `openai` (cloud) | **required** (no default) |
| `EMBEDDING_MODEL` | model for the embedding provider (fastembed / `text-embedding-3-small`) | `jinaai/jina-embeddings-v2-base-code` |
| `GITHUB_TOKEN` | GitHub auth for higher rate limits | — |
| `LANGCHAIN_API_KEY` | enables LangSmith tracing | — |

## Usage

### Run the API

```bash
uvicorn ask_my_github.main:app --reload
```

### API endpoints

- `POST /ingest/github` with JSON body `{"username": "octocat"}` — scrapes the
  user's repos, builds and persists the FAISS index, and saves the repo-stats
  SQLite table.
- `POST /query` with JSON body `{"question": "..."}` — answers using the fast or
  agentic path depending on `IS_FAST_RAG`.

## Notes

- The agentic path's router and ReAct tool fallback require a
  **tool-calling-capable** model. The default agentic provider is OpenAI
  (`gpt-4o-mini`), which works out of the box. If you switch the agentic path to
  Ollama, use a model with solid tool-calling support (e.g. `qwen2.5-coder`).
  `llama3.2` supports tools too, but smaller models can be unreliable at
  emitting valid tool calls.
- Repo-level questions ("most stars", "highest commits", "which are forks")
  are answered from the stats table via the router; code questions go through
  the FAISS index. Re-ingest after changing scraped repos to refresh both.

## Project Structure

```
ask_my_github/
  __init__.py
  main.py
  config.py
  logging_config.py
  api/
    __init__.py
    ingest.py
    query.py
  github/
    __init__.py
    loader.py
    repo_stats.py
  rag/
    __init__.py
    embeddings.py
    llm.py
    prompt.py
    retriever.py
    splitter.py
    store.py
  agentic/
    __init__.py
    state.py
    tools.py
    nodes.py
    graph.py
.github/
  workflows/
    gitleaks.yml
pyproject.toml
uv.lock
```
