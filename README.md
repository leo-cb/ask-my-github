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
- **Agentic path** — a Corrective RAG LangGraph with query rewriting, document
  grading, and a ReAct tool-agent fallback (GitHub code search / file read).
- Path selected via `IS_FAST_RAG`.
- Cloud (OpenAI/Anthropic/DeepSeek) and local (Ollama) LLMs, switchable per path.
- LangSmith tracing for the agentic graph and chains.
- FAISS index persisted to disk per user.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- GitHub token (recommended for higher rate limits)

## Install

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
  user's repos, builds and persists the FAISS index.
- `POST /query` with JSON body `{"question": "..."}` — answers using the fast or
  agentic path depending on `IS_FAST_RAG`.

## Notes

- The agentic path's ReAct tool fallback requires a **tool-calling-capable**
  model. The default agentic provider is OpenAI (`gpt-4o-mini`), which works out
  of the box. If you switch the agentic path to Ollama, use a model with solid
  tool-calling support (e.g. `qwen2.5-coder`). `llama3.2` supports tools too,
  but smaller models can be unreliable at emitting valid tool calls.

## Project Structure

```
ask_my_github/
  __init__.py
  main.py
  config.py
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
