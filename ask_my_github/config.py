"""Application settings loaded from environment variables or a .env file."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Ask My GitHub."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    is_fast_rag: str | None = None
    llm_provider: str | None = None

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    deepseek_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.2

    github_token: str | None = None
    github_username: str | None = None

    embedding_provider: str | None = None
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    faiss_dir: str = "./.data/faiss"
    max_file_bytes: int = 200_000
    max_concurrency: int = 8
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 15

    langchain_api_key: str | None = None
    langchain_project: str = "ask-my-github"
    langchain_endpoint: str | None = None

    log_dir: str = "./.logs"
    log_file: str = "app.log"
    log_level: str = "INFO"

    @property
    def use_fast_rag(self) -> bool:
        """Return True only when IS_FAST_RAG is exactly "1"."""
        return self.is_fast_rag == "1"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def configure_tracing(settings: Settings) -> None:
    """Enable LangSmith tracing when an API key is configured."""
    if not settings.langchain_api_key:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
