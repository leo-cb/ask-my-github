"""Chat model factory that resolves the provider from settings."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from ask_my_github.config import Settings, get_settings

_FAST_DEFAULT_PROVIDER = "ollama"
_AGENTIC_DEFAULT_PROVIDER = "openai"


def get_fast_chat_model() -> BaseChatModel:
    """Return the chat model for the fast one-shot path."""
    settings = get_settings()
    return _build_chat_model(settings, settings.llm_provider or _FAST_DEFAULT_PROVIDER)


def get_agentic_chat_model() -> BaseChatModel:
    """Return the chat model for the agentic path."""
    settings = get_settings()
    return _build_chat_model(settings, settings.llm_provider or _AGENTIC_DEFAULT_PROVIDER)


def _build_chat_model(settings: Settings, provider: str) -> BaseChatModel:
    kwargs: dict[str, Any] = {
        "model": _model_for(settings, provider),
        "model_provider": provider,
        "temperature": settings.llm_temperature,
    }
    if provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url
    return init_chat_model(**kwargs)


def _model_for(settings: Settings, provider: str) -> str:
    return {
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model,
        "ollama": settings.ollama_model,
    }[provider]
