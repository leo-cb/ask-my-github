"""Chat model factory that resolves the provider from settings."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from ask_my_github.config import Settings, get_settings
from ask_my_github.logging_config import get_logger

logger = get_logger(__name__)

VALID_PROVIDERS = ("openai", "anthropic", "deepseek", "ollama")


def get_fast_chat_model() -> BaseChatModel:
    """Return the chat model for the fast one-shot path."""
    settings = get_settings()
    return _build_chat_model(settings, _require_provider(settings))


def get_agentic_chat_model() -> BaseChatModel:
    """Return the chat model for the agentic path."""
    settings = get_settings()
    return _build_chat_model(settings, _require_provider(settings))


def _require_provider(settings: Settings) -> str:
    provider = settings.llm_provider
    if not provider:
        raise ValueError(
            "LLM_PROVIDER is not set. Set it in .env to choose the model: "
            + " | ".join(VALID_PROVIDERS)
        )
    return provider


def _build_chat_model(settings: Settings, provider: str) -> BaseChatModel:
    kwargs: dict[str, Any] = {
        "model": _model_for(settings, provider),
        "model_provider": provider,
        "temperature": settings.llm_temperature,
    }
    if provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url
    if provider == "deepseek":
        kwargs["model_provider"] = "openai"
        kwargs["base_url"] = settings.deepseek_base_url
        if settings.deepseek_api_key:
            kwargs["api_key"] = settings.deepseek_api_key
    logger.info("Building chat model: provider=%s model=%s", provider, kwargs["model"])
    return init_chat_model(**kwargs)


def _model_for(settings: Settings, provider: str) -> str:
    return {
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model,
        "ollama": settings.ollama_model,
        "deepseek": settings.deepseek_model,
    }[provider]
