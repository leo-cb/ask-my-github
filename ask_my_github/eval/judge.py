"""LLM-as-judge factory for DeepEval metrics.

DeepSeek is OpenAI-compatible, so DeepEval ships a ready-made ``DeepSeekModel``
judge. This module just wires it to the project settings so every metric uses
the same judge without repeating credentials.
"""

from deepeval.models import DeepSeekModel

from ask_my_github.config import get_settings


def get_judge_model() -> DeepSeekModel:
    """Return the DeepSeek judge used by all evaluation metrics."""
    settings = get_settings()
    return DeepSeekModel(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        # Judges should be deterministic; the answer model keeps its own temperature.
        temperature=0.0,
    )
