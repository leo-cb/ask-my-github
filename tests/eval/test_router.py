"""Accuracy check for the agentic router (stats vs code classification)."""

import pytest

from ask_my_github.agentic.nodes import route_node
from ask_my_github.rag.llm import get_agentic_chat_model

_STATS = [
    "Which of my repositories has the most stars?",
    "How many commits does the anomaly-detection repository have?",
    "What is my oldest repository?",
    "Which repository was most recently pushed to?",
]

_CODE = [
    "How does the anomaly detection model work?",
    "What does the HeartDiseasePrediction_WebApp Flask app do?",
    "Which libraries does ask-my-github use internally?",
    "How is the FAISS index built in ask-my-github?",
]


@pytest.fixture(scope="module")
def router_llm():
    """The LLM backing the router node."""
    return get_agentic_chat_model()


def _classify(question: str, llm) -> str:
    return route_node({"question": question}, llm=llm)["route"]


def test_router_accuracy(router_llm):
    """Assert the router classifies a labeled stats/code set above a threshold."""
    labeled = [(q, "stats") for q in _STATS] + [(q, "code") for q in _CODE]
    correct = sum(_classify(q, router_llm) == expected for q, expected in labeled)
    accuracy = correct / len(labeled)
    assert accuracy >= 0.8, f"router accuracy {accuracy:.2f} is below 0.8"
