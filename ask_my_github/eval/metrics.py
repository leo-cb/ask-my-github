"""Metric factories for the RAG evaluation.

Each function builds a DeepEval metric bound to the shared DeepSeek judge.
Retrieval metrics score the FAISS retriever only; generation metrics score the
answer given the retrieved context.
"""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepSeekModel
from deepeval.test_case import SingleTurnParams

# Rubric for the correctness metric. It checks factual accuracy against the
# repository content rather than fluency, which is the real goal of a code RAG.
_CORRECTNESS_STEPS = [
    "The answer is factually correct about the repository, code, or project it describes.",
    "The answer does not invent files, functions, libraries, or behavior not present in the context.",
    "The answer directly addresses the question and, when relevant, names the correct repository or file.",
]

# Fields the correctness rubric is allowed to read when scoring.
_CORRECTNESS_PARAMS = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT,
    SingleTurnParams.RETRIEVAL_CONTEXT,
]


def retrieval_metrics(judge: DeepSeekModel) -> list:
    """Metrics that score the retrieved context against the question and answer."""
    return [
        ContextualPrecisionMetric(model=judge),
        ContextualRecallMetric(model=judge),
        ContextualRelevancyMetric(model=judge),
    ]


def generation_metrics(judge: DeepSeekModel) -> list:
    """Metrics that score the generated answer given the retrieved context."""
    return [
        FaithfulnessMetric(model=judge),
        AnswerRelevancyMetric(model=judge),
        GEval(
            name="Correctness",
            criteria=(
                "Evaluate whether the answer is factually correct about the "
                "repository or code described in the question."
            ),
            evaluation_steps=_CORRECTNESS_STEPS,
            evaluation_params=_CORRECTNESS_PARAMS,
            model=judge,
        ),
    ]
