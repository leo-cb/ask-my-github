"""Generation-quality evaluation of the agentic (corrective RAG) path."""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from ask_my_github.eval.dataset import load_goldens
from ask_my_github.eval.metrics import generation_metrics

GOLDENS = load_goldens()


@pytest.mark.parametrize("golden", GOLDENS, ids=[g["input"][:40] for g in GOLDENS])
def test_agentic_rag_answer(golden, agentic_pipeline, judge):
    """Score the agentic answer for faithfulness, relevancy, and correctness."""
    result = agentic_pipeline.answer(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=result["answer"],
        expected_output=golden["expected_output"],
        retrieval_context=result["retrieval_context"],
        context=golden.get("context"),
    )
    assert_test(test_case, metrics=generation_metrics(judge))
