"""Retrieval-quality evaluation of the shared FAISS retriever.

These metrics score only the retrieved chunks (precision, recall, relevancy),
so they run the retriever once per question without any answer generation.
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from ask_my_github.eval.dataset import load_goldens
from ask_my_github.eval.metrics import retrieval_metrics

GOLDENS = load_goldens()


@pytest.mark.parametrize("golden", GOLDENS, ids=[g["input"][:40] for g in GOLDENS])
def test_retrieval_quality(golden, fast_pipeline, judge):
    """Score the top-k chunks the retriever returns for each golden question."""
    retrieval_context = fast_pipeline.retrieve(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output="",
        expected_output=golden["expected_output"],
        retrieval_context=retrieval_context,
        context=golden.get("context"),
    )
    assert_test(test_case, metrics=retrieval_metrics(judge))
