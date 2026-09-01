"""Prompt templates for general-purpose repository Q&A."""

from langchain_core.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant that answers questions about software repositories.\n"
        "Use ONLY the provided repository content to answer the question.\n"
        "If the answer is not in the content, say you don't know.\n\n"
        "Repository content:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

GRADE_PROMPT = PromptTemplate(
    input_variables=["question", "document"],
    template=(
        'Assess whether the document below is relevant to the question.\n'
        'Reply with a single word: "yes" or "no".\n\n'
        "Question: {question}\n\n"
        "Document:\n{document}"
    ),
)

REWRITE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Rewrite the following question to be better suited for semantic search "
        "over code repositories. Return only the rewritten question.\n\n"
        "Question: {question}"
    ),
)

GENERATE_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about software repositories and code.\n"
    "Answer using only the provided context. Cite file paths when relevant.\n"
    "If the context is insufficient to answer, say so instead of guessing."
)


ROUTER_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        'Classify the question into exactly one category.\n'
        '- "stats": repository-level or aggregate questions about commit counts, '
        'stars, forks, languages, or creation/update dates, or comparisons using '
        '"most", "highest", "fewest", "oldest", or "newest".\n'
        '- "code": questions about source code, files, implementation, or content '
        'inside a repository.\n'
        'Reply with a single word: "stats" or "code".\n\n'
        'Question: {question}'
    ),
)
