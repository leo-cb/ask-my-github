"""Prompt templates for the dashboard's LLM-backed widgets."""

from langchain_core.prompts import PromptTemplate

REPO_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["repo_name", "context"],
    template=(
        "You are summarizing a software repository.\n"
        "Using ONLY the provided content, write a concise summary (2-4 sentences) "
        "of what the repository '{repo_name}' does, its purpose, and its main features.\n"
        "Do not mention that this summary was generated or reference source files.\n\n"
        "Repository content:\n{context}\n\n"
        "Summary:"
    ),
)

TECHNOLOGIES_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=(
        "Analyze the dependency and manifest files below.\n"
        "Identify the programming languages used and the libraries/frameworks "
        "associated with each language.\n"
        "Return ONLY a JSON object mapping each language to a list of its "
        "libraries/frameworks.\n"
        "List at most 15 libraries/frameworks per language.\n"
        'Example: {{"Python": ["pandas", "fastapi"], "JavaScript": ["react"]}}\n'
        "Include a language even if its library list is empty.\n\n"
        "Manifest content:\n{context}\n\n"
        "JSON:"
    ),
)
