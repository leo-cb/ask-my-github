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
    input_variables=["repo_name", "context"],
    template=(
        "The excerpts below are files from the repository '{repo_name}'.\n"
        "Identify the technologies actually used: (1) programming languages with "
        "the third-party libraries/frameworks imported or used in the code, and "
        "(2) DevOps/infrastructure tools and platforms (such as Docker, "
        "Kubernetes, Jenkins, Ansible, Terraform, Tomcat, Maven, GitHub Actions, "
        "AWS, GCP, etc.) when the content is configuration, deployment, or CI/CD "
        "related.\n"
        "Base your answer ONLY on what is visible in the content: imports and "
        "usage in code, and tool names referenced in configs, Dockerfiles, "
        "deployment manifests, and CI/CD files.\n"
        "Use the canonical package name (e.g. 'scikit-learn' not 'sklearn', "
        "'Pillow' not 'PIL'). Do not invent tools that are not visible.\n"
        "Exclude low-level/utility modules and standard-library modules (such as "
        "os, sys, json, re, datetime, itertools, typing, pathlib, collections, "
        "subprocess, functools, abc, networkx, pickleshare) and list only "
        "meaningful, user-facing libraries, frameworks, and tools.\n"
        "List at most 15 items per language/tool group, ordered from most to "
        "least used. Put DevOps/infrastructure tools under the key "
        '"DevOps & Infra".\n'
        "Return ONLY a JSON object mapping each language or tool group to a list "
        "of its technologies.\n"
        'Example: {{"Python": ["pandas", "fastapi"], "JavaScript": ["react"], '
        '"DevOps & Infra": ["docker", "kubernetes"]}}\n\n'
        "Files:\n{context}\n\n"
        "JSON:"
    ),
)
