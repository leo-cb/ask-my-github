"""Prompt templates for recruiter-facing answers."""

from langchain_core.prompts import PromptTemplate


RECRUITER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are helping a recruiter understand a software developer based on their GitHub repositories.

Use ONLY the information below to answer the question.
Be clear, concise, and avoid unnecessary technical jargon.
Always reference specific projects when possible.

GitHub project information:
{context}

Question:
{question}

Answer:
"""
)
