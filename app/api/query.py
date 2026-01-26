from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from app.rag.retriever import build_retriever
from app.rag.prompt import RECRUITER_PROMPT

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
def query_github(request: QueryRequest):
    from app.main import app

    vector_store = getattr(app.state, "vector_store", None)

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No GitHub data ingested yet. Call /ingest/github first."
        )

    try:
        retriever = build_retriever(vector_store)

        llm = OllamaLLM(
            model="llama3.2:1b",
            base_url="http://localhost:11434",
            temperature=0.2,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": RECRUITER_PROMPT},
            return_source_documents=True,
        )

        result = qa_chain.invoke({"query": request.question})

        sources = [
            {
                "repo": doc.metadata.get("repo"),
                "url": doc.metadata.get("url"),
                "language": doc.metadata.get("language"),
            }
            for doc in result.get("source_documents", [])
        ]

        return {
            "question": request.question,
            "answer": result["result"],
            "sources": sources,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
