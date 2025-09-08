from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import os
from models import * 
from rag import RAG

app = FastAPI(title="RAG API", version="1.0.0")

# Initialize RAG once
rag: Optional[RAG] = None

@app.on_event("startup")
def _startup() -> None:
    global rag
    try:
        rag = RAG()
        # Allow overriding embedding model via env `EMBEDDING_MODEL` when loading
        rag.load("vector_store")
    except Exception as e:
        # Let startup continue; requests will error until vector store exists
        print(f"[startup] RAG initialization failed: {e}")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    try:
        results = rag.retrieve(
            query=req.question,
            bulletin_department=req.bulletin_department,
            cab_department=req.cab_department,
            top_k_bulletin=req.top_k_bulletin,
            top_k_cab=req.top_k_cab,
        )
        context = "\n".join(r.get("text", "") for r in results)
        answer = rag.generate(req.question, context)
        return QueryResponse(answer=answer, retrieved=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Simple LLM-based evaluator: ask the model to grade 0-1 with brief rationale."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        rubric = PromptTemplate.from_template(
            """
            You are grading a RAG answer using only the provided context snippets.
            Return a JSON with keys score (0..1) and reasoning.

            Question: {question}
            Answer: {answer}
            Context:
            {context}

            Grading criteria:
            - Faithfulness to context (no hallucination)
            - Relevance to the question
            - Clarity and completeness

            Output strictly as JSON: {{"score": <0..1>, "reasoning": "..."}}
            """
        )
        rendered = rubric.format(
            question=req.question,
            answer=req.answer,
            context="\n---\n".join(req.context),
        )
        resp = model.invoke(rendered).content
        # Best-effort parse
        import json
        try:
            data = json.loads(resp)
            score = float(data.get("score", 0))
            reasoning = str(data.get("reasoning", ""))
        except Exception:
            # Fallback simple heuristic
            score = 0.0
            reasoning = f"Unparseable model output: {resp}"
        score = max(0.0, min(1.0, score))
        return EvaluateResponse(score=score, reasoning=reasoning)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Local dev entrypoint: uvicorn api:app --reload
