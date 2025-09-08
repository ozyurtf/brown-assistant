from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional, Dict
import os
import json
from statistics import mean
from models import * 
from rag import RAG
from utils import bleu_score as compute_bleu_score, exact_match as compute_exact_match

app = FastAPI(title="RAG API", version="1.0.0")

# Store multiple RAG instances for different models
rag_instances: Dict[str, RAG] = {}
cached_evals: Dict[str, EvaluateResponse] = {}

# Control whether to precompute evaluation metrics on startup
PRECOMPUTE_EVAL = os.getenv("PRECOMPUTE_EVAL", "false").lower() in {"1", "true", "yes", "on"}

# Available embedding models
AVAILABLE_MODELS = {
    "sentence_transformer": "all-MiniLM-L6-v2",
    "openai": "text-embedding-3-large"
}

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "openai"

def get_or_create_rag(embedding_model: str) -> RAG:
    """Get existing RAG instance or create new one for the specified model."""
    if embedding_model not in rag_instances:
        try:
            model_name = AVAILABLE_MODELS[embedding_model]
            rag = RAG(model_name=model_name)
            rag.load("vector_store", model_name=model_name)
            rag_instances[embedding_model] = rag
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RAG: {e}")
    
    return rag_instances[embedding_model]

def compute_evaluation_summary(embedding_model: str) -> EvaluateResponse:
    with open("evaluation.json", "r") as f:
        evaluation = json.load(f)

    rag = get_or_create_rag(embedding_model)

    bleu_scores: List[float] = []
    exact_match_scores: List[float] = []

    for data in evaluation:
        question = data.get('question', '')
        golden_answer = data.get('answer', '')
        bulletin_department = data.get('bulletin_department', '')
        cab_department = data.get('cab_department', '')

        results = rag.retrieve(
            query=question,
            bulletin_department=bulletin_department,
            cab_department=cab_department,
            top_k_bulletin=3,
            top_k_cab=5,
            rerank_top_n=5,
            rerank_min_score=0.2,
            rerank_model_name="BAAI/bge-reranker-base",
        )

        context = "\n".join(r.get("text", "") for r in results)
        generated_answer = rag.generate(question, context)

        bleu = compute_bleu_score(generated_answer, golden_answer)
        exact_match = compute_exact_match(generated_answer, golden_answer)

        bleu_scores.append(float(bleu))
        exact_match_scores.append(float(exact_match))

    return EvaluateResponse(
        bleu_score=float(mean(bleu_scores)) if bleu_scores else 0.0,
        exact_match_score=float(mean(exact_match_scores)) if exact_match_scores else 0.0,
        embedding_model=embedding_model
    )

@app.on_event("startup")
def startup() -> None:
    """Initialize RAG instances for available models at startup."""
    print("Starting up RAG API...")
    
    # Try to preload both models
    for embedding_model in AVAILABLE_MODELS.keys():
        try:
            get_or_create_rag(embedding_model)
            print(f"Successfully initialized {embedding_model}")
        except Exception as e:
            print(f"Failed to initialize {embedding_model}: {e}")
    
    # Optionally precompute evaluations for available models
    if PRECOMPUTE_EVAL:
        for embedding_model in AVAILABLE_MODELS.keys():
            try:
                if embedding_model in rag_instances:
                    cached_evals[embedding_model] = compute_evaluation_summary(embedding_model)
                    print("--------------------------------")
                    print(cached_evals[embedding_model])
                    print("--------------------------------")
                    print(f"Precomputed evaluation for {embedding_model}")
            except Exception as e:
                print(f"Failed to precompute evaluation for {embedding_model}: {e}")

@app.get("/models")
def get_available_models():
    """Return list of available embedding models."""
    return {
        "available_models": list(AVAILABLE_MODELS.keys()),
        "model_details": AVAILABLE_MODELS,
        "default_model": DEFAULT_EMBEDDING_MODEL
    }

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    # Get embedding model from request, default to DEFAULT_EMBEDDING_MODEL
    embedding_model = req.embedding_model or DEFAULT_EMBEDDING_MODEL
    
    if embedding_model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid embedding model: {embedding_model}. Available: {list(AVAILABLE_MODELS.keys())}"
        )
    
    try:
        rag = get_or_create_rag(embedding_model)
        
        results = rag.retrieve(
            query=req.question,
            bulletin_department = req.bulletin_department,
            cab_department = req.cab_department,
            top_k_bulletin = 2,
            top_k_cab = 5,
            rerank_top_n = 5,
            rerank_min_score=0.2,
            rerank_model_name = req.rerank_model_name,
        )
        context = "\n".join(r.get("text", "") for r in results)
        answer = rag.generate(req.question, context)
        return QueryResponse(answer=answer, retrieved=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Return cached evaluation summary for specified model."""
    embedding_model = getattr(req, 'embedding_model', None) or DEFAULT_EMBEDDING_MODEL
    
    if embedding_model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid embedding model: {embedding_model}. Available: {list(AVAILABLE_MODELS.keys())}"
        )
    
    if embedding_model not in cached_evals:
        # Compute on demand if not cached
        try:
            cached_evals[embedding_model] = compute_evaluation_summary(embedding_model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    
    return cached_evals[embedding_model]