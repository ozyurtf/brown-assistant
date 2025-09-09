from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional, Dict
import os
import time
import logging
from datetime import datetime
import json
from statistics import mean
from models import * 
from rag import RAG
from utils import compute_bleu_score, compute_rouge_score

app = FastAPI(title="RAG API", version="1.0.0")

LOGGER_NAME = "rag_logger"
LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE = os.path.join(LOG_DIR, "records.txt")
EVALUATION_FILE_PATH = "files/evaluation.json"

def init_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = init_logger()

def log_block(lines: List[str]) -> None:
    try:
        logger.info("\n".join(lines) + "\n")
    except Exception:
        pass

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
    try:
        with open(EVALUATION_FILE_PATH, "r") as f:
            evaluation = json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Evaluation file not found at '{EVALUATION_FILE_PATH}'. "
            f"Set EVALUATION_FILE env var or place evaluation.json under 'files/'."
        ) from e

    rag = get_or_create_rag(embedding_model)

    bleu_scores: List[float] = []
    rouge_scores: List[float] = []
    eval_start = time.perf_counter()

    for data in evaluation:
        question = data.get('question', '')
        golden_answer = data.get('answer', '')
        bulletin_department = data.get('bulletin_department', '')
        cab_department = data.get('cab_department', '')

        q_start = time.perf_counter()
        results = rag.retrieve(
            query=question,
            bulletin_department=bulletin_department,
            cab_department=cab_department,
            top_k_bulletin=3,
            top_k_cab=5,
            # rerank_top_n=5,
            # rerank_min_score=0.2,
            # rerank_model_name="BAAI/bge-reranker-base",
        )
        retrieval_s = (time.perf_counter() - q_start)
        context = "\n".join(r.get("text", "") for r in results)
        gen_start = time.perf_counter()
        generated_answer = rag.generate(question, context, max_tokens = 1000)
        gen_s = (time.perf_counter() - gen_start)

        bleu = compute_bleu_score(generated_answer, golden_answer)
        rouge = compute_rouge_score(generated_answer, golden_answer)

        bleu_scores.append(float(bleu))
        rouge_scores.append(float(rouge))

        # Log per-sample evaluation details
        try:
            ts = datetime.utcnow().isoformat() + "Z"
            lines = [
                "----- EVAL SAMPLE -----",
                f"timestamp: {ts}",
                f"embedding_model: {embedding_model}",
                f"question: {question}",
                f"bulletin_department: {bulletin_department}",
                f"cab_department: {cab_department}",
                f"retrieval_time_s: {retrieval_s:.3f}",
                f"generation_time_s: {gen_s:.3f}",
                f"retrieved_count: {len(results)}",
                f"bleu: {float(bleu):.4f}",
                f"rouge_l_f1: {float(rouge):.4f}",
                "generated_answer:",
                f"  {generated_answer}",
                "top_results:",
            ]
            for idx, r in enumerate(results[:3], start=1):
                meta = r.get("metadata", {}) or {}
                src = meta.get("source", "")
                dept = meta.get("department", "")
                sim = r.get("similarity_score", 0.0)
                lines.append(f"  {idx}) source={src} dept={dept} sim={sim}")
            lines.append("------------------------")
            log_block(lines)
        except Exception:
            pass

    total_eval_s = (time.perf_counter() - eval_start)

    summary = EvaluateResponse(
        bleu_score=float(mean(bleu_scores)) if bleu_scores else 0.0,
        rouge_score=float(mean(rouge_scores)) if rouge_scores else 0.0,
        embedding_model=embedding_model
    )

    # Log summary
    try:
        ts = datetime.utcnow().isoformat() + "Z"
        lines = [
            "----- EVAL SUMMARY -----",
            f"timestamp: {ts}",
            f"embedding_model: {embedding_model}",
            f"num_samples: {len(bleu_scores)}",
            f"avg_bleu: {summary.bleu_score:.4f}",
            f"avg_rouge_l_f1: {summary.rouge_score:.4f}",
            f"total_time_s: {total_eval_s:.3f}",
            "-------------------------",
        ]
        log_block(lines)
    except Exception:
        pass

    return summary

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
                    print(f"Precomputed evaluation for {embedding_model}")
            except Exception as e:
                print(f"Failed to precompute evaluation for {embedding_model}: {e}")

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
        req_start = time.perf_counter()
        rag = get_or_create_rag(embedding_model)
        
        ret_start = time.perf_counter()
        results = rag.retrieve(
            query=req.question,
            bulletin_department = req.bulletin_department,
            cab_department = req.cab_department,
            top_k_bulletin = 2,
            top_k_cab = 10,
            rerank_top_n = 5,
            rerank_min_score=0.2,
            rerank_model_name = req.rerank_model_name,
        )
        retrieval_s = (time.perf_counter() - ret_start)
        context = "\n".join(r.get("text", "") for r in results)
        gen_start = time.perf_counter()
        answer = rag.generate(req.question, context, max_tokens = 1000)
        gen_s = (time.perf_counter() - gen_start)
        total_s = (time.perf_counter() - req_start)

        # Log query details
        try:
            ts = datetime.utcnow().isoformat() + "Z"
            lines = [
                "-------- QUERY ---------",
                f"timestamp: {ts}",
                f"embedding_model: {embedding_model}",
                f"question: {req.question}",
                f"bulletin_department: {req.bulletin_department}",
                f"cab_department: {req.cab_department}",
                f"retrieval_time_s: {retrieval_s:.3f}",
                f"generation_time_s: {gen_s:.3f}",
                f"total_time_s: {total_s:.3f}",
                f"retrieved_count: {len(results)}",
                "generated_answer:",
                f"  {answer}",
                "top_results:",
            ]
            for idx, r in enumerate(results[:3], start=1):
                meta = r.get("metadata", {}) or {}
                src = meta.get("source", "")
                dept = meta.get("department", "")
                sim = r.get("similarity_score", 0.0)
                lines.append(f"  {idx}) source={src} dept={dept} sim={sim}")
            lines.append("------------------------")
            log_block(lines)
        except Exception:
            pass

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
        try:
            cached_evals[embedding_model] = compute_evaluation_summary(embedding_model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    
    return cached_evals[embedding_model]