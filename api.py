from fastapi import FastAPI, HTTPException, Query, Depends, Header
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv
import time
import logging
from datetime import datetime
import json
from statistics import mean
from models import * 
from rag import RAG
from fastapi.responses import StreamingResponse

load_dotenv()
app = FastAPI(title="RAG API", version="1.0.0")

API_TOKEN = os.getenv("API_TOKEN")

def verify_token(authorization: str = Header(None)):
    """Verify API token from Authorization header."""
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="API_TOKEN not configured")
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format. Use: Bearer <token>")
    
    token = authorization.replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    return token

LOGGER_NAME = "rag_logger"
LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE = os.path.join(LOG_DIR, "records.txt")

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

rag_instances: Dict[str, RAG] = {}

AVAILABLE_MODELS = {
    "openai": "text-embedding-3-large"
}

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

@app.on_event("startup")
def startup() -> None:
    """Initialize RAG instances for available models at startup."""
    print("Starting up RAG API...")
    
    for embedding_model in AVAILABLE_MODELS.keys():
        try:
            get_or_create_rag(embedding_model)
            print(f"Successfully initialized {embedding_model}")
        except Exception as e:
            print(f"Failed to initialize {embedding_model}: {e}")
    
@app.post("/query")
def query(req: QueryRequest, token: str = Depends(verify_token)) -> StreamingResponse:
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
            top_k = 5,
            rerank_top_n = None,
            rerank_min_score = None,
            rerank_model_name = "BAAI/bge-reranker-base",
        )
        retrieval_s = (time.perf_counter() - ret_start)
        context = "\n".join(r.get("text", "") for r in results)
        gen_start = time.perf_counter()
        answer = rag.generate(req.question, context, max_tokens = 1000)
        gen_s = (time.perf_counter() - gen_start)
        total_s = (time.perf_counter() - req_start)

        try:
            ts = datetime.utcnow().isoformat() + "Z"
            lines = [
                "-------- QUERY ---------",
                f"timestamp: {ts}",
                f"embedding_model: {embedding_model}",
                f"question: {req.question}",
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
                sim = r.get("similarity_score", 0.0)
                lines.append(f"  {idx}) source={src} sim={sim}")
            lines.append("------------------------")
            log_block(lines)
        except Exception:
            pass
        
        def generate(answer):
            yield f"data: {json.dumps({'retrieved': results})}\n\n"
            for chunk in answer: 
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        return StreamingResponse(generate(answer), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))