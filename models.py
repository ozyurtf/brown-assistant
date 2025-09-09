from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    concentration: Optional[str] = Field(None, description="Concentration key, e.g., 'apma'")
    department: Optional[str] = Field(None, description="Department key, e.g., 'APMA'")
    top_k_concentration: int = Field(5, ge=1, le=20)
    top_k_department: int = Field(5, ge=1, le=20)
    rerank_top_n: Optional[int] = Field(None, description="If set, keep only top N after reranking")
    rerank_min_score: Optional[float] = Field(None, description="If set, drop items with score below this threshold")
    rerank_model_name: Optional[str] = Field(None, description="CrossEncoder model name")
    embedding_model: str = "sentence_transformer"  

class RetrievedDocument(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    similarity_score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedDocument]

class EvaluateRequest(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    context: List[str] = Field(default_factory=list)
    embedding_model: str = "sentence_transformer"
    force: bool = False

class EvaluateResponse(BaseModel):
    bleu_score: float = Field(..., ge=0.0, le=1.0)
    rouge_score: float = Field(..., ge=0.0, le=1.0)
    embedding_model: str = "sentence_transformer"  