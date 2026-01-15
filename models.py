from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    rerank_top_n: Optional[int] = Field(None, description="If set, keep only top N after reranking")
    rerank_min_score: Optional[float] = Field(None, description="If set, drop items with score below this threshold")
    rerank_model_name: Optional[str] = Field(None, description="CrossEncoder model name")
    embedding_model: str = "openai"  

class RetrievedDocument(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    similarity_score: Optional[float] = None

