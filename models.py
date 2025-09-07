from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    bulletin_department: Optional[str] = Field(None, description="Bulletin key, e.g., 'apma'")
    cab_department: Optional[str] = Field(None, description="CAB key, e.g., 'APMA'")
    top_k_bulletin: int = Field(5, ge=1, le=20)
    top_k_cab: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    retrieved: List[Dict[str, Any]]


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    context: List[str] = Field(default_factory=list)


class EvaluateResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
