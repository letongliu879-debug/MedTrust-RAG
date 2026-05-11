"""Pydantic schemas for MedTrust-RAG API"""

from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    department: Optional[str] = None


class Citation(BaseModel):
    text: str
    department: Optional[str] = None
    source_id: Optional[str] = None


class SafetyInfo(BaseModel):
    risk_level: str
    flagged_segments: list[str] = []
    suggestions: list[str] = []
    contradictions: list[str] = []


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    model_used: str
    safety: SafetyInfo
    citations: list[Citation] = []
    trace: Optional[dict] = None


class ModelInfo(BaseModel):
    name: str


class DepartmentInfo(BaseModel):
    name: str


class HealthResponse(BaseModel):
    status: str
    bm25_ready: bool
    available_models: list[str] = []
