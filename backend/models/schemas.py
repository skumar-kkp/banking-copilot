from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    query: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    confidence_level: str
    compliance_status: str
    sources: List[str] = []
    trace: Dict[str, Any] = {}

class ChunkGrade(BaseModel):
    is_relevant: bool
    score: float
    reason: str

class RegulatoryCheck(BaseModel):
    is_compliant: bool
    reason: str
    disclaimer_injected: bool
