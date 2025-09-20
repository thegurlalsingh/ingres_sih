# app/schemas.py
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class IngestResponse(BaseModel):
    success: bool
    inserted: int

class QueryRequest(BaseModel):
    query: str
    lang: Optional[str] = "en"
    top_k: Optional[int] = 5
    return_visual: Optional[bool] = False
    visualization_type: Optional[str] = "recharge_vs_extraction"  # example

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    image_url: Optional[str] = None
