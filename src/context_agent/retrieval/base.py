from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    source: str = "hybrid"
    metadata: dict[str, Any] = Field(default_factory=dict)


def simple_tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())
