"""EvidencePack: the hard boundary between Context-1 (retrieval) and any answer model."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    query: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    trajectory_summary: dict[str, Any] = Field(default_factory=dict)

    def document_ids(self) -> list[str]:
        seen: list[str] = []
        for item in self.evidence:
            if item.document_id not in seen:
                seen.append(item.document_id)
        return seen

    def chunk_ids(self) -> list[str]:
        return [item.chunk_id for item in self.evidence]

    def render_text(self) -> str:
        """Compact text rendering suitable for feeding to a downstream answer model."""
        lines = [f"Question: {self.query}", ""]
        for i, item in enumerate(self.evidence, start=1):
            lines.append(f"[{i}] ({item.document_id}) {item.text}")
        if self.unresolved_questions:
            lines.append("")
            lines.append("Unresolved sub-questions:")
            for q in self.unresolved_questions:
                lines.append(f"- {q}")
        return "\n".join(lines)
