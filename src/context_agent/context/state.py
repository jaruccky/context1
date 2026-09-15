"""TrajectoryState: the full-fidelity record of an agent run.

This is append-only and never pruned. `ContextManager` layers a smaller
"model-visible" view on top of it (see context/manager.py) — `prune_chunks` only
removes from that view, it never removes anything recorded here. This split is what
lets the agent manage its own working context while still leaving a complete record
for debugging, evaluation, and training-data extraction.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepRecord(BaseModel):
    step: int
    context_snapshot: str = ""
    action: dict[str, Any]
    observation: dict[str, Any]
    latency_ms: float
    error: str | None = None


class RetrievedChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float | None = None
    source: str | None = None
    first_seen_step: int


class TrajectoryState(BaseModel):
    query: str
    task_id: str | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    retrieved_chunks: dict[str, RetrievedChunkRecord] = Field(default_factory=dict)
    discarded_chunk_ids: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    duplicate_call_count: int = 0
    invalid_action_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0

    def add_step(
        self,
        step: int,
        action: dict[str, Any],
        observation: dict[str, Any],
        latency_ms: float,
        error: str | None = None,
        context_snapshot: str = "",
    ) -> None:
        self.steps.append(
            StepRecord(
                step=step,
                context_snapshot=context_snapshot,
                action=action,
                observation=observation,
                latency_ms=latency_ms,
                error=error,
            )
        )
        self.total_latency_ms += latency_ms
        if error:
            self.error_count += 1

    def record_chunk(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        score: float | None,
        source: str | None,
        step: int,
    ) -> bool:
        """Record a chunk as seen. Returns True if this is the first time we've seen it."""
        if chunk_id in self.retrieved_chunks:
            return False
        self.retrieved_chunks[chunk_id] = RetrievedChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            score=score,
            source=source,
            first_seen_step=step,
        )
        return True

    def record_discard(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            if chunk_id not in self.discarded_chunk_ids:
                self.discarded_chunk_ids.append(chunk_id)
