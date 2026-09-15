"""ContextManager: the agent's model-visible working context.

Owns a TrajectoryState (the full, append-only record) plus a `visible_chunk_ids`
list — an ordered view of which retrieved chunks the agent currently "sees" when its
context is rendered. `prune_chunks` mutates only `visible_chunk_ids`; the underlying
TrajectoryState.retrieved_chunks record is untouched, so nothing is ever truly lost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from context_agent.context.state import TrajectoryState
from context_agent.evidence.models import EvidenceItem, EvidencePack
from context_agent.utils.tokens import estimate_tokens


@dataclass
class AddChunksResult:
    added: list[str] = field(default_factory=list)
    resurfaced: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)


@dataclass
class PruneResult:
    removed: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)


class ContextManager:
    def __init__(
        self,
        query: str,
        task_id: str | None = None,
        token_budget: int = 4000,
        recent_observations: int = 3,
    ):
        self.state = TrajectoryState(query=query, task_id=task_id)
        self.visible_chunk_ids: list[str] = []
        self.unresolved_questions: list[str] = []
        self.constraints: list[str] = []
        self.token_budget = token_budget
        self.recent_observations = recent_observations
        self.auto_pruned_chunk_ids: list[str] = []

    # -- ingesting retrieval results ------------------------------------------------

    def add_chunks(self, chunks: list[dict[str, Any]], step: int) -> AddChunksResult:
        """chunks: list of dicts with chunk_id, document_id, text, score, source."""
        result = AddChunksResult()
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            was_seen = chunk_id in self.state.retrieved_chunks
            self.state.record_chunk(
                chunk_id=chunk_id,
                document_id=chunk["document_id"],
                text=chunk["text"],
                score=chunk.get("score"),
                source=chunk.get("source"),
                step=step,
            )
            if not was_seen:
                result.added.append(chunk_id)
                self.visible_chunk_ids.append(chunk_id)
            elif chunk_id not in self.visible_chunk_ids:
                result.resurfaced.append(chunk_id)
                self.visible_chunk_ids.append(chunk_id)
            else:
                result.duplicates.append(chunk_id)
        self._enforce_budget()
        return result

    # -- pruning ----------------------------------------------------------------

    def prune(self, chunk_ids: list[str]) -> PruneResult:
        result = PruneResult()
        for chunk_id in chunk_ids:
            if chunk_id in self.visible_chunk_ids:
                self.visible_chunk_ids.remove(chunk_id)
                result.removed.append(chunk_id)
            else:
                result.not_found.append(chunk_id)
        self.state.record_discard(result.removed)
        return result

    def _enforce_budget(self) -> list[str]:
        """Drop lowest-score visible chunks until under budget. Never touches full trajectory."""
        dropped: list[str] = []
        while self.visible_chunk_ids and estimate_tokens(self.render()) > self.token_budget:
            ranked = sorted(
                self.visible_chunk_ids,
                key=lambda cid: (self.state.retrieved_chunks[cid].score or 0.0),
            )
            weakest = ranked[0]
            self.visible_chunk_ids.remove(weakest)
            dropped.append(weakest)
        if dropped:
            self.state.record_discard(dropped)
            self.auto_pruned_chunk_ids.extend(dropped)
        return dropped

    # -- unresolved sub-questions -------------------------------------------------

    def add_unresolved_question(self, question: str) -> None:
        if question not in self.unresolved_questions:
            self.unresolved_questions.append(question)

    def resolve_question(self, question: str) -> None:
        if question in self.unresolved_questions:
            self.unresolved_questions.remove(question)

    # -- rendering ----------------------------------------------------------------

    def render(self) -> str:
        lines = [f"Question: {self.state.query}", ""]
        if self.constraints:
            lines.append("Constraints:")
            lines.extend(f"- {c}" for c in self.constraints)
            lines.append("")
        lines.append("Current evidence:")
        if self.visible_chunk_ids:
            for chunk_id in self.visible_chunk_ids:
                rec = self.state.retrieved_chunks[chunk_id]
                lines.append(f"- [{chunk_id}] ({rec.document_id}) {rec.text}")
        else:
            lines.append("(none yet)")
        if self.unresolved_questions:
            lines.append("")
            lines.append("Unresolved sub-questions:")
            lines.extend(f"- {q}" for q in self.unresolved_questions)
        recent = self.state.steps[-self.recent_observations :]
        if recent:
            lines.append("")
            lines.append("Recent observations:")
            for step in recent:
                obs = json.dumps(step.observation, default=str)[:200]
                lines.append(f"- step {step.step}: {obs}")
        return "\n".join(lines)

    # -- stats + evidence pack ------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "visible_chunks": len(self.visible_chunk_ids),
            "seen_chunks": len(self.state.retrieved_chunks),
            "pruned_chunks": len(self.state.discarded_chunk_ids),
            "estimated_tokens": estimate_tokens(self.render()),
        }

    def to_evidence_pack(self) -> EvidencePack:
        evidence = [
            EvidenceItem(
                chunk_id=chunk_id,
                document_id=rec.document_id,
                text=rec.text,
                score=rec.score,
                source=rec.source,
            )
            for chunk_id in self.visible_chunk_ids
            for rec in [self.state.retrieved_chunks[chunk_id]]
        ]
        return EvidencePack(
            query=self.state.query,
            evidence=evidence,
            unresolved_questions=list(self.unresolved_questions),
            trajectory_summary={
                "steps": len(self.state.steps),
                "tool_call_count": self.state.tool_call_count,
                "duplicate_call_count": self.state.duplicate_call_count,
                "invalid_action_count": self.state.invalid_action_count,
                "error_count": self.state.error_count,
                "total_latency_ms": self.state.total_latency_ms,
                **self.stats(),
            },
        )
