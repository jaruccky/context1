"""Four comparable pipelines, all producing an EvidencePack + answer + stats.

baseline_1_direct         -- no retrieval at all.
baseline_2_one_shot_rag    -- single retrieval call, small top_k.
baseline_3_long_context_rag -- single retrieval call, large top_k.
baseline_4_agentic         -- Context-1: full agent harness (search/grep/read/prune).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from context_agent.agent.harness import AgentHarness
from context_agent.answering.generator import AnswerGenerator
from context_agent.evidence.models import EvidenceItem, EvidencePack
from context_agent.retrieval.hybrid import HybridRetriever
from context_agent.utils.tokens import estimate_tokens


@dataclass
class BaselineRunResult:
    name: str
    evidence_pack: EvidencePack
    answer: str
    stats: dict[str, Any] = field(default_factory=dict)
    trajectory: dict[str, Any] | None = None


def _pack_from_results(query: str, results) -> EvidencePack:
    evidence = [
        EvidenceItem(chunk_id=r.chunk_id, document_id=r.document_id, text=r.text, score=r.score, source=r.source)
        for r in results
    ]
    return EvidencePack(query=query, evidence=evidence)


def baseline_direct(query: str, answer_generator: AnswerGenerator) -> BaselineRunResult:
    pack = EvidencePack(query=query, evidence=[])
    answer = answer_generator.generate(pack)
    return BaselineRunResult(
        name="direct", evidence_pack=pack, answer=answer, stats={"tool_calls": 0, "steps": 0, "context_tokens": 0}
    )


def baseline_one_shot_rag(
    query: str, retriever: HybridRetriever, answer_generator: AnswerGenerator, top_k: int = 5
) -> BaselineRunResult:
    results = retriever.search(query, top_k=top_k)
    pack = _pack_from_results(query, results)
    answer = answer_generator.generate(pack)
    return BaselineRunResult(
        name="one_shot_rag",
        evidence_pack=pack,
        answer=answer,
        stats={"tool_calls": 1, "steps": 1, "context_tokens": estimate_tokens(pack.render_text())},
    )


def baseline_long_context_rag(
    query: str, retriever: HybridRetriever, answer_generator: AnswerGenerator, top_k: int = 20
) -> BaselineRunResult:
    results = retriever.search(query, top_k=top_k)
    pack = _pack_from_results(query, results)
    answer = answer_generator.generate(pack)
    return BaselineRunResult(
        name="long_context_rag",
        evidence_pack=pack,
        answer=answer,
        stats={"tool_calls": 1, "steps": 1, "context_tokens": estimate_tokens(pack.render_text())},
    )


def baseline_agentic(
    query: str, harness: AgentHarness, answer_generator: AnswerGenerator, task_id: str | None = None
) -> BaselineRunResult:
    result = harness.run(query, task_id=task_id)
    pack = result.evidence_pack
    answer = answer_generator.generate(pack)
    summary = pack.trajectory_summary
    stats = {
        "tool_calls": summary.get("tool_call_count", 0),
        "steps": summary.get("steps", 0),
        "context_tokens": summary.get("estimated_tokens", 0),
        "duplicate_calls": summary.get("duplicate_call_count", 0),
        "invalid_actions": summary.get("invalid_action_count", 0),
        "pruned_chunks": summary.get("pruned_chunks", 0),
        "stop_reason": result.stop_reason,
    }
    trajectory = result.context_manager.state.model_dump(mode="json")
    return BaselineRunResult(name="agentic", evidence_pack=pack, answer=answer, stats=stats, trajectory=trajectory)


BASELINE_NAMES = ("direct", "one_shot_rag", "long_context_rag", "agentic")
