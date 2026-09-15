"""Retrieval, agent, and answer metrics."""
from __future__ import annotations

import collections
import re
import string
from typing import Any

from context_agent.data.benchmark import BenchmarkTask
from context_agent.evidence.models import EvidencePack

# -- retrieval metrics ---------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 1.0
    gold = set(gold_ids)
    hit = len(gold & set(retrieved_ids))
    return hit / len(gold)


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    gold = set(gold_ids)
    hit = len(gold & set(retrieved_ids))
    return hit / len(retrieved_ids)


def supporting_document_recall(evidence_pack: EvidencePack, task: BenchmarkTask) -> float:
    return recall_at_k(evidence_pack.document_ids(), task.supporting_document_ids)


def supporting_fact_recall(evidence_pack: EvidencePack, task: BenchmarkTask) -> float:
    """Fraction of gold supporting facts whose document made it into the evidence pack.

    This is a document-level approximation of sentence-level supporting-fact recall:
    we track which documents evidence came from, not which exact sentence, so a fact
    counts as "covered" when its source document is present in the pack.
    """
    if not task.supporting_facts:
        return 1.0
    covered_docs = set(evidence_pack.document_ids())
    hit = sum(1 for sf in task.supporting_facts if sf.document_id in covered_docs)
    return hit / len(task.supporting_facts)


def retrieval_metrics(evidence_pack: EvidencePack, task: BenchmarkTask) -> dict[str, float]:
    retrieved_ids = evidence_pack.document_ids()
    gold_ids = task.supporting_document_ids
    return {
        "recall_at_k": recall_at_k(retrieved_ids, gold_ids),
        "precision_at_k": precision_at_k(retrieved_ids, gold_ids),
        "supporting_document_recall": supporting_document_recall(evidence_pack, task),
        "supporting_fact_recall": supporting_fact_recall(evidence_pack, task),
    }


# -- agent metrics --------------------------------------------------------------------


def agent_metrics(trajectory_summary: dict[str, Any], stop_reason: str) -> dict[str, Any]:
    tool_calls = trajectory_summary.get("tool_call_count", 0)
    invalid = trajectory_summary.get("invalid_action_count", 0)
    attempted = tool_calls + invalid
    tool_validity_rate = tool_calls / attempted if attempted else 1.0
    return {
        "steps": trajectory_summary.get("steps", 0),
        "tool_calls": tool_calls,
        "invalid_actions": invalid,
        "tool_validity_rate": tool_validity_rate,
        "duplicate_calls": trajectory_summary.get("duplicate_call_count", 0),
        "retrieved_chunks": trajectory_summary.get("seen_chunks", 0),
        "pruned_chunks": trajectory_summary.get("pruned_chunks", 0),
        "context_tokens": trajectory_summary.get("estimated_tokens", 0),
        "latency_ms": trajectory_summary.get("total_latency_ms", 0.0),
        "stop_reason": stop_reason,
        "success": stop_reason == "finished",
    }


# -- answer metrics ---------------------------------------------------------------


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    return {"exact_match": exact_match(prediction, gold), "token_f1": token_f1(prediction, gold)}


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of every numeric field across a list of per-task metric dicts."""
    if not rows:
        return {}
    keys = {k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {k: sum(row.get(k, 0.0) for row in rows) / len(rows) for k in keys}
