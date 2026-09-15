from context_agent.data.benchmark import BenchmarkTask, SupportingFact
from context_agent.evaluation.metrics import (
    agent_metrics,
    aggregate_metrics,
    answer_metrics,
    exact_match,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
    supporting_document_recall,
    supporting_fact_recall,
    token_f1,
)
from context_agent.evidence.models import EvidenceItem, EvidencePack


def test_recall_and_precision_at_k():
    assert recall_at_k(["a", "b"], ["a", "b", "c"]) == 2 / 3
    assert precision_at_k(["a", "b", "x"], ["a", "b", "c"]) == 2 / 3
    assert recall_at_k([], []) == 1.0
    assert precision_at_k([], ["a"]) == 0.0


def _pack(doc_ids):
    return EvidencePack(
        query="q", evidence=[EvidenceItem(chunk_id=f"{d}::0", document_id=d, text="x") for d in doc_ids]
    )


def test_supporting_document_and_fact_recall():
    task = BenchmarkTask(
        task_id="t1",
        question="q",
        answer="a",
        supporting_document_ids=["d1", "d2"],
        supporting_facts=[SupportingFact(document_id="d1", sentence_index=0), SupportingFact(document_id="d2", sentence_index=1)],
    )
    pack = _pack(["d1"])
    assert supporting_document_recall(pack, task) == 0.5
    assert supporting_fact_recall(pack, task) == 0.5

    metrics = retrieval_metrics(pack, task)
    assert metrics["supporting_document_recall"] == 0.5


def test_exact_match_and_f1_normalize_text():
    assert exact_match("The Cat.", "the cat") == 1.0
    assert exact_match("a dog", "a cat") == 0.0
    assert token_f1("the quick brown fox", "quick brown fox jumps") > 0.5


def test_answer_metrics_bundle():
    metrics = answer_metrics("Paris", "paris")
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0


def test_agent_metrics_computes_validity_rate_and_success():
    summary = {
        "steps": 4,
        "tool_call_count": 3,
        "invalid_action_count": 1,
        "duplicate_call_count": 0,
        "seen_chunks": 5,
        "pruned_chunks": 2,
        "estimated_tokens": 800,
        "total_latency_ms": 120.0,
    }
    metrics = agent_metrics(summary, stop_reason="finished")
    assert metrics["tool_validity_rate"] == 0.75
    assert metrics["success"] is True
    assert metrics["stop_reason"] == "finished"


def test_aggregate_metrics_averages_numeric_fields():
    rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
    agg = aggregate_metrics(rows)
    assert agg["a"] == 2.0
    assert agg["b"] == 3.0


def test_aggregate_metrics_empty_returns_empty():
    assert aggregate_metrics([]) == {}
