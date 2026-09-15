"""Run a benchmark of tasks through one of the four baselines and collect metrics."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from context_agent.agent.harness import AgentHarness
from context_agent.answering.generator import AnswerGenerator
from context_agent.data.benchmark import BenchmarkTask
from context_agent.evaluation.baselines import (
    baseline_agentic,
    baseline_direct,
    baseline_long_context_rag,
    baseline_one_shot_rag,
)
from context_agent.evaluation.metrics import agent_metrics, aggregate_metrics, answer_metrics, retrieval_metrics
from context_agent.retrieval.hybrid import HybridRetriever


@dataclass
class EvaluationRow:
    task_id: str
    baseline: str
    question: str
    prediction: str
    retrieval: dict[str, float]
    answer: dict[str, float]
    stats: dict[str, Any]
    agent: dict[str, Any] | None = None
    trajectory: dict[str, Any] | None = None


def evaluate_baseline(
    name: str,
    tasks: list[BenchmarkTask],
    answer_generator: AnswerGenerator,
    retriever: HybridRetriever | None = None,
    harness_factory: Callable[[str], AgentHarness] | None = None,
    top_k: int = 5,
    long_context_top_k: int = 20,
) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for task in tasks:
        if name == "direct":
            result = baseline_direct(task.question, answer_generator)
        elif name == "one_shot_rag":
            assert retriever is not None, "one_shot_rag requires a retriever"
            result = baseline_one_shot_rag(task.question, retriever, answer_generator, top_k=top_k)
        elif name == "long_context_rag":
            assert retriever is not None, "long_context_rag requires a retriever"
            result = baseline_long_context_rag(
                task.question, retriever, answer_generator, top_k=long_context_top_k
            )
        elif name == "agentic":
            assert harness_factory is not None, "agentic requires a harness_factory"
            result = baseline_agentic(
                task.question, harness_factory(task.question), answer_generator, task_id=task.task_id
            )
        else:
            raise ValueError(f"unknown baseline: {name}")

        agent = None
        if name == "agentic":
            agent = agent_metrics(result.evidence_pack.trajectory_summary, result.stats["stop_reason"])

        rows.append(
            EvaluationRow(
                task_id=task.task_id,
                baseline=name,
                question=task.question,
                prediction=result.answer,
                retrieval=retrieval_metrics(result.evidence_pack, task),
                answer=answer_metrics(result.answer, task.answer),
                stats=result.stats,
                agent=agent,
                trajectory=result.trajectory,
            )
        )
    return rows


def summarize(rows: list[EvaluationRow]) -> dict[str, Any]:
    agent_rows = [r.agent for r in rows if r.agent is not None]
    return {
        "n_tasks": len(rows),
        "retrieval": aggregate_metrics([r.retrieval for r in rows]),
        "answer": aggregate_metrics([r.answer for r in rows]),
        "agent": aggregate_metrics(agent_rows) if agent_rows else {},
    }
