#!/usr/bin/env python
"""Run all configured baselines on the benchmark and print a comparison table.

Fast, console-only -- for ablations and quick iteration. For a saved, artifact-backed
run (config + full trajectories + metrics.json) use scripts/evaluate.py instead.

Usage:
    python scripts/run_baselines.py
"""
from __future__ import annotations

import argparse

from context_agent.agent.harness import AgentHarness
from context_agent.config import (
    build_answer_generator,
    build_harness_config,
    build_policy,
    build_retriever_from_config,
    load_corpus,
    load_yaml,
)
from context_agent.data.benchmark import load_tasks_jsonl
from context_agent.evaluation.evaluator import evaluate_baseline, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-config", default="configs/evaluation.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    args = parser.parse_args()

    eval_cfg = load_yaml(args.evaluation_config)
    agent_cfg = load_yaml(args.agent_config)
    retrieval_cfg = load_yaml(args.retrieval_config)

    corpus = load_corpus(eval_cfg)
    tasks = load_tasks_jsonl(eval_cfg["benchmark_path"])
    retriever = build_retriever_from_config(corpus, retrieval_cfg)
    answer_generator = build_answer_generator(eval_cfg, llm_raw=agent_cfg.get("llm"))

    def harness_factory(query: str) -> AgentHarness:
        policy = build_policy(agent_cfg, query=query)
        return AgentHarness(policy, corpus, retriever, build_harness_config(agent_cfg))

    header = f"{'baseline':<18}{'doc_recall':<12}{'sf_recall':<12}{'EM':<8}{'F1':<8}{'tool_calls':<12}{'ctx_tokens':<12}"
    print(header)
    print("-" * len(header))

    for name in eval_cfg.get("baselines", ["direct", "one_shot_rag", "long_context_rag", "agentic"]):
        rows = evaluate_baseline(
            name,
            tasks,
            answer_generator,
            retriever=retriever,
            harness_factory=harness_factory if name == "agentic" else None,
            top_k=eval_cfg.get("one_shot_top_k", 5),
            long_context_top_k=eval_cfg.get("long_context_top_k", 20),
        )
        summary = summarize(rows)
        r, a = summary["retrieval"], summary["answer"]
        avg_tool_calls = sum(row.stats.get("tool_calls", 0) for row in rows) / len(rows) if rows else 0.0
        avg_tokens = sum(row.stats.get("context_tokens", 0) for row in rows) / len(rows) if rows else 0.0
        print(
            f"{name:<18}{r.get('supporting_document_recall', 0):<12.3f}"
            f"{r.get('supporting_fact_recall', 0):<12.3f}{a.get('exact_match', 0):<8.3f}"
            f"{a.get('token_f1', 0):<8.3f}{avg_tool_calls:<12.2f}{avg_tokens:<12.1f}"
        )


if __name__ == "__main__":
    main()
