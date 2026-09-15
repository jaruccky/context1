#!/usr/bin/env python
"""Run a full, saved evaluation over the benchmark.

Writes artifacts/runs/<run_id>/{config.json, trajectories.jsonl, metrics.json} --
trajectories.jsonl holds the full TrajectoryState for every agentic-baseline task, for
later debugging or SFT/GRPO dataset extraction (see training/dataset.py).

Usage:
    python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from context_agent.utils.ids import new_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-config", default="configs/evaluation.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--run-id", default=None)
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

    baselines = eval_cfg.get("baselines", ["direct", "one_shot_rag", "long_context_rag", "agentic"])
    summaries: dict[str, dict] = {}
    trajectories: list[dict] = []

    for name in baselines:
        rows = evaluate_baseline(
            name,
            tasks,
            answer_generator,
            retriever=retriever,
            harness_factory=harness_factory if name == "agentic" else None,
            top_k=eval_cfg.get("one_shot_top_k", 5),
            long_context_top_k=eval_cfg.get("long_context_top_k", 20),
        )
        summaries[name] = summarize(rows)
        for row in rows:
            if row.trajectory is not None:
                trajectories.append(row.trajectory)
        print(f"{name}: {json.dumps(summaries[name])}")

    run_id = args.run_id or new_run_id()
    out_dir = Path(eval_cfg.get("output_dir", "artifacts/runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(
        json.dumps(
            {"evaluation": eval_cfg, "agent": agent_cfg, "retrieval": retrieval_cfg, "n_tasks": len(tasks)},
            indent=2,
        ),
        encoding="utf-8",
    )
    with open(out_dir / "trajectories.jsonl", "w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(f"\nSaved run artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
