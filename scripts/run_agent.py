#!/usr/bin/env python
"""Run Context-1 on a single question and print the resulting EvidencePack.

Usage:
    python scripts/run_agent.py --query "Who designed the machine that the first
        computer programmer wrote notes about, and in what city was that person born?"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--data-config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--answer", action="store_true", help="Also run the downstream AnswerGenerator.")
    parser.add_argument("--save-trajectory", default=None, help="Optional path to save the full trajectory JSON.")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    retrieval_cfg = load_yaml(args.retrieval_config)
    agent_cfg = load_yaml(args.agent_config)

    corpus = load_corpus(data_cfg)
    retriever = build_retriever_from_config(corpus, retrieval_cfg)
    policy = build_policy(agent_cfg, query=args.query)
    harness = AgentHarness(policy, corpus, retriever, build_harness_config(agent_cfg))

    result = harness.run(args.query)
    pack = result.evidence_pack

    print(f"stop_reason: {result.stop_reason}")
    print(json.dumps(pack.model_dump(), indent=2, ensure_ascii=False))

    if args.answer:
        answer_generator = build_answer_generator({"answer_generator": "mock"})
        print("\nAnswer:", answer_generator.generate(pack))

    if args.save_trajectory:
        path = Path(args.save_trajectory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.context_manager.state.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nSaved full trajectory -> {path}")


if __name__ == "__main__":
    main()
