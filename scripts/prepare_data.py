#!/usr/bin/env python
"""Build the local corpus + benchmark used by the rest of Context-1.

Usage:
    python scripts/prepare_data.py                     # synthetic, offline, always works
    python scripts/prepare_data.py --hotpotqa -n 40     # small real HotpotQA slice (needs network)
"""
from __future__ import annotations

import argparse

from context_agent.data.benchmark import build_hotpotqa_subset, load_synthetic_benchmark, save_tasks_jsonl
from context_agent.data.corpus import Corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hotpotqa", action="store_true", help="Pull a small HotpotQA slice instead of the synthetic fixture."
    )
    parser.add_argument("-n", "--n-examples", type=int, default=40)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--corpus-out", default="data/corpus/synthetic.jsonl")
    parser.add_argument("--benchmark-out", default="data/benchmark/synthetic.jsonl")
    args = parser.parse_args()

    if args.hotpotqa:
        documents, tasks = build_hotpotqa_subset(n_examples=args.n_examples, split=args.split)
    else:
        documents, tasks = load_synthetic_benchmark()

    corpus = Corpus(documents)
    corpus.to_jsonl(args.corpus_out)
    save_tasks_jsonl(tasks, args.benchmark_out)

    print(f"Wrote {len(documents)} documents -> {args.corpus_out}")
    print(f"Wrote {len(tasks)} benchmark tasks -> {args.benchmark_out}")


if __name__ == "__main__":
    main()
