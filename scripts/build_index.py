#!/usr/bin/env python
"""Build the retrieval indexes for the prepared corpus and sanity-check them.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import argparse

from context_agent.config import build_retriever_from_config, load_corpus, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/default.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--query", default=None, help="Optional sanity-check search query.")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    retrieval_cfg = load_yaml(args.retrieval_config)

    corpus = load_corpus(data_cfg)
    retriever = build_retriever_from_config(corpus, retrieval_cfg)

    print(f"Indexed {len(corpus.all_chunks())} chunks from {len(corpus.documents)} documents.")
    print(
        f"use_bm25={retrieval_cfg.get('use_bm25', True)} "
        f"use_dense={retrieval_cfg.get('use_dense', True)} "
        f"use_reranker={retrieval_cfg.get('use_reranker', False)}"
    )

    query = args.query or (next(iter(corpus.documents.values())).title if corpus.documents else None)
    if query:
        results = retriever.search(query, top_k=3)
        print(f"\nSanity search for {query!r}:")
        for r in results:
            print(f"  [{r.rank}] score={r.score:.4f} doc={r.document_id} :: {r.text[:80]}")


if __name__ == "__main__":
    main()
