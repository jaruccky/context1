"""Cross-encoder reranking.

Default model is a small CPU-friendly cross-encoder; set `BGE_RERANKER_MODEL` in
config to use `BAAI/bge-reranker-v2-m3` for the "production-like" setup. Not
exercised by the default test suite (would download weights). `IdentityReranker` is
the no-op used in ablations and in tests.
"""
from __future__ import annotations

from typing import Protocol

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip((cid for cid, _ in candidates), (float(s) for s in scores)), key=lambda x: -x[1]
        )
        return ranked


class IdentityReranker:
    """No-op reranker: preserves incoming order, used for ablations/tests."""

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        n = len(candidates)
        return [(cid, float(n - i)) for i, (cid, _) in enumerate(candidates)]
