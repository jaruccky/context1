"""Lexical retrieval via rank_bm25."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from context_agent.data.corpus import Chunk
from context_agent.retrieval.base import simple_tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [simple_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(simple_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: -x[1])
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]
