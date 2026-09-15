"""Hybrid retrieval: BM25 + dense -> RRF fusion -> optional reranker -> top-k.

Every stage is independently toggleable so the same code path supports the required
ablations (BM25-only, dense-only, hybrid, with/without reranker).
"""
from __future__ import annotations

from dataclasses import dataclass

from context_agent.data.corpus import Corpus
from context_agent.retrieval.base import SearchResult
from context_agent.retrieval.bm25 import BM25Index
from context_agent.retrieval.dense import DenseEmbedder
from context_agent.retrieval.fusion import reciprocal_rank_fusion
from context_agent.retrieval.qdrant_store import QdrantVectorStore
from context_agent.retrieval.reranker import Reranker


@dataclass
class RetrievalConfig:
    use_bm25: bool = True
    use_dense: bool = True
    use_reranker: bool = False
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 20
    final_top_k: int = 5


class HybridRetriever:
    def __init__(
        self,
        corpus: Corpus,
        config: RetrievalConfig,
        bm25_index: BM25Index | None = None,
        dense_embedder: DenseEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        reranker: Reranker | None = None,
    ):
        self.corpus = corpus
        self.config = config
        self.bm25_index = bm25_index
        self.dense_embedder = dense_embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        top_k = top_k or self.config.final_top_k
        rankings: list[list[str]] = []

        if self.config.use_bm25 and self.bm25_index is not None:
            bm25_hits = self.bm25_index.search(query, top_k=self.config.bm25_top_k)
            rankings.append([cid for cid, _ in bm25_hits])

        if self.config.use_dense and self.dense_embedder is not None and self.vector_store is not None:
            query_vector = self.dense_embedder.embed([query])[0]
            dense_hits = self.vector_store.search(query_vector, top_k=self.config.dense_top_k)
            rankings.append([cid for cid, _ in dense_hits])

        if not rankings:
            raise ValueError("HybridRetriever has no retrieval stage enabled")

        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k)
        fused_scores = dict(fused)
        candidate_ids = [cid for cid, _ in fused][: self.config.rerank_candidates]

        if self.config.use_reranker and self.reranker is not None and candidate_ids:
            candidates = [
                (cid, chunk.text)
                for cid in candidate_ids
                if (chunk := self.corpus.get_chunk(cid)) is not None
            ]
            final = self.reranker.rerank(query, candidates)[:top_k]
            source = "reranker"
        else:
            final = [(cid, fused_scores[cid]) for cid in candidate_ids[:top_k]]
            source = "fusion" if len(rankings) > 1 else "bm25"

        results: list[SearchResult] = []
        for rank, (chunk_id, score) in enumerate(final, start=1):
            chunk = self.corpus.get_chunk(chunk_id)
            if chunk is None:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=score,
                    rank=rank,
                    source=source,
                )
            )
        return results
