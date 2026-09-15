"""Build a HybridRetriever's indexes from a Corpus."""
from __future__ import annotations

from context_agent.data.corpus import Corpus
from context_agent.retrieval.bm25 import BM25Index
from context_agent.retrieval.dense import build_embedder
from context_agent.retrieval.hybrid import HybridRetriever, RetrievalConfig
from context_agent.retrieval.qdrant_store import QdrantVectorStore
from context_agent.retrieval.reranker import BGE_RERANKER_MODEL, CrossEncoderReranker


def build_retriever(
    corpus: Corpus,
    config: RetrievalConfig,
    dense_model: str = "hashing",
    reranker_model: str = BGE_RERANKER_MODEL,
    qdrant_path: str | None = None,
) -> HybridRetriever:
    bm25_index = BM25Index(corpus.all_chunks()) if config.use_bm25 else None

    dense_embedder = None
    vector_store = None
    if config.use_dense:
        chunks = corpus.all_chunks()
        dense_embedder = build_embedder(dense_model)
        if chunks:
            vectors = dense_embedder.embed([c.text for c in chunks])
            vector_store = QdrantVectorStore(dim=vectors.shape[1], path=qdrant_path)
            vector_store.upsert(
                chunk_ids=[c.chunk_id for c in chunks],
                vectors=vectors,
                payloads=[{"document_id": c.document_id} for c in chunks],
            )

    reranker = CrossEncoderReranker(model_name=reranker_model) if config.use_reranker else None

    return HybridRetriever(
        corpus=corpus,
        config=config,
        bm25_index=bm25_index,
        dense_embedder=dense_embedder,
        vector_store=vector_store,
        reranker=reranker,
    )
