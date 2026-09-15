"""Hybrid retrieval end-to-end: BM25 + dense (hashing embedder + embedded Qdrant) +
RRF + optional reranker. Uses only offline, dependency-free backends -- no model
download, no external Qdrant server -- so this runs in any CI environment.
"""
from context_agent.retrieval.dense import HashingEmbedder
from context_agent.retrieval.hybrid import HybridRetriever, RetrievalConfig
from context_agent.retrieval.qdrant_store import QdrantVectorStore
from context_agent.retrieval.reranker import IdentityReranker


def _build_hybrid_retriever(corpus):
    from context_agent.retrieval.bm25 import BM25Index

    chunks = corpus.all_chunks()
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed([c.text for c in chunks])
    store = QdrantVectorStore(dim=vectors.shape[1])
    store.upsert(
        chunk_ids=[c.chunk_id for c in chunks],
        vectors=vectors,
        payloads=[{"document_id": c.document_id} for c in chunks],
    )
    config = RetrievalConfig(use_bm25=True, use_dense=True, use_reranker=False, final_top_k=5)
    return HybridRetriever(
        corpus=corpus,
        config=config,
        bm25_index=BM25Index(chunks),
        dense_embedder=embedder,
        vector_store=store,
    )


def test_hybrid_search_returns_results(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    retriever = _build_hybrid_retriever(corpus)

    results = retriever.search("Ada Lovelace computer programmer", top_k=3)
    assert results
    assert results[0].source in {"fusion", "reranker"}
    assert any(r.document_id == "ada-lovelace" for r in results)


def test_bm25_only_ablation(bm25_retriever):
    results = bm25_retriever.search("Guido van Rossum Python", top_k=3)
    assert results
    assert all(r.source == "bm25" for r in results)


def test_reranker_ablation_changes_result_source(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    from context_agent.retrieval.bm25 import BM25Index

    config = RetrievalConfig(use_bm25=True, use_dense=False, use_reranker=True, final_top_k=3)
    retriever = HybridRetriever(
        corpus=corpus,
        config=config,
        bm25_index=BM25Index(corpus.all_chunks()),
        reranker=IdentityReranker(),
    )
    results = retriever.search("Turing Award ACM", top_k=3)
    assert results
    assert all(r.source == "reranker" for r in results)


def test_no_retrieval_stage_enabled_raises(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    config = RetrievalConfig(use_bm25=False, use_dense=False)
    retriever = HybridRetriever(corpus=corpus, config=config)
    try:
        retriever.search("anything")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no retrieval stage is enabled")
