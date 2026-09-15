import pytest

from context_agent.data.benchmark import load_synthetic_benchmark
from context_agent.data.corpus import Corpus
from context_agent.retrieval.bm25 import BM25Index
from context_agent.retrieval.hybrid import HybridRetriever, RetrievalConfig


@pytest.fixture
def synthetic_corpus_and_tasks():
    documents, tasks = load_synthetic_benchmark()
    # one chunk per (short) synthetic document, keeps retrieval tests simple/predictable
    corpus = Corpus(documents, chunk_size=1000, overlap=0)
    return corpus, tasks


@pytest.fixture
def bm25_retriever(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    bm25_index = BM25Index(corpus.all_chunks())
    config = RetrievalConfig(use_bm25=True, use_dense=False, use_reranker=False, final_top_k=5)
    return HybridRetriever(corpus=corpus, config=config, bm25_index=bm25_index)
