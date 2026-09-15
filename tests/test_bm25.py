from context_agent.data.corpus import Chunk
from context_agent.retrieval.bm25 import BM25Index


def test_bm25_ranks_matching_chunk_first():
    chunks = [
        Chunk(chunk_id="a", document_id="a", chunk_index=0, text="the cat sat on the mat"),
        Chunk(chunk_id="b", document_id="b", chunk_index=0, text="rockets launch into orbit around the earth"),
        Chunk(chunk_id="c", document_id="c", chunk_index=0, text="a cat and a dog played in the yard"),
    ]
    index = BM25Index(chunks)

    hits = index.search("cat", top_k=3)
    hit_ids = [chunk_id for chunk_id, _ in hits]

    assert hit_ids[0] in {"a", "c"}
    assert "b" not in hit_ids


def test_bm25_empty_index_returns_no_hits():
    index = BM25Index([])
    assert index.search("anything") == []


def test_bm25_no_match_returns_empty():
    chunks = [Chunk(chunk_id="a", document_id="a", chunk_index=0, text="completely unrelated content")]
    index = BM25Index(chunks)
    assert index.search("zzzznonexistentzzzz") == []
