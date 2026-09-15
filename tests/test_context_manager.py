from context_agent.context.manager import ContextManager


def make_chunk(chunk_id, document_id="doc1", text="hello world", score=1.0, source="bm25"):
    return {"chunk_id": chunk_id, "document_id": document_id, "text": text, "score": score, "source": source}


def test_add_chunks_tracks_new_and_duplicates():
    cm = ContextManager(query="q", token_budget=100_000)
    result = cm.add_chunks([make_chunk("c1"), make_chunk("c2")], step=1)
    assert result.added == ["c1", "c2"]
    assert cm.visible_chunk_ids == ["c1", "c2"]

    result2 = cm.add_chunks([make_chunk("c1"), make_chunk("c3")], step=2)
    assert result2.duplicates == ["c1"]
    assert result2.added == ["c3"]
    assert set(cm.state.retrieved_chunks.keys()) == {"c1", "c2", "c3"}


def test_prune_removes_from_visible_but_not_from_trajectory():
    cm = ContextManager(query="q", token_budget=100_000)
    cm.add_chunks([make_chunk("c1"), make_chunk("c2")], step=1)

    result = cm.prune(["c1", "missing"])
    assert result.removed == ["c1"]
    assert result.not_found == ["missing"]
    assert cm.visible_chunk_ids == ["c2"]
    # full trajectory keeps everything ever retrieved
    assert "c1" in cm.state.retrieved_chunks
    assert "c1" in cm.state.discarded_chunk_ids


def test_resurfacing_after_prune():
    cm = ContextManager(query="q", token_budget=100_000)
    cm.add_chunks([make_chunk("c1")], step=1)
    cm.prune(["c1"])
    assert cm.visible_chunk_ids == []

    result = cm.add_chunks([make_chunk("c1")], step=2)
    assert result.resurfaced == ["c1"]
    assert result.added == []
    assert cm.visible_chunk_ids == ["c1"]


def test_render_only_shows_visible_chunks():
    cm = ContextManager(query="What is X?", token_budget=100_000)
    cm.add_chunks([make_chunk("c1", text="alpha content")], step=1)
    cm.add_chunks([make_chunk("c2", text="beta content")], step=2)
    cm.prune(["c1"])

    rendered = cm.render()
    assert "beta content" in rendered
    assert "alpha content" not in rendered
    assert "What is X?" in rendered


def test_stats_reflect_visible_seen_and_pruned():
    cm = ContextManager(query="q", token_budget=100_000)
    cm.add_chunks([make_chunk("c1"), make_chunk("c2")], step=1)
    cm.prune(["c1"])

    stats = cm.stats()
    assert stats["visible_chunks"] == 1
    assert stats["seen_chunks"] == 2
    assert stats["pruned_chunks"] == 1
    assert stats["estimated_tokens"] > 0


def test_token_budget_auto_prunes_lowest_score_first():
    cm = ContextManager(query="q", token_budget=70)
    long_text = "x" * 200
    cm.add_chunks(
        [
            make_chunk("weak", text=long_text, score=1.0),
            make_chunk("mid", text=long_text, score=2.0),
            make_chunk("strong", text=long_text, score=3.0),
        ],
        step=1,
    )

    # budget forced some eviction; the highest-scored chunk should survive
    assert "strong" in cm.visible_chunk_ids
    assert "weak" not in cm.visible_chunk_ids
    assert cm.state.retrieved_chunks.keys() >= {"weak", "mid", "strong"}
    assert "weak" in cm.state.discarded_chunk_ids


def test_to_evidence_pack_reflects_visible_only():
    cm = ContextManager(query="q", token_budget=100_000)
    cm.add_chunks([make_chunk("c1"), make_chunk("c2")], step=1)
    cm.prune(["c1"])
    cm.add_unresolved_question("who did what?")

    pack = cm.to_evidence_pack()
    assert [e.chunk_id for e in pack.evidence] == ["c2"]
    assert pack.unresolved_questions == ["who did what?"]
    assert pack.trajectory_summary["seen_chunks"] == 2
