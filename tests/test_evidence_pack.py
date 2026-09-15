from context_agent.evidence.models import EvidenceItem, EvidencePack


def test_document_ids_deduplicates_preserving_order():
    pack = EvidencePack(
        query="q",
        evidence=[
            EvidenceItem(chunk_id="c1", document_id="d1", text="a"),
            EvidenceItem(chunk_id="c2", document_id="d2", text="b"),
            EvidenceItem(chunk_id="c3", document_id="d1", text="c"),
        ],
    )
    assert pack.document_ids() == ["d1", "d2"]


def test_chunk_ids_preserves_order():
    pack = EvidencePack(
        query="q",
        evidence=[
            EvidenceItem(chunk_id="c1", document_id="d1", text="a"),
            EvidenceItem(chunk_id="c2", document_id="d2", text="b"),
        ],
    )
    assert pack.chunk_ids() == ["c1", "c2"]


def test_render_text_includes_query_evidence_and_unresolved():
    pack = EvidencePack(
        query="What is X?",
        evidence=[EvidenceItem(chunk_id="c1", document_id="d1", text="X is a thing")],
        unresolved_questions=["what caused X?"],
    )
    rendered = pack.render_text()
    assert "What is X?" in rendered
    assert "X is a thing" in rendered
    assert "what caused X?" in rendered


def test_empty_pack_round_trips_through_json():
    pack = EvidencePack(query="q")
    restored = EvidencePack.model_validate_json(pack.model_dump_json())
    assert restored.query == "q"
    assert restored.evidence == []
