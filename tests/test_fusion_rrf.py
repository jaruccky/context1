from context_agent.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_promotes_items_ranked_high_in_multiple_lists():
    bm25 = ["a", "b", "c", "d"]
    dense = ["b", "a", "d", "c"]

    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    fused_ids = [item_id for item_id, _ in fused]

    # a and b are top-2 in both lists, so they should fuse ahead of c and d
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_single_list_preserves_order():
    ranking = ["x", "y", "z"]
    fused = reciprocal_rank_fusion([ranking])
    assert [item_id for item_id, _ in fused] == ["x", "y", "z"]


def test_rrf_weights_favor_higher_weighted_list():
    list_a = ["a", "b"]
    list_b = ["b", "a"]
    fused = reciprocal_rank_fusion([list_a, list_b], weights=[10.0, 1.0])
    assert fused[0][0] == "a"


def test_rrf_mismatched_weights_raises():
    try:
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched weights")
