"""Reciprocal Rank Fusion of multiple ranked lists."""
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse several ranked lists of ids into one, using RRF.

    score(id) = sum_over_lists( weight_i / (k + rank_in_list_i) )
    Ids absent from a list simply don't contribute from that list.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match number of rankings")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda x: -x[1])
