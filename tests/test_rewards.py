from context_agent.training.rewards import compute_reward


def base_summary(**overrides):
    summary = {
        "tool_call_count": 2,
        "duplicate_call_count": 0,
        "estimated_tokens": 500,
        "invalid_action_count": 0,
    }
    summary.update(overrides)
    return summary


def test_perfect_efficient_success_scores_highest():
    reward = compute_reward(
        base_summary(), retrieved_document_ids=["d1", "d2"], gold_document_ids=["d1", "d2"], success=True
    )
    assert reward.evidence_recall == 1.0
    assert reward.success_bonus == 1.0
    assert reward.total > 1.5


def test_more_tool_calls_lowers_reward():
    cheap = compute_reward(base_summary(tool_call_count=2), ["d1"], ["d1"], success=True)
    expensive = compute_reward(base_summary(tool_call_count=20), ["d1"], ["d1"], success=True)
    assert expensive.total < cheap.total


def test_duplicate_calls_are_penalized():
    clean = compute_reward(base_summary(duplicate_call_count=0), ["d1"], ["d1"], success=True)
    dupey = compute_reward(base_summary(duplicate_call_count=5), ["d1"], ["d1"], success=True)
    assert dupey.total < clean.total


def test_invalid_actions_are_penalized_more_than_tool_calls():
    invalid = compute_reward(base_summary(invalid_action_count=3), ["d1"], ["d1"], success=True)
    valid = compute_reward(base_summary(invalid_action_count=0), ["d1"], ["d1"], success=True)
    assert invalid.total < valid.total
    assert invalid.invalid_action_penalty > 0


def test_failure_without_evidence_scores_lowest():
    reward = compute_reward(base_summary(), retrieved_document_ids=[], gold_document_ids=["d1"], success=False)
    assert reward.evidence_recall == 0.0
    assert reward.success_bonus == 0.0
    assert reward.total < 0
