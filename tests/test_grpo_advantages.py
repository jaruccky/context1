from context_agent.training.grpo import group_advantages


def test_higher_reward_gets_positive_advantage():
    advantages = group_advantages([0.0, 1.0])
    assert advantages[0] < 0.0
    assert advantages[1] > 0.0


def test_advantages_are_centered_around_zero():
    advantages = group_advantages([0.2, 0.5, 0.9, 1.4])
    assert abs(sum(advantages)) < 1e-6


def test_single_sample_group_has_no_signal():
    assert group_advantages([0.7]) == [0.0]


def test_empty_group_returns_empty():
    assert group_advantages([]) == []


def test_identical_rewards_have_no_signal():
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_advantage_magnitude_is_scale_invariant():
    small_gap = group_advantages([0.0, 0.1])
    large_gap = group_advantages([0.0, 10.0])
    assert abs(small_gap[1] - large_gap[1]) < 1e-2  # normalized by the group's own std
