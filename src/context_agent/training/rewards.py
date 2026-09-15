"""Reward function for future GRPO training.

reward = evidence_recall + success_bonus
         - alpha * (tool_calls + duplicate_calls)
         - beta * context_tokens
         - gamma * invalid_actions

Pure and independently testable: takes plain data (trajectory summary dict + gold/
retrieved document ids), not a live TrajectoryState, so it can be unit tested with
synthetic inputs and reused offline over logged trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from context_agent.evaluation.metrics import recall_at_k


@dataclass
class RewardComponents:
    evidence_recall: float
    success_bonus: float
    tool_call_penalty: float
    context_token_penalty: float
    invalid_action_penalty: float
    total: float


def compute_reward(
    trajectory_summary: dict[str, Any],
    retrieved_document_ids: list[str],
    gold_document_ids: list[str],
    success: bool,
    alpha: float = 0.02,
    beta: float = 0.0005,
    gamma: float = 0.1,
) -> RewardComponents:
    evidence_recall = recall_at_k(retrieved_document_ids, gold_document_ids)
    success_bonus = 1.0 if success else 0.0

    tool_calls = trajectory_summary.get("tool_call_count", 0)
    duplicate_calls = trajectory_summary.get("duplicate_call_count", 0)
    context_tokens = trajectory_summary.get("estimated_tokens", 0)
    invalid_actions = trajectory_summary.get("invalid_action_count", 0)

    tool_call_penalty = alpha * (tool_calls + duplicate_calls)
    context_token_penalty = beta * context_tokens
    invalid_action_penalty = gamma * invalid_actions

    total = (
        evidence_recall
        + success_bonus
        - tool_call_penalty
        - context_token_penalty
        - invalid_action_penalty
    )

    return RewardComponents(
        evidence_recall=evidence_recall,
        success_bonus=success_bonus,
        tool_call_penalty=tool_call_penalty,
        context_token_penalty=context_token_penalty,
        invalid_action_penalty=invalid_action_penalty,
        total=total,
    )
