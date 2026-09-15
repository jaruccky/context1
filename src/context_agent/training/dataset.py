"""Convert logged trajectories into (state, action) SFT examples.

Each recorded step carries the exact `context_snapshot` (the model-visible context
that was rendered right before the action was generated), so we can reconstruct the
real (prompt, completion) pair without replaying retrieval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from context_agent.agent.actions import TOOL_NAMES
from context_agent.agent.prompts import SYSTEM_PROMPT, render_user_turn


def trajectory_to_sft_examples(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    """trajectory: the JSON-logged dict form of a TrajectoryState."""
    examples: list[dict[str, str]] = []
    for step in trajectory.get("steps", []):
        snapshot = step.get("context_snapshot")
        action = step.get("action")
        if not snapshot or not action:
            continue
        examples.append(
            {
                "system": SYSTEM_PROMPT,
                "prompt": render_user_turn(snapshot, list(TOOL_NAMES)),
                "completion": json.dumps(action, ensure_ascii=False),
            }
        )
    return examples


def trajectories_to_sft_dataset(trajectories: list[dict[str, Any]]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for trajectory in trajectories:
        examples.extend(trajectory_to_sft_examples(trajectory))
    return examples


def save_sft_jsonl(examples: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_trajectories_jsonl(path: str | Path) -> list[dict[str, Any]]:
    trajectories = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories
