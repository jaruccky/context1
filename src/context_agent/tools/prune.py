"""prune_chunks tool: remove chunks from model-visible context (not from full trajectory)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from context_agent.context.manager import ContextManager

TOOL_NAME = "prune_chunks"


class PruneArguments(BaseModel):
    chunk_ids: list[str] = Field(min_length=1)


def execute(args: PruneArguments, context_manager: ContextManager, step: int) -> dict:
    result = context_manager.prune(args.chunk_ids)
    return {
        "tool": TOOL_NAME,
        "removed": result.removed,
        "not_found": result.not_found,
    }
