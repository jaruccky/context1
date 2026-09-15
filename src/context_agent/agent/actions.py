"""Structured agent actions and their JSON parsing.

The policy always emits a JSON object shaped like an AgentAction. Parsing never
raises: malformed JSON, unknown tools, or a missing `thought` all turn into a typed
ActionParseError so the harness can turn it into an observation and keep going.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

TOOL_NAMES = ("search_corpus", "grep_corpus", "read_document", "prune_chunks")


class ToolName(str, Enum):
    SEARCH_CORPUS = "search_corpus"
    GREP_CORPUS = "grep_corpus"
    READ_DOCUMENT = "read_document"
    PRUNE_CHUNKS = "prune_chunks"


class AgentAction(BaseModel):
    thought: str = ""
    tool: ToolName | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    finish: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not self.finish and self.tool is None:
            raise ValueError("action must either set finish=true or specify a tool")


class ActionParseError(BaseModel):
    raw: str
    error: str


def parse_action(raw: str) -> AgentAction | ActionParseError:
    """Parse a raw model completion string into an AgentAction.

    Tolerates:
    - text wrapped around a single JSON object (extracts the first `{...}` span)
    - invalid JSON -> ActionParseError
    - unknown tool name -> ActionParseError
    - missing/mistyped fields -> ActionParseError (via pydantic ValidationError)
    """
    payload = _extract_json_object(raw)
    if payload is None:
        return ActionParseError(raw=raw, error="no JSON object found in model output")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return ActionParseError(raw=raw, error=f"invalid JSON: {exc}")

    if not isinstance(data, dict):
        return ActionParseError(raw=raw, error="JSON payload is not an object")

    tool = data.get("tool")
    if tool is not None and tool not in TOOL_NAMES:
        return ActionParseError(raw=raw, error=f"unknown tool: {tool!r}")

    try:
        return AgentAction.model_validate(data)
    except (ValidationError, ValueError) as exc:
        return ActionParseError(raw=raw, error=f"schema validation failed: {exc}")


def _extract_json_object(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None
