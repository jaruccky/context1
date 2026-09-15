"""Cheap, dependency-free token estimation.

We don't want a tokenizer download just to budget context size, so we approximate
tokens as ~4 characters each (a common heuristic for English text). Good enough for
budgeting decisions; not meant to match any specific tokenizer exactly.
"""
from __future__ import annotations

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)
