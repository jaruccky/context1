"""grep_corpus tool: exact word/phrase/regex search over the corpus.

Used once the agent already knows what it's looking for and semantic search is
overkill or too fuzzy — e.g. finding an exact quoted phrase or a specific term.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from context_agent.context.manager import ContextManager
from context_agent.data.corpus import Corpus
from context_agent.tools.errors import ToolError

TOOL_NAME = "grep_corpus"


class GrepArguments(BaseModel):
    pattern: str = Field(min_length=1)
    document_ids: list[str] | None = None
    regex: bool = False
    max_results: int = Field(default=10, ge=1, le=100)


def execute(args: GrepArguments, corpus: Corpus, context_manager: ContextManager, step: int) -> dict:
    if args.regex:
        try:
            compiled = re.compile(args.pattern, re.IGNORECASE)
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}") from exc
        matches = lambda text: compiled.search(text) is not None  # noqa: E731
    else:
        needle = args.pattern.lower()
        matches = lambda text: needle in text.lower()  # noqa: E731

    if args.document_ids:
        unknown = [d for d in args.document_ids if corpus.get_document(d) is None]
        if unknown:
            raise ToolError(f"unknown document_ids: {unknown}")
        chunk_pool = [c for d in args.document_ids for c in corpus.get_document_chunks(d)]
    else:
        chunk_pool = corpus.all_chunks()

    hits = [c for c in chunk_pool if matches(c.text)][: args.max_results]

    add_result = context_manager.add_chunks(
        [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "score": None,
                "source": "grep",
            }
            for c in hits
        ],
        step=step,
    )
    return {
        "tool": TOOL_NAME,
        "pattern": args.pattern,
        "results": [
            {"chunk_id": c.chunk_id, "document_id": c.document_id, "snippet": c.text[:200]}
            for c in hits
        ],
        "added": add_result.added,
        "resurfaced": add_result.resurfaced,
        "duplicates": add_result.duplicates,
    }
