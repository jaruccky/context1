"""search_corpus tool: hybrid semantic + lexical search over the corpus."""
from __future__ import annotations

from pydantic import BaseModel, Field

from context_agent.context.manager import ContextManager
from context_agent.retrieval.hybrid import HybridRetriever

TOOL_NAME = "search_corpus"


class SearchArguments(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


def execute(
    args: SearchArguments, retriever: HybridRetriever, context_manager: ContextManager, step: int
) -> dict:
    results = retriever.search(args.query, top_k=args.top_k)
    add_result = context_manager.add_chunks(
        [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "score": r.score,
                "source": r.source,
            }
            for r in results
        ],
        step=step,
    )
    return {
        "tool": TOOL_NAME,
        "query": args.query,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "score": r.score,
                "snippet": r.text[:200],
            }
            for r in results
        ],
        "added": add_result.added,
        "resurfaced": add_result.resurfaced,
        "duplicates": add_result.duplicates,
    }
