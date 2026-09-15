"""read_document tool: read a document's chunks, or a specific chunk range."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from context_agent.context.manager import ContextManager
from context_agent.data.corpus import Corpus
from context_agent.tools.errors import ToolError

TOOL_NAME = "read_document"


class ReadArguments(BaseModel):
    document_id: str = Field(min_length=1)
    start_chunk: int = Field(default=0, ge=0)
    end_chunk: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> "ReadArguments":
        if self.end_chunk is not None and self.end_chunk < self.start_chunk:
            raise ValueError("end_chunk must be >= start_chunk")
        return self


def execute(args: ReadArguments, corpus: Corpus, context_manager: ContextManager, step: int) -> dict:
    document = corpus.get_document(args.document_id)
    if document is None:
        raise ToolError(f"unknown document_id: {args.document_id}")

    doc_chunks = corpus.get_document_chunks(args.document_id)
    end = args.end_chunk if args.end_chunk is not None else len(doc_chunks) - 1
    selected = [c for c in doc_chunks if args.start_chunk <= c.chunk_index <= end]

    add_result = context_manager.add_chunks(
        [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "score": None,
                "source": "read",
            }
            for c in selected
        ],
        step=step,
    )
    return {
        "tool": TOOL_NAME,
        "document_id": args.document_id,
        "title": document.title,
        "results": [
            {"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "text": c.text} for c in selected
        ],
        "added": add_result.added,
        "resurfaced": add_result.resurfaced,
        "duplicates": add_result.duplicates,
    }
