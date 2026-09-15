"""Corpus data model: documents, chunks, and chunking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str
    title: str
    text: str
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_document(document: Document, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Split a document's text into overlapping word-window chunks."""
    words = document.text.split()
    if not words:
        return [
            Chunk(
                chunk_id=f"{document.document_id}::0",
                document_id=document.document_id,
                chunk_index=0,
                text="",
                title=document.title,
            )
        ]

    step = max(chunk_size - overlap, 1)
    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}::{idx}",
                document_id=document.document_id,
                chunk_index=idx,
                text=" ".join(window),
                title=document.title,
            )
        )
        idx += 1
        start += step
    return chunks


class Corpus:
    """In-memory corpus: documents + their chunks, with lookup helpers."""

    def __init__(self, documents: list[Document], chunk_size: int = 400, overlap: int = 50):
        self.documents: dict[str, Document] = {d.document_id: d for d in documents}
        self.chunks: dict[str, Chunk] = {}
        self.chunks_by_document: dict[str, list[Chunk]] = {}
        for document in documents:
            doc_chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
            self.chunks_by_document[document.document_id] = doc_chunks
            for chunk in doc_chunks:
                self.chunks[chunk.chunk_id] = chunk

    def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        return self.chunks_by_document.get(document_id, [])

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks.values())

    @classmethod
    def from_jsonl(cls, path: str | Path, chunk_size: int = 400, overlap: int = 50) -> "Corpus":
        documents = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                documents.append(Document.model_validate_json(line))
        return cls(documents, chunk_size=chunk_size, overlap=overlap)

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for document in self.documents.values():
                f.write(document.model_dump_json() + "\n")


def load_documents_jsonl(path: str | Path) -> list[Document]:
    documents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(Document.model_validate_json(line))
    return documents
