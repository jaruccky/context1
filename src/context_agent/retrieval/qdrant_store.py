"""Vector store backed by Qdrant, running in embedded/local mode (no server process).

Uses `QdrantClient(location=":memory:")` for ephemeral in-process indexes (tests,
short-lived scripts) or `QdrantClient(path=...)` for a persisted local collection —
either way this is Qdrant's own client and storage engine, just not a networked
deployment.
"""
from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore:
    def __init__(self, dim: int, path: str | None = None, collection_name: str = "chunks"):
        self.client = QdrantClient(location=":memory:") if path is None else QdrantClient(path=path)
        self.collection_name = collection_name
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
        )

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        points = [
            qmodels.PointStruct(
                id=_point_id(chunk_id),
                vector=vector.tolist(),
                payload={**payload, "chunk_id": chunk_id},
            )
            for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads)
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            limit=top_k,
        ).points
        return [(r.payload["chunk_id"], float(r.score)) for r in results]
