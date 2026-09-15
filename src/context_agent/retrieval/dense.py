"""Dense embedding backends.

`SentenceTransformerEmbedder` is the real, model-backed embedder — default model is
configurable and defaults to a small CPU-friendly model for dev; set it to
`BAAI/bge-m3` in config for the "production-like" setup the spec asks for. It is not
exercised by the default test suite (would download weights).

`HashingEmbedder` is a dependency-free, deterministic bag-of-words hashing embedder
used as the default in tests and offline dev so `use_dense: true` can be exercised
without any model download or network access.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from context_agent.retrieval.base import simple_tokenize

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_M3_MODEL = "BAAI/bge-m3"


class DenseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic hashing-trick bag-of-words embedder. No model download."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in simple_tokenize(text):
                bucket = hash(token) % self.dim
                out[i, bucket] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def build_embedder(name: str) -> DenseEmbedder:
    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name=name)
