"""Embedding abstraction.

Search code depends only on :class:`EmbeddingProvider`; the concrete
provider is chosen in one place (:func:`get_embedding_provider`). Milestone
1 ships a single implementation, :class:`HashingEmbeddingProvider` — a
deterministic lexical baseline that needs no model download and no network,
which keeps tests hermetic and CI fast. It is NOT a semantic model: texts
score as similar when they share tokens. Model-backed providers (local
sentence-transformers, hosted APIs) plug in behind the same interface.
"""

import hashlib
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.text import tokenize


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input text, in order."""
        ...


class HashingEmbeddingProvider:
    """Feature-hashing bag-of-words embedding.

    Each token is hashed with BLAKE2b (stable across processes and
    platforms, unlike the built-in ``hash``); the digest picks a bucket and
    a sign, token counts accumulate into the buckets, and the result is
    L2-normalized. Text with no tokens embeds to the zero vector.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension

        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector

        return [value / norm for value in vector]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return HashingEmbeddingProvider(dimension=get_settings().embedding_dim)
