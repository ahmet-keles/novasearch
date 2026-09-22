"""Embedding abstraction.

Search code depends only on :class:`EmbeddingProvider`; the concrete
provider is chosen in one place (:func:`get_embedding_provider`, driven by
``NOVA_EMBEDDING_PROVIDER``). Two implementations ship:

- :class:`HashingEmbeddingProvider` (default) — a deterministic lexical
  baseline that needs no model download and no network, which keeps tests
  hermetic and CI fast. It is NOT a semantic model: texts score as similar
  when they share tokens.
- :class:`SentenceTransformerEmbeddingProvider` — a real sentence-
  transformer model (all-MiniLM-L6-v2 by default) served via ONNX through
  the optional ``[model]`` extra.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.text import tokenize


@dataclass(frozen=True)
class EmbeddingSpace:
    """Identity of the space a provider's vectors live in.

    Two vectors are comparable only when they come from the same space:
    same provider type, same model (where one exists), same dimension.
    Equal dimensions are NOT enough — a 384-dim hashing vector and a
    384-dim MiniLM vector are mathematically compatible but semantically
    unrelated, which is exactly the mix-up the persisted space identity
    (see app.embedding_space) exists to prevent.
    """

    provider: str
    model_name: str | None
    dimension: int

    def describe(self) -> str:
        if self.model_name is None:
            return f"{self.provider} ({self.dimension}-dim)"
        return f"{self.provider} {self.model_name} ({self.dimension}-dim)"


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def space(self) -> EmbeddingSpace:
        """The embedding space this provider's vectors belong to."""
        ...

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

    @property
    def space(self) -> EmbeddingSpace:
        return EmbeddingSpace(provider="hashing", model_name=None, dimension=self._dimension)

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


class SentenceTransformerEmbeddingProvider:
    """A real sentence-transformer model, served via ONNX (fastembed).

    The default model is sentence-transformers/all-MiniLM-L6-v2 (384-dim).
    fastembed runs the exported ONNX weights on CPU, so the ``[model]``
    extra stays small — no torch — while producing genuine semantic
    embeddings: paraphrases score as similar without sharing vocabulary.

    Construction downloads the model on first use (network required once);
    ``fastembed`` is imported lazily so the base install never needs it.
    Inference runs in batches of ``batch_size`` and the output is
    L2-normalized defensively (MiniLM's ONNX export already normalizes).
    """

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "the model embedding provider requires the [model] extra: "
                'pip install -e ".[model]"'
            ) from error

        self._batch_size = batch_size
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

        # The model's true output dimension, measured rather than assumed,
        # so dimension validation checks reality — not registry metadata.
        [probe] = self.embed(["dimension probe"])
        self._dimension = len(probe)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def space(self) -> EmbeddingSpace:
        return EmbeddingSpace(
            provider="model", model_name=self._model_name, dimension=self._dimension
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._model.embed(list(texts), batch_size=self._batch_size)
        return [self._normalized(vector.tolist()) for vector in vectors]

    @staticmethod
    def _normalized(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()

    if settings.embedding_provider == "model":
        return SentenceTransformerEmbeddingProvider(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            cache_dir=settings.embedding_cache_dir,
        )

    return HashingEmbeddingProvider(dimension=settings.embedding_dim)
