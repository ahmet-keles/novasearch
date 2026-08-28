"""Shared tokenization: the single definition of "indexable" text.

The hashing embedder, ingestion, and query validation all use this
tokenizer, so what embeds to a non-zero vector, what is worth storing as
a chunk, and what is accepted as a search query cannot drift apart. Text
with no tokens (punctuation or whitespace only) would embed to a zero
vector, which has no direction — cosine distance against it is undefined
— so such input is rejected rather than silently treated as searchable.
"""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, in order of appearance."""
    return _TOKEN_RE.findall(text.lower())
