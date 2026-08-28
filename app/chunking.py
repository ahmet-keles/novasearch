"""Deterministic word-window chunking.

Text is normalized to single-space-separated words and split into windows
of at most ``max_words`` words, each window overlapping the previous one by
``overlap_words``. The same input always produces the same chunks — chunk
identity is stable across re-ingestion, machines, and Python versions.

Whitespace normalization is a deliberate trade-off: chunks are retrieval
units, and the original formatting stays available verbatim on the parent
document row.
"""


def chunk_text(text: str, *, max_words: int = 200, overlap_words: int = 40) -> list[str]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    words = text.split()
    if not words:
        return []

    stride = max_words - overlap_words
    chunks: list[str] = []

    for start in range(0, len(words), stride):
        window = words[start : start + max_words]
        chunks.append(" ".join(window))

        if start + max_words >= len(words):
            break

    return chunks
