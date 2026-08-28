from itertools import pairwise

import pytest

from app.chunking import chunk_text


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_yields_single_chunk() -> None:
    assert chunk_text("hello world", max_words=10, overlap_words=2) == ["hello world"]


def test_empty_and_whitespace_only_yield_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunks_never_exceed_max_words() -> None:
    chunks = chunk_text(words(505), max_words=100, overlap_words=20)

    assert all(len(c.split()) <= 100 for c in chunks)


def test_consecutive_chunks_share_the_overlap() -> None:
    chunks = chunk_text(words(250), max_words=100, overlap_words=20)

    for previous, current in pairwise(chunks):
        assert previous.split()[-20:] == current.split()[:20]


def test_every_word_is_covered_in_order() -> None:
    text = words(333)
    chunks = chunk_text(text, max_words=100, overlap_words=25)

    reconstructed: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_words = chunk.split()
        reconstructed.extend(chunk_words if i == 0 else chunk_words[25:])

    assert reconstructed == text.split()


def test_chunking_is_deterministic() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 40

    assert chunk_text(text) == chunk_text(text)


def test_whitespace_is_normalized_deterministically() -> None:
    assert chunk_text("a  b\n\nc\td") == chunk_text("a b c d")


def test_no_trailing_stub_when_text_ends_on_a_window_boundary() -> None:
    chunks = chunk_text(words(100), max_words=100, overlap_words=20)

    assert chunks == [words(100)]


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", max_words=0)
    with pytest.raises(ValueError):
        chunk_text("x", max_words=10, overlap_words=-1)
    with pytest.raises(ValueError):
        chunk_text("x", max_words=10, overlap_words=10)
