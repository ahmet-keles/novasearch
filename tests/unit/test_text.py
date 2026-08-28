import pytest

from app.text import tokenize


def test_tokenize_lowercases_and_extracts_alphanumeric_runs() -> None:
    assert tokenize("Hybrid-Search v2!") == ["hybrid", "search", "v2"]


@pytest.mark.parametrize("text", ["!!!", "---", "...,;:", "—…", "!?!?"])
def test_punctuation_only_text_has_no_tokens(text: str) -> None:
    assert tokenize(text) == []


def test_whitespace_only_text_has_no_tokens() -> None:
    assert tokenize("   \n\t ") == []
    assert tokenize("") == []


def test_mixed_text_keeps_only_its_tokens() -> None:
    assert tokenize("!!! stop --- the 3rd presses !!!") == ["stop", "the", "3rd", "presses"]
