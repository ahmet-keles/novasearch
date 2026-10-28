from types import SimpleNamespace
from uuid import UUID

import pytest

import app.search as search_module
from app.reranking import rerank_hits
from app.routes.search import search_cache_mode
from app.search import SearchHit, SearchMode


def hit(index: int, title: str) -> SearchHit:
    return SearchHit(
        chunk_id=UUID(int=index + 1),
        document_id=UUID(int=index + 101),
        document_title=title,
        chunk_index=0,
        content=f"content for {title}",
        score=float(10 - index),
    )


class FixedReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.query: str | None = None
        self.documents: list[str] | None = None

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.query = query
        self.documents = documents
        return self.scores


def test_rerank_hits_sorts_by_score_and_preserves_hit_fields() -> None:
    candidates = [hit(0, "first"), hit(1, "second"), hit(2, "third")]
    reranker = FixedReranker([0.1, 2.5, -1.0])

    ranked = rerank_hits("query", candidates, reranker)

    assert [item.document_title for item in ranked] == ["second", "first", "third"]
    assert [item.score for item in ranked] == [2.5, 0.1, -1.0]
    assert ranked[0].chunk_id == candidates[1].chunk_id
    assert reranker.query == "query"
    assert reranker.documents == [
        "first\n\ncontent for first",
        "second\n\ncontent for second",
        "third\n\ncontent for third",
    ]


def test_rerank_hits_rejects_mismatched_score_count() -> None:
    with pytest.raises(RuntimeError, match="different number of scores"):
        rerank_hits("query", [hit(0, "only")], FixedReranker([]))


def test_rerank_configuration_has_its_own_hybrid_cache_key() -> None:
    assert search_cache_mode(
        SearchMode.hybrid,
        rerank_enabled=True,
        rerank_model="model-a",
    ) == "hybrid:rerank:model-a"
    assert search_cache_mode(
        SearchMode.hybrid,
        rerank_enabled=False,
        rerank_model="model-a",
    ) == "hybrid"
    assert search_cache_mode(
        SearchMode.keyword,
        rerank_enabled=True,
        rerank_model="model-a",
    ) == "keyword"


def test_hybrid_search_reranks_fused_candidates_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = [hit(0, "first"), hit(1, "second"), hit(2, "third")]
    keyword = [hit(1, "second"), hit(0, "first")]
    monkeypatch.setattr(search_module, "semantic_search", lambda *args: semantic)
    monkeypatch.setattr(search_module, "keyword_search", lambda *args: keyword)
    monkeypatch.setattr(
        search_module,
        "get_settings",
        lambda: SimpleNamespace(search_candidate_pool=3, rerank_enabled=True),
    )
    reranker = FixedReranker([0.0] * 3)
    monkeypatch.setattr(search_module, "get_cross_encoder_reranker", lambda: reranker)

    ranked = search_module.hybrid_search(None, None, "query", 2)

    assert reranker.query == "query"
    assert len(reranker.documents) == 3
    assert len(ranked) == 2


def test_hybrid_search_does_not_load_reranker_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search_module, "semantic_search", lambda *args: [hit(0, "first")])
    monkeypatch.setattr(search_module, "keyword_search", lambda *args: [])
    monkeypatch.setattr(
        search_module,
        "get_settings",
        lambda: SimpleNamespace(search_candidate_pool=3, rerank_enabled=False),
    )

    def unexpected_reranker() -> None:
        pytest.fail("disabled reranking must not load the model")

    monkeypatch.setattr(search_module, "get_cross_encoder_reranker", unexpected_reranker)

    assert search_module.hybrid_search(None, None, "query", 1)[0].document_title == "first"
