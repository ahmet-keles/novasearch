import pytest

from app.search import reciprocal_rank_fusion


def test_item_in_both_rankings_beats_single_ranking_items() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])

    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_known_values_with_default_k() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b"]], k=60)

    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_higher_rank_scores_higher_within_one_ranking() -> None:
    scores = reciprocal_rank_fusion([["first", "second", "third"]])

    assert scores["first"] > scores["second"] > scores["third"]


def test_weights_bias_the_fusion() -> None:
    unweighted = reciprocal_rank_fusion([["a"], ["b"]])
    weighted = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0])

    assert unweighted["a"] == pytest.approx(unweighted["b"])
    assert weighted["a"] == pytest.approx(2 * weighted["b"])


def test_empty_rankings_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_mismatched_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
