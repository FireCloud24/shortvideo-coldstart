"""Metrics against hand-computed examples."""
import math

import pytest
import torch

from src.metrics import ndcg_at_k, recall_at_k, summarise, target_ranks

# Three users, five items. Hand-computed ranks of each target:
#   user 0: target item 2 (score 0.9) is the top score          -> rank 1
#   user 1: target item 0 (score 0.3); items 1, 3 score higher  -> rank 3
#   user 2: target item 4 (score 0.1); all four others higher   -> rank 5
SCORES = torch.tensor(
    [
        [0.1, 0.2, 0.9, 0.3, 0.4],
        [0.3, 0.8, 0.1, 0.5, 0.2],
        [0.5, 0.4, 0.3, 0.2, 0.1],
    ]
)
TARGETS = torch.tensor([2, 0, 4])
ALL = torch.ones_like(SCORES, dtype=torch.bool)


def test_ranks_by_hand():
    assert target_ranks(SCORES, TARGETS, ALL).tolist() == [1, 3, 5]


def test_recall_by_hand():
    ranks = target_ranks(SCORES, TARGETS, ALL)
    assert recall_at_k(ranks, 1) == pytest.approx(1 / 3)
    assert recall_at_k(ranks, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranks, 5) == pytest.approx(1.0)


def test_ndcg_by_hand():
    ranks = target_ranks(SCORES, TARGETS, ALL)
    # ranks 1, 3, 5 -> gains 1/log2(2), 1/log2(4), 1/log2(6)
    assert ndcg_at_k(ranks, 1) == pytest.approx((1.0 + 0 + 0) / 3)
    assert ndcg_at_k(ranks, 3) == pytest.approx((1.0 + 0.5 + 0) / 3)
    assert ndcg_at_k(ranks, 5) == pytest.approx((1.0 + 0.5 + 1 / math.log2(6)) / 3)


def test_excluded_candidates_do_not_count():
    # User 1: remove item 1 (0.8) from the candidates -> only item 3 is above the target.
    cand = ALL.clone()
    cand[1, 1] = False
    assert target_ranks(SCORES, TARGETS, cand).tolist() == [1, 2, 5]


def test_target_counts_even_if_masked_out_of_candidates():
    cand = ALL.clone()
    cand[0, 2] = False  # the target's own column
    assert target_ranks(SCORES, TARGETS, cand).tolist() == [1, 3, 5]


def test_ties_count_against_the_model():
    scores = torch.zeros(1, 4)  # constant scorer
    ranks = target_ranks(scores, torch.tensor([1]), torch.ones(1, 4, dtype=torch.bool))
    assert ranks.tolist() == [4]
    assert recall_at_k(ranks, 3) == 0.0


def test_minus_inf_scores_are_ranked_below():
    scores = torch.tensor([[float("-inf"), 0.2, 0.1]])
    assert target_ranks(scores, torch.tensor([2]), torch.ones(1, 3, dtype=torch.bool)).tolist() == [2]


def test_nan_scores_raise():
    with pytest.raises(ValueError):
        target_ranks(torch.tensor([[float("nan"), 1.0]]), torch.tensor([1]), torch.ones(1, 2, dtype=torch.bool))


def test_summarise_keys():
    out = summarise(torch.tensor([1, 3, 5]), ks=(3, 10))
    assert out["n"] == 3
    assert out["recall@3"] == pytest.approx(2 / 3)
    assert out["recall@10"] == pytest.approx(1.0)
    assert set(out) == {"n", "recall@3", "ndcg@3", "recall@10", "ndcg@10"}
