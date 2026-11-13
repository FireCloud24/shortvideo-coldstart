"""Exposure: share of top-k slots that go to new videos in a mixed feed."""
import numpy as np
import torch

from src.exposure import bootstrap_share, cold_slots_in_top_k, cold_slots_per_user, histogram, summarise_exposure
from src.models import build_model
from src.train import run
from tests.test_content_models import content_setup


def test_cold_slots_are_counted_in_the_top_k_only():
    #                 item: 0    1    2    3    4    5
    scores = torch.tensor([[0.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                           [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    candidates = torch.ones(2, 6, dtype=torch.bool)
    candidates[:, 0] = False
    is_cold = torch.tensor([False, False, False, False, True, True])
    # Row 0's top 2 are items 1 and 2 (both old); row 1's are items 5 and 4 (both new).
    assert cold_slots_in_top_k(scores, candidates, is_cold, k=2).tolist() == [0, 2]
    assert cold_slots_in_top_k(scores, candidates, is_cold, k=4).tolist() == [1, 2]


def test_items_that_are_not_candidates_take_no_slot():
    scores = torch.tensor([[9.0, 8.0, 7.0, 1.0]])
    is_cold = torch.tensor([True, True, False, False])
    candidates = torch.tensor([[False, False, True, True]])
    # The two best scores are cold but not candidates, so they cannot appear.
    assert cold_slots_in_top_k(scores, candidates, is_cold, k=2).tolist() == [0]
    # Asking for more slots than there are candidates counts only real candidates.
    assert cold_slots_in_top_k(scores, torch.tensor([[True, False, False, False]]), is_cold, k=3).tolist() == [1]


def test_histogram_and_summary():
    hist = histogram(np.array([0, 0, 1, 3]), k=10)
    assert len(hist) == 11 and hist[:4] == [2, 1, 0, 1]
    s = summarise_exposure(hist, n_cold_items=10, n_items=100)
    assert s["n_users"] == 4 and s["k"] == 10
    assert abs(s["cold_slot_share"] - 4 / 40) < 1e-12      # 4 cold slots out of 4 users x 10
    assert abs(s["users_with_any_cold"] - 0.5) < 1e-12
    assert abs(s["cold_catalogue_share"] - 0.1) < 1e-12
    assert summarise_exposure([0] * 11)["cold_slot_share"] is None


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    hist = histogram(np.array([0] * 80 + [1] * 15 + [4] * 5), k=10)
    share = summarise_exposure(hist)["cold_slot_share"]
    low, high = bootstrap_share(hist, n_samples=500, seed=1)
    assert low < share < high
    assert (low, high) == bootstrap_share(hist, n_samples=500, seed=1)
    # Every user identical -> no uncertainty.
    assert bootstrap_share(histogram(np.array([2] * 50), k=10), n_samples=200) == (0.2, 0.2)


def test_exposure_of_a_model_that_cannot_score_new_videos_is_zero(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "popularity")
    model = build_model(cfg, split)
    slots = cold_slots_per_user(model, split.test, split, cfg["data"]["max_len"], "cpu", k=3)
    # Popularity gives new videos a count of zero, so they never reach a top 3 of 12 old videos.
    assert len(slots) == len(split.test) and slots.sum() == 0


def test_run_saves_exposure(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "content_knn")
    result = run(cfg, seed=0, split=split, features=feats, log=lambda *_: None)
    e = result["exposure"]
    assert e["n_users"] == len(split.test) and sum(e["users_by_cold_slots"]) == len(split.test)
    assert 0.0 <= e["cold_slot_share"] <= 1.0
    assert abs(e["cold_catalogue_share"] - 2 / 14) < 1e-12
