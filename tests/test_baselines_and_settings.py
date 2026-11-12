"""The matched warm setting, the training-free baselines and the saved per-example ranks."""
import numpy as np
import pandas as pd
import torch

from src.data import build_split
from src.metrics import summarise
from src.models import CategoryPopularity, ContentKNN, RecentPopularity, build_model
from src.train import candidate_mask, evaluate, run, top_k_ranks
from tests.test_content_models import content_setup
from tests.test_models import cyclic_df, tiny_cfg


def split_with_cold(n_cold=4):
    """Cyclic warm data (12 items) plus ``n_cold`` new items watched late."""
    df = cyclic_df()
    late = [(u + 1, 13 + (u % n_cold), 10**9 + u) for u in range(24)]
    df = pd.concat([df, pd.DataFrame(late, columns=df.columns)], ignore_index=True)
    return build_split(df, cold_item_frac=n_cold / (12 + n_cold))


def test_matched_warm_setting_has_as_many_candidates_as_the_cold_setting():
    split = split_with_cold(4)
    device = torch.device("cpu")
    matched = candidate_mask(split, "warm_matched", device)
    assert matched.sum() == candidate_mask(split, "cold", device).sum() == 4
    assert not (matched & torch.as_tensor(split.is_cold)).any()      # warm items only
    assert not matched[0]                                            # never the padding row
    assert torch.equal(matched, candidate_mask(split, "warm_matched", device))  # same sample every call


def test_matched_warm_ranks_are_never_worse_than_full_warm_ranks(tmp_path):
    split = split_with_cold(4)
    model = build_model(tiny_cfg(tmp_path), split)
    device = torch.device("cpu")
    full = evaluate(model, split.test, split, "warm", 8, device)
    matched = evaluate(model, split.test, split, "warm_matched", 8, device)
    assert (matched <= full).all()          # a subset of the competitors
    assert matched.max() <= 4 + 1           # the sample, plus the target when it is not in it


def test_cold_candidates_keep_cold_items_the_user_takes_later():
    # One user: three warm items, then cold item 21 at t=100 and cold item 22 at t=200.
    rows = [(1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 21, 100), (1, 22, 200)]
    # Other users pin the first-appearance order so that 21 and 22 are the newest items.
    rows += [(2, i, i) for i in range(1, 19)]
    split = build_split(pd.DataFrame(rows, columns=["user", "item", "timestamp"]), cold_item_frac=2 / 20)
    index = {int(orig): i for i, orig in enumerate(split.item_ids)}
    by_target = dict(zip(split.cold.targets, split.cold.excludes))
    assert index[22] not in by_target[index[21]]     # 22 comes later: still a candidate
    assert index[21] in by_target[index[22]]         # 21 was already taken: excluded


def test_recent_popularity_counts_only_the_last_part_of_training_time():
    split = split_with_cold(4)
    assert split.recent_counts.sum() > 0
    assert split.recent_counts.sum() < sum(len(s) for s in split.train_seqs)
    assert split.recent_counts[split.is_cold].sum() == 0
    model = RecentPopularity(split.n_items).fit_counts(split.recent_counts)
    scores = model.score_last(torch.zeros(3, 8, dtype=torch.long))
    assert scores.shape == (3, split.n_items + 1) and scores[0, 0] == 0


def test_category_popularity_scores_a_cold_item_by_its_category():
    split = split_with_cold(4)
    counts = np.zeros(split.n_items + 1)
    for seq in split.train_seqs:
        np.add.at(counts, seq, 1)
    # Warm items 1-6 are "a", 7-12 are "b"; cold items 13, 14 are "a", 15 is "b", 16 is unknown.
    categories = [None] + ["a"] * 6 + ["b"] * 6 + ["a", "a", "b", None]
    model = CategoryPopularity(split.n_items).fit_categories(split.train_seqs, categories, split.is_cold)
    scores = model.score_last(torch.zeros(1, 8, dtype=torch.long))[0]
    assert scores[13] == scores[14] == scores[1]
    assert torch.isclose(scores[13], torch.tensor(counts[1:7].mean(), dtype=scores.dtype))
    assert torch.isclose(scores[15], torch.tensor(counts[7:13].mean(), dtype=scores.dtype))
    assert torch.isclose(scores[16], torch.tensor(counts[1:13].mean(), dtype=scores.dtype))


def test_content_knn_ranks_the_item_most_like_the_history_first(tmp_path):
    _, split, feats = content_setup(tmp_path, "sasrec_content")
    model = ContentKNN(feats)
    # History of item 3 only: cold item 13 has item 3's features, so it wins among cold items.
    scores = model.score_last(torch.tensor([[0, 0, 3]]))[0]
    assert scores[13] > scores[14]
    assert torch.isclose(scores[13], torch.tensor(1.0))
    # Padding does not change the profile.
    assert torch.allclose(scores, model.score_last(torch.tensor([[3]]))[0])
    # It is training-free: there is nothing to optimise.
    assert not any(p.requires_grad for p in model.parameters())


def test_run_writes_matched_metrics_and_per_example_ranks(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "content_knn")
    result = run(cfg, seed=0, split=split, features=feats, log=lambda *_: None)
    assert result["training"]["epochs_run"] == 0
    assert set(result["metrics"]) == {"valid", "warm", "warm_matched", "cold", "cold_all"}
    saved = np.load(tmp_path / "runs" / "tiny" / "seed0" / "top10_ranks.npz")
    assert len(saved["warm"]) == len(saved["warm_matched"]) == len(split.test)
    assert len(saved["cold"]) == len(saved["cold_all"]) == len(split.cold)
    # The saved ranks reproduce the reported metric.
    recall = float((saved["cold"] > 0).mean())
    assert abs(recall - result["metrics"]["cold"]["recall@10"]) < 1e-12


def test_top_k_ranks_zeroes_everything_outside_the_top_k():
    assert top_k_ranks(torch.tensor([1, 10, 11, 5000])).tolist() == [1, 10, 0, 0]
    assert summarise(torch.tensor([1, 11]), (10,))["recall@10"] == 0.5
