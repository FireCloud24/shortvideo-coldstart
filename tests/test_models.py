"""Model tests on tiny synthetic data (CPU, a few seconds)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import apply_overrides, deep_merge, load_config
from src.data import PAD, build_split
from src.metrics import summarise
from src.models import IdItems, Popularity, SASRec, build_model
from src.train import candidate_mask, evaluate, run

N_ITEMS = 12


def cyclic_df(n_users=24, length=6):
    """User u watches items u, u+1, ..., u+5 (mod 12): the next item is always
    'previous + 1', so a sequence model can learn it perfectly."""
    rows = []
    for u in range(n_users):
        for step in range(length):
            rows.append((u + 1, (u + step) % N_ITEMS + 1, 1000 * u + step))
    return pd.DataFrame(rows, columns=["user", "item", "timestamp"])


def tiny_cfg(tmp_path, **model):
    cfg = load_config(overrides=["name=tiny"])
    cfg = deep_merge(cfg, {
        "data": {"max_len": 8},
        "model": {"dim": 32, "n_layers": 1, "n_heads": 2, "dropout": 0.0, **model},
        "train": {"epochs": 60, "batch_size": 8, "lr": 0.01, "patience": 60},
        "out_dir": str(tmp_path / "runs"),
    })
    return cfg


def test_config_merge_and_overrides():
    cfg = load_config(overrides=["name=x", "train.epochs=3", "model.features=[a, b]"])
    assert cfg["train"]["epochs"] == 3 and cfg["model"]["features"] == ["a", "b"]
    assert cfg["train"]["lr"] == 0.001  # untouched default
    assert apply_overrides({"a": {"b": 1}}, ["a.c=2"]) == {"a": {"b": 1, "c": 2}}
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_popularity_counts_training_interactions_only():
    split = build_split(cyclic_df(), cold_item_frac=0.0)
    pop = Popularity(split.n_items).fit(split.train_seqs)
    assert pop.counts[PAD] == 0
    assert pop.counts.sum() == sum(len(s) for s in split.train_seqs)
    scores = pop.score_last(torch.zeros(3, 4, dtype=torch.long))
    assert scores.shape == (3, N_ITEMS + 1) and torch.equal(scores[0], scores[2])


def test_sasrec_is_causal_and_ignores_padding():
    torch.manual_seed(0)
    model = SASRec(IdItems(N_ITEMS, 16), 16, max_len=6, n_layers=2, n_heads=2, dropout=0.0).eval()
    a = torch.tensor([[PAD, PAD, 3, 4, 5, 6]])
    b = torch.tensor([[PAD, PAD, 3, 4, 5, 9]])  # differs only in the last item
    ha, _ = model(a)
    hb, _ = model(b)
    assert torch.isfinite(ha).all()
    assert torch.allclose(ha[:, :5], hb[:, :5], atol=1e-6), "a later item changed an earlier state"
    assert not torch.allclose(ha[:, 5], hb[:, 5])
    # A shorter window of the same history gives the same next-item scores.
    short = torch.tensor([[3, 4, 5, 6]])
    assert torch.allclose(model.score_last(a), model.score_last(short), atol=1e-5)


def test_sasrec_overfits_a_tiny_dataset(tmp_path):
    cfg = tiny_cfg(tmp_path)
    split = build_split(cyclic_df(), cold_item_frac=0.0)
    result = run(cfg, seed=0, split=split, log=lambda *_: None)
    first, last = result["training"]["history"][0], result["training"]["history"][-1]
    assert last["train_loss"] < 0.2 < first["train_loss"]
    assert result["metrics"]["warm"]["recall@10"] > 0.95
    assert result["metrics"]["warm"]["ndcg@10"] > 0.9

    saved = json.loads((tmp_path / "runs" / "tiny" / "seed0" / "metrics.json").read_text())
    assert saved["metrics"] == result["metrics"] and saved["seed"] == 0


def test_evaluation_settings_and_cold_gap(tmp_path):
    """With cold items held out, an untrained ID row cannot be ranked well: the
    ID model's cold metrics must sit far below its warm metrics."""
    df = cyclic_df()
    # Two new items, each watched late by half of the users.
    late = [(u + 1, 13 + (u % 2), 10**9 + u) for u in range(24)]
    df = pd.concat([df, pd.DataFrame(late, columns=df.columns)], ignore_index=True)
    split = build_split(df, cold_item_frac=2 / 14)
    assert split.n_cold == 2 and len(split.cold) == 24

    device = torch.device("cpu")
    assert candidate_mask(split, "warm", device).sum() == 12
    assert candidate_mask(split, "cold", device).sum() == 2
    assert candidate_mask(split, "cold_all", device).sum() == 14

    model = build_model(tiny_cfg(tmp_path), split)
    ranks = evaluate(model, split.cold, split, "cold", 8, device)
    assert ranks.min() >= 1 and ranks.max() <= 2  # only two cold candidates
    ranks_all = evaluate(model, split.cold, split, "cold_all", 8, device)
    assert (ranks_all >= ranks).all()

    # Seen items are excluded: a user's warm test target is ranked among at most 12 - history items.
    warm_ranks = evaluate(model, split.test, split, "warm", 8, device)
    assert warm_ranks.max() <= 12 - min(len(h) for h in split.test.histories)

    pop = Popularity(split.n_items).fit(split.train_seqs)
    pop_cold = summarise(evaluate(pop, split.cold, split, "cold", 8, device), (1,))
    assert pop_cold["recall@1"] == 0.0  # cold items all have zero count: ties count against the model
