"""Split tests on a 10-user toy file."""
import numpy as np
import pandas as pd
import pytest

from src.data import PAD, build_split, choose_cold_items, load_interactions, pad_left, training_arrays

N_USERS = 10
N_ITEMS = 20
COLD_FRAC = 0.10  # -> 2 cold items


@pytest.fixture()
def toy_path(tmp_path):
    """10 users, 20 items. Item k first appears at time 100 * k, so items 119 and
    120 (original ids; 100 + k) are the two newest and must be the cold ones.

    Users 1..9 interact with 5 early items each; the first six also take cold
    item 119 and the first three cold item 120. User 10 is the first viewer of
    every warm item, which pins the first-appearance times.
    """
    rows = []
    for u in range(N_USERS):
        warm = [(u + j) % 18 + 1 for j in range(5)] if u != 9 else []
        for step, k in enumerate(warm):
            rows.append((u + 1, 100 + k, 100 * k + 10_000 * (step + 1) + u))
        if u < 6:
            rows.append((u + 1, 119, 1_000_000 + u))
        if u < 3:
            rows.append((u + 1, 120, 2_000_000 + u))
    # Seed first appearances so item k is first seen at 100 * k.
    for k in range(1, 19):
        rows.append((10, 100 + k, 100 * k))
    df = pd.DataFrame(rows, columns=["user", "item", "timestamp"]).sample(frac=1.0, random_state=0)
    path = tmp_path / "pairs.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def split(toy_path):
    return build_split(load_interactions(toy_path), cold_item_frac=COLD_FRAC)


def test_cold_items_are_the_newest(toy_path):
    cold = choose_cold_items(load_interactions(toy_path), COLD_FRAC)
    assert sorted(cold) == [119, 120]


def test_split_sizes_match_config(split):
    assert split.n_items == N_ITEMS
    assert split.n_cold == round(COLD_FRAC * N_ITEMS) == 2
    assert split.n_warm == 18
    assert split.item_ids[0] == -1 and not split.is_cold[PAD]
    assert sorted(split.item_ids[split.is_cold]) == [119, 120]
    assert len(split.train_seqs) == N_USERS  # every user has at least one warm item
    assert len(split.valid) == len(split.test) == N_USERS  # user 10 has 18 warm items
    assert len(split.cold) == 6 + 3  # one example per (user, cold item) interaction


def test_no_cold_item_in_train_or_any_model_input(split):
    cold_idx = set(np.flatnonzero(split.is_cold))
    for seq in split.train_seqs:
        assert not cold_idx & set(seq)
    for ev in (split.valid, split.test, split.cold):
        for h in ev.histories:
            assert not cold_idx & set(h)
    assert not cold_idx & set(split.valid.targets)
    assert not cold_idx & set(split.test.targets)
    assert set(split.cold.targets) <= cold_idx


def test_leave_one_out_is_in_time_order(toy_path, split):
    df = load_interactions(toy_path)
    index_of = {orig: i for i, orig in enumerate(split.item_ids)}
    cold_orig = {119, 120}
    for row, (_, g) in enumerate(df.groupby("user", sort=True)):
        warm = [index_of[i] for i in g.sort_values("timestamp")["item"] if i not in cold_orig]
        assert list(split.train_seqs[row]) == warm[:-2]
        assert split.valid.targets[row] == warm[-2]
        assert split.test.targets[row] == warm[-1]
        assert list(split.test.histories[row]) == warm[:-1]
        # Targets never leak into their own input.
        assert split.valid.targets[row] not in split.valid.histories[row]
        assert split.test.targets[row] not in split.test.histories[row]


def test_cold_examples_use_only_earlier_warm_history(toy_path, split):
    df = load_interactions(toy_path)
    index_of = {orig: i for i, orig in enumerate(split.item_ids)}
    expected = 0
    seen = list(zip(map(tuple, split.cold.histories), split.cold.targets))
    for _, g in df.groupby("user", sort=True):
        g = g.sort_values("timestamp")
        for _, r in g[g["item"].isin([119, 120])].iterrows():
            before = g[(g["timestamp"] < r["timestamp"]) & ~g["item"].isin([119, 120])]
            hist = tuple(index_of[i] for i in before["item"])
            assert (hist, index_of[r["item"]]) in seen
            expected += 1
    assert expected == len(split.cold)
    for target, exclude in zip(split.cold.targets, split.cold.excludes):
        assert target not in exclude


def test_users_with_too_few_warm_items_only_train():
    df = pd.DataFrame(
        [(1, 1, 1), (1, 2, 2), (1, 3, 3), (2, 1, 4), (2, 2, 5)], columns=["user", "item", "timestamp"]
    )
    s = build_split(df, cold_item_frac=0.0)
    assert len(s.train_seqs) == 2 and len(s.valid) == len(s.test) == 1
    assert list(s.train_seqs[1]) == [1, 2]


def test_pad_left_keeps_the_most_recent_items():
    out = pad_left([np.array([1, 2, 3, 4]), np.array([5])], max_len=3)
    assert out.tolist() == [[2, 3, 4], [PAD, PAD, 5]]


def test_training_arrays_use_every_interaction_once_as_target():
    seq = np.arange(1, 8)  # 7 items, windows of 3
    inputs, targets = training_arrays([seq, np.array([9])], max_len=3)
    assert sorted(targets[targets != PAD].tolist()) == [2, 3, 4, 5, 6, 7]
    # Each target is the item right after its aligned input.
    for i, t in zip(inputs.ravel(), targets.ravel()):
        if t != PAD:
            assert t == i + 1


def test_duplicate_user_item_pairs_are_rejected():
    df = pd.DataFrame([(1, 1, 1), (1, 1, 2), (1, 2, 3)], columns=["user", "item", "timestamp"])
    with pytest.raises(ValueError, match="duplicate"):
        build_split(df, cold_item_frac=0.0)
