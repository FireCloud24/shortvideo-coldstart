"""Interaction loading and the warm / cold-start split.

The split, in one place (it decides every number in the README):

1. Items are ordered by the time of their first interaction. The last
   ``cold_item_frac`` of items are **cold-start items**. Every interaction with
   a cold item is removed from training, so no model ever sees a cold item's ID
   or any behaviour on it.
2. What is left is each user's **warm sequence**, in time order. Leave-one-out:
   the last warm item is the warm test target, the one before it the validation
   target, the rest is training. Users with fewer than ``min_warm_len`` warm
   interactions only contribute training data.
3. Each removed interaction (user u, cold item c, time t) becomes one **cold
   evaluation example**: the input is u's warm interactions before t, the
   target is c.

Items are re-indexed to 1..n_items (0 is padding). ``item_ids[i]`` gives the
original MicroLens id of index i.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PAD = 0


@dataclass
class EvalSet:
    """Evaluation examples. ``histories[i]`` is the model input (item indices in
    time order), ``targets[i]`` the item to rank, ``excludes[i]`` items removed
    from the candidate list for this example (things the user already has)."""

    histories: list[np.ndarray] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    excludes: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.targets)


@dataclass
class Split:
    n_items: int  # number of real items; valid indices are 1..n_items
    item_ids: np.ndarray  # [n_items + 1] original ids; item_ids[0] == -1
    is_cold: np.ndarray  # [n_items + 1] bool; is_cold[0] is False
    train_seqs: list[np.ndarray]  # one warm training sequence per user (len >= 1)
    valid: EvalSet
    test: EvalSet  # warm test
    cold: EvalSet

    @property
    def n_cold(self) -> int:
        return int(self.is_cold.sum())

    @property
    def n_warm(self) -> int:
        return self.n_items - self.n_cold

    def summary(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_warm_items": self.n_warm,
            "n_cold_items": self.n_cold,
            "n_train_users": len(self.train_seqs),
            "n_train_interactions": int(sum(len(s) for s in self.train_seqs)),
            "n_valid": len(self.valid),
            "n_warm_test": len(self.test),
            "n_cold_test": len(self.cold),
        }


def load_interactions(path: str | Path) -> pd.DataFrame:
    """MicroLens pairs csv: columns user,item,timestamp (ms since epoch)."""
    df = pd.read_csv(path)
    missing = {"user", "item", "timestamp"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df[["user", "item", "timestamp"]]


def choose_cold_items(df: pd.DataFrame, cold_item_frac: float) -> np.ndarray:
    """Original ids of the newest ``cold_item_frac`` of items, by first interaction.

    Ties in first-appearance time are broken by item id so the split is deterministic.
    """
    if not 0.0 <= cold_item_frac < 1.0:
        raise ValueError("cold_item_frac must be in [0, 1)")
    first = df.groupby("item")["timestamp"].min().reset_index()
    first = first.sort_values(["timestamp", "item"], kind="stable")
    n_cold = int(round(cold_item_frac * len(first)))
    return first["item"].to_numpy()[len(first) - n_cold :]


def build_split(df: pd.DataFrame, cold_item_frac: float = 0.10, min_warm_len: int = 3) -> Split:
    if min_warm_len < 3:
        raise ValueError("min_warm_len must be >= 3 (train, valid and test each need an item)")
    if df.duplicated(["user", "item"]).any():
        # A repeat could put a user's test target inside their own input.
        raise ValueError("duplicate (user, item) pairs in interactions")

    # Re-index items to 1..n in original-id order.
    item_ids = np.sort(df["item"].unique())
    index_of = pd.Series(np.arange(1, len(item_ids) + 1), index=item_ids)
    n_items = len(item_ids)

    is_cold = np.zeros(n_items + 1, dtype=bool)
    is_cold[index_of.loc[choose_cold_items(df, cold_item_frac)].to_numpy()] = True

    # Stable sort keeps file order for identical timestamps.
    df = df.assign(idx=index_of.loc[df["item"]].to_numpy())
    df = df.sort_values(["user", "timestamp"], kind="stable")

    train_seqs: list[np.ndarray] = []
    valid, test, cold = EvalSet(), EvalSet(), EvalSet()

    for _, g in df.groupby("user", sort=True):
        items = g["idx"].to_numpy()
        times = g["timestamp"].to_numpy()
        cold_mask = is_cold[items]
        warm = items[~cold_mask]
        warm_times = times[~cold_mask]

        if len(warm) >= min_warm_len:
            train = warm[:-2]
            train_seqs.append(train)
            valid.histories.append(train)
            valid.targets.append(int(warm[-2]))
            valid.excludes.append(train)
            test.histories.append(warm[:-1])
            test.targets.append(int(warm[-1]))
            test.excludes.append(warm[:-1])
        elif len(warm) >= 1:
            train_seqs.append(warm)

        user_cold = items[cold_mask]
        for c, t in zip(user_cold, times[cold_mask]):
            history = warm[warm_times < t]
            if len(history) == 0:
                continue  # nothing to condition on
            cold.histories.append(history)
            cold.targets.append(int(c))
            # Candidates the user already took: their warm history and their other cold items.
            cold.excludes.append(np.concatenate([history, user_cold[user_cold != c]]))

    ids = np.concatenate([[-1], item_ids])
    return Split(n_items, ids, is_cold, train_seqs, valid, test, cold)


def pad_left(seqs: list[np.ndarray], max_len: int) -> np.ndarray:
    """Keep the last ``max_len`` items of each sequence, left-padded with PAD."""
    out = np.full((len(seqs), max_len), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        if len(s):
            out[i, max_len - len(s) :] = s
    return out


def training_arrays(train_seqs: list[np.ndarray], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Next-item training pairs: inputs[i, t] is followed by targets[i, t].

    Sequences longer than max_len + 1 are cut into consecutive windows so that
    every training interaction is used exactly once as a target (apart from each
    user's first item, which has nothing before it).
    """
    inputs, targets = [], []
    for s in train_seqs:
        if len(s) < 2:
            continue
        end = len(s)
        while end >= 2:
            start = max(0, end - (max_len + 1))
            window = s[start:end]
            inputs.append(window[:-1])
            targets.append(window[1:])
            end = start + 1  # the window's first item becomes the next window's last target
            if start == 0:
                break
    return pad_left(inputs, max_len), pad_left(targets, max_len)
