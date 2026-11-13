"""Exposure: how many of the top-k slots in a mixed feed go to new (cold) videos.

Recall asks "did the right new video reach the top 10 for the user who wanted it?".
Exposure asks a different question: "when every user gets a top-10 list drawn from
all videos, old and new together, how many of those slots do new videos get at all?".
A model can only recommend a new video well if it is willing to show new videos.

One feed is built per test user, from the history that user has at test time, with
the videos they already interacted with removed. What is saved per run is a histogram:
``users_by_cold_slots[c]`` is the number of users whose top-k held exactly ``c`` new
videos. Every exposure number, and its bootstrap interval over users, comes from that.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import EvalSet, Split, pad_left


def cold_slots_in_top_k(scores: torch.Tensor, candidates: torch.Tensor, is_cold: torch.Tensor,
                        k: int = 10) -> torch.Tensor:
    """For each row, how many of its k best-scoring candidates are cold. [B] long.

    scores [B, N] float, candidates [B, N] bool, is_cold [N] bool. Rows with fewer
    than k candidates count only the candidates they have.
    """
    if scores.shape != candidates.shape:
        raise ValueError(f"scores {tuple(scores.shape)} and candidates {tuple(candidates.shape)} differ")
    k = min(k, scores.shape[1])
    masked = scores.float().masked_fill(~candidates, float("-inf"))
    top = masked.topk(k, dim=1)
    real = torch.isfinite(top.values)  # a slot filled by a non-candidate is not a slot
    return (is_cold[top.indices] & real).sum(dim=1)


@torch.no_grad()
def cold_slots_per_user(model, evalset: EvalSet, split: Split, max_len: int, device, k: int = 10,
                        batch_size: int = 1024) -> np.ndarray:
    """Cold videos in each user's top-k, ranking all items (warm and cold) together."""
    from .train import candidate_mask  # here, not at the top: train.py imports this module

    model.eval()
    base = candidate_mask(split, "cold_all", device)
    is_cold = torch.as_tensor(split.is_cold, device=device)
    out = []
    for start in range(0, len(evalset), batch_size):
        stop = min(start + batch_size, len(evalset))
        seqs = torch.as_tensor(pad_left(evalset.histories[start:stop], max_len), device=device)
        cand = base.unsqueeze(0).repeat(stop - start, 1)
        for row, seen in enumerate(evalset.excludes[start:stop]):
            cand[row, torch.as_tensor(np.asarray(seen), device=device)] = False
        out.append(cold_slots_in_top_k(model.score_last(seqs), cand, is_cold, k).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def histogram(cold_slots: np.ndarray, k: int = 10) -> list[int]:
    """[k + 1] counts: users with 0, 1, ..., k cold videos in their top-k."""
    return np.bincount(np.asarray(cold_slots, dtype=np.int64), minlength=k + 1).tolist()


def summarise_exposure(hist: list[int], n_cold_items: int | None = None, n_items: int | None = None) -> dict:
    """Exposure numbers from a histogram of users by cold slots."""
    hist = np.asarray(hist, dtype=np.float64)
    k, n = len(hist) - 1, hist.sum()
    out = {"k": int(k), "n_users": int(n), "users_by_cold_slots": [int(h) for h in hist]}
    if n == 0:
        out.update({"cold_slot_share": None, "users_with_any_cold": None})
        return out
    out["cold_slot_share"] = float((hist * np.arange(k + 1)).sum() / (n * k))  # share of all slots
    out["users_with_any_cold"] = float(hist[1:].sum() / n)                     # share of users
    if n_cold_items is not None and n_items:
        out["cold_catalogue_share"] = n_cold_items / n_items  # what an even spread over items would give
    return out


def bootstrap_share(hist: list[int], n_samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the cold slot share, resampling users. Drawing users with
    replacement is a multinomial draw over the histogram bins."""
    hist = np.asarray(hist, dtype=np.float64)
    k, n = len(hist) - 1, int(hist.sum())
    if n == 0:
        return float("nan"), float("nan")
    draws = np.random.RandomState(seed).multinomial(n, hist / n, size=n_samples)
    shares = (draws * np.arange(k + 1)).sum(axis=1) / (n * k)
    low, high = np.percentile(shares, [2.5, 97.5])
    return float(low), float(high)
