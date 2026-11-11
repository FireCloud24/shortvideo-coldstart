"""Recall@K and NDCG@K with full ranking (no sampled negatives).

Each evaluation example has exactly one relevant item (the target), so:

    Recall@K = 1 if rank <= K else 0
    NDCG@K   = 1 / log2(rank + 1) if rank <= K else 0      (ideal DCG is 1)

``rank`` is the target's 1-based position among the candidate items. Ties are
resolved against the model: any candidate scoring the same as the target is
counted as ranked above it. A model that gives every item the same score
therefore gets rank = number of candidates, not a lucky rank of 1.
"""
from __future__ import annotations

import torch


def target_ranks(scores: torch.Tensor, targets: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """1-based rank of each target among its candidates.

    scores:     [B, N] float, higher is better.
    targets:    [B] long, column of the target item.
    candidates: [B, N] bool, True where the item competes with the target. The
                target itself is always treated as a candidate.
    """
    if scores.shape != candidates.shape:
        raise ValueError(f"scores {tuple(scores.shape)} and candidates {tuple(candidates.shape)} differ")
    rows = torch.arange(scores.shape[0], device=scores.device)
    target_scores = scores[rows, targets].unsqueeze(1)
    if torch.isnan(scores).any():
        raise ValueError("NaN in scores")
    others = candidates.clone()
    others[rows, targets] = False
    not_below = (scores >= target_scores) & others
    return not_below.sum(dim=1) + 1


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return (ranks <= k).double().mean().item()


def ndcg_at_k(ranks: torch.Tensor, k: int) -> float:
    gains = (ranks <= k).double() / torch.log2(ranks.double() + 1.0)
    return gains.mean().item()


def summarise(ranks: torch.Tensor, ks=(10,)) -> dict:
    out = {"n": int(ranks.numel())}
    for k in ks:
        # No examples -> no number (None), rather than a NaN that looks like a result.
        out[f"recall@{k}"] = recall_at_k(ranks, k) if ranks.numel() else None
        out[f"ndcg@{k}"] = ndcg_at_k(ranks, k) if ranks.numel() else None
    return out
