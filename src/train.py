"""Training and evaluation, shared by every model.

Evaluation settings (all full ranking, see src/metrics.py):

* ``warm``          warm test target ranked among all warm items.
* ``warm_matched``  warm test target ranked among a fixed random sample of warm
                    items, as many as there are cold items. Same task size as
                    ``cold``, so the two numbers can be compared directly.
* ``cold``          cold target ranked among all cold items: "which new video
                    do we show this user?".
* ``cold_all``      cold target ranked among all items, warm and cold together:
                    can a new video compete with established ones in one feed?
                    This is the realistic setting and the hardest one.

In every setting, items the user already interacted with are removed from the
candidates.
"""
from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .exposure import cold_slots_per_user, histogram, summarise_exposure
from .data import PAD, EvalSet, Split, build_split, load_interactions, pad_left, training_arrays
from .features import load_feature_table, read_categories
from .metrics import summarise, target_ranks
from .models import TRAINING_FREE, build_model

MATCHED_SAMPLE_SEED = 0  # the warm_matched candidate sample is the same for every model and seed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def candidate_mask(split: Split, setting: str, device) -> torch.Tensor:
    """[n_items + 1] bool: which items compete in this evaluation setting."""
    is_cold = torch.as_tensor(split.is_cold, device=device)
    real = torch.ones_like(is_cold)
    real[PAD] = False
    if setting == "warm":
        return real & ~is_cold
    if setting == "warm_matched":
        warm = np.flatnonzero(~split.is_cold[1:]) + 1
        size = min(split.n_cold, len(warm))
        chosen = np.random.RandomState(MATCHED_SAMPLE_SEED).choice(warm, size=size, replace=False)
        mask = torch.zeros_like(is_cold)
        mask[torch.as_tensor(chosen, device=device)] = True
        return mask
    if setting == "cold":
        return is_cold
    if setting == "cold_all":
        return real
    raise ValueError(f"unknown evaluation setting {setting!r}")


@torch.no_grad()
def evaluate(model, evalset: EvalSet, split: Split, setting: str, max_len: int, device,
             batch_size: int = 1024) -> torch.Tensor:
    """Rank of every target in ``evalset`` (CPU long tensor)."""
    model.eval()
    base = candidate_mask(split, setting, device)
    ranks = []
    for start in range(0, len(evalset), batch_size):
        stop = min(start + batch_size, len(evalset))
        seqs = torch.as_tensor(pad_left(evalset.histories[start:stop], max_len), device=device)
        targets = torch.as_tensor(evalset.targets[start:stop], device=device)
        scores = model.score_last(seqs)

        cand = base.unsqueeze(0).repeat(stop - start, 1)
        rows = np.concatenate([np.full(len(e), i) for i, e in enumerate(evalset.excludes[start:stop])])
        cols = np.concatenate(evalset.excludes[start:stop])
        cand[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)] = False
        ranks.append(target_ranks(scores, targets, cand).cpu())
    return torch.cat(ranks) if ranks else torch.zeros(0, dtype=torch.long)


def popularity_log_prior(train_seqs, n_items: int) -> torch.Tensor:
    """[n_items + 1] log of each item's share of training interactions.

    Items with no training interactions (padding, cold items) get 0; they are masked
    out of the training loss anyway.
    """
    counts = torch.zeros(n_items + 1, dtype=torch.float64)
    for seq in train_seqs:
        counts.index_add_(0, torch.as_tensor(seq, dtype=torch.long), torch.ones(len(seq), dtype=torch.float64))
    counts[PAD] = 0
    seen = counts > 0
    prior = torch.zeros(n_items + 1, dtype=torch.float64)
    prior[seen] = torch.log(counts[seen] / counts.sum())
    return prior.float()


def train_epoch(model, inputs, targets, train_items, optimizer, batch_size, device, generator,
                log_prior: torch.Tensor | None = None) -> float:
    """One pass over the training windows. Loss: softmax over all warm items.

    With ``log_prior`` (already multiplied by the correction strength), the loss is
    computed on ``score + log_prior``. The prior then explains "this video is popular",
    so the model's own score does not have to. At serving time the prior is left out
    (``score_last`` never sees it), which removes the popularity signal from the ranking.
    This is the logit-adjustment / logQ idea applied to a full softmax.
    """
    model.train()
    order = torch.randperm(len(inputs), generator=generator)
    total, count = 0.0, 0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        seqs = inputs[idx].to(device)
        tgt = targets[idx].to(device)
        hidden, table = model(seqs)
        keep = tgt != PAD
        logits = hidden[keep] @ table.T  # [positions, n_items + 1]
        if log_prior is not None:
            logits = logits + log_prior
        logits = logits.masked_fill(~train_items, float("-inf"))  # padding and cold items cannot be predicted
        loss = F.cross_entropy(logits, tgt[keep])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * int(keep.sum())
        count += int(keep.sum())
    return total / max(count, 1)


def fit(model, split: Split, cfg: dict, device, seed: int, log=print) -> dict:
    """Train with early stopping on validation NDCG@10. Returns the training log;
    the model is left holding its best-validation weights."""
    t, max_len = cfg["train"], cfg["data"]["max_len"]
    inputs, targets = (torch.as_tensor(a) for a in training_arrays(split.train_seqs, max_len))
    train_items = candidate_mask(split, "warm", device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=t["lr"], weight_decay=t["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed)
    strength = float(t.get("pop_correction") or 0.0)
    log_prior = strength * popularity_log_prior(split.train_seqs, split.n_items).to(device) if strength else None

    best, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, t["epochs"] + 1):
        loss = train_epoch(model, inputs, targets, train_items, optimizer, t["batch_size"], device, generator,
                           log_prior)
        ranks = evaluate(model, split.valid, split, "warm", max_len, device, cfg["eval"]["batch_size"])
        ndcg = summarise(ranks, (10,))["ndcg@10"]
        history.append({"epoch": epoch, "train_loss": loss, "valid_ndcg@10": ndcg})
        log(f"  epoch {epoch:3d}  loss {loss:.4f}  valid ndcg@10 {ndcg:.4f}")
        if ndcg > best:
            best, bad_epochs = ndcg, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda h: h["valid_ndcg@10"])["epoch"] if history else 0
    return {"history": history, "best_epoch": best_epoch, "epochs_run": len(history)}


def git_head() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def per_item_table(evalset: EvalSet, ranks: torch.Tensor, split: Split, k: int = 10) -> list[dict]:
    """Cold results aggregated per target item, for the error analysis in the report."""
    targets = np.asarray(evalset.targets)
    r = ranks.numpy()
    rows = []
    for item in np.unique(targets):
        rr = r[targets == item]
        rows.append({
            "item": int(split.item_ids[item]),
            "n": int(len(rr)),
            f"recall@{k}": float((rr <= k).mean()),
            f"ndcg@{k}": float(np.where(rr <= k, 1.0 / np.log2(rr + 1.0), 0.0).mean()),
        })
    return rows


def top_k_ranks(ranks: torch.Tensor, k: int = 10) -> np.ndarray:
    """Ranks as uint8 with everything outside the top ``k`` stored as 0. This is all
    Recall@k and NDCG@k need, and it compresses to almost nothing."""
    r = ranks.numpy()
    return np.where(r <= k, r, 0).astype(np.uint8)


def run(cfg: dict, seed: int, split: Split | None = None, features=None, log=print, categories=None) -> dict:
    """Train and evaluate one model with one seed; write results/runs/<name>/seed<seed>/."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if split is None:
        split = build_split(load_interactions(cfg["data"]["pairs"]), cfg["data"]["cold_item_frac"],
                            cfg["data"]["min_warm_len"], cfg["data"].get("recent_frac", 0.10))
    max_len, ks, bs = cfg["data"]["max_len"], tuple(cfg["eval"]["ks"]), cfg["eval"]["batch_size"]

    if features is None and cfg["model"]["features"]:
        features = load_feature_table(cfg["data"]["features_dir"], cfg["model"]["features"], split.item_ids)
    if categories is None and cfg["model"]["kind"] == "popularity_category":
        by_id = read_categories(cfg["data"]["categories"])
        categories = [None] + [by_id.get(int(i)) for i in split.item_ids[1:]]
    model = build_model(cfg, split, features, categories).to(device)
    start = time.time()
    if isinstance(model, TRAINING_FREE):
        training = {"history": [], "best_epoch": 0, "epochs_run": 0}
    else:
        training = fit(model, split, cfg, device, seed, log)
    train_seconds = time.time() - start

    valid_ranks = evaluate(model, split.valid, split, "warm", max_len, device, bs)
    warm_ranks = evaluate(model, split.test, split, "warm", max_len, device, bs)
    warm_matched_ranks = evaluate(model, split.test, split, "warm_matched", max_len, device, bs)
    cold_ranks = evaluate(model, split.cold, split, "cold", max_len, device, bs)
    cold_all_ranks = evaluate(model, split.cold, split, "cold_all", max_len, device, bs)

    # One mixed feed per warm test user: how many of its top-k slots are new videos.
    k_exposure = max(ks)
    cold_slots = cold_slots_per_user(model, split.test, split, max_len, device, k_exposure, bs)
    exposure = summarise_exposure(histogram(cold_slots, k_exposure), split.n_cold, split.n_items)

    result = {
        "name": cfg["name"],
        "seed": seed,
        "config": cfg,
        "split": split.summary(),
        "metrics": {
            "valid": summarise(valid_ranks, ks),
            "warm": summarise(warm_ranks, ks),
            "warm_matched": summarise(warm_matched_ranks, ks),
            "cold": summarise(cold_ranks, ks),
            "cold_all": summarise(cold_all_ranks, ks),
        },
        "exposure": exposure,
        "training": training,
        "epochs_cap": cfg["train"]["epochs"],
        "train_seconds": round(train_seconds, 1),
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
        "torch": torch.__version__,
        "git_head": git_head(),
    }
    out = Path(cfg["out_dir"]) / cfg["name"] / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "cold_by_item.json").write_text(json.dumps(per_item_table(split.cold, cold_ranks, split)),
                                           encoding="utf-8")
    # Per-example top-10 ranks, in evaluation-set order, for paired comparisons in the report.
    np.savez_compressed(out / "top10_ranks.npz", warm=top_k_ranks(warm_ranks),
                        warm_matched=top_k_ranks(warm_matched_ranks), cold=top_k_ranks(cold_ranks),
                        cold_all=top_k_ranks(cold_all_ranks))
    if cfg["train"].get("save_model") and not isinstance(model, TRAINING_FREE):
        ckpt = Path(cfg["train"].get("checkpoint_dir", "checkpoints"))
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt / f"{cfg['name']}_seed{seed}.pt")
    m = result["metrics"]
    log(f"{cfg['name']} seed {seed}: warm ndcg@10 {m['warm'].get('ndcg@10')}  "
        f"cold ndcg@10 {m['cold'].get('ndcg@10')}  best epoch {training['best_epoch']} "
        f"of {training['epochs_run']} run (cap {cfg['train']['epochs']})  -> {out}")
    return result
