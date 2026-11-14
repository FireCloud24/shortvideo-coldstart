"""Are new videos buried because their vectors are smaller, or because IDs know more?

In the ID + content model an old video's vector is content + gated ID, and a new
video's vector is content alone. Scores are dot products, so a shorter vector gets
scores closer to zero whatever it says. If that is why new videos never reach the
top 10, then stretching the new videos' vectors to the typical length of an old
video's, with nothing retrained, should bring them back.

For each saved model (checkpoints/<name>_seed<k>.pt) this script measures the vector
lengths, then multiplies every new video's vector by a factor and re-evaluates:

* exposure: share of top-10 slots going to new videos in a mixed feed;
* cold vs all: Recall@10 for the new video a user actually took, ranked among all videos;
* warm vs all: Recall@10 for the old video a user took, ranked among all videos
  (what stretching new videos costs old ones).

Factor 1 is the model as trained. "matched" is the factor that makes the mean length
of a new video's vector equal the mean length of an old video's. "calibrated" is the
factor at which new videos get the same share of top-10 slots as they have of the
catalogue; it is found on the validation users' feeds, where no test target is used,
and then applied to the test users.

    python scripts/scale_test.py --models id_content_dropout_000 id_content_dropout_050
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FACTORS = [1.0, 1.25, 1.5, 2.0, 3.0]


class ScaledCold(nn.Module):
    """A trained model whose new-video vectors are multiplied by ``factor`` when
    scoring candidates. Histories are read with the vectors as trained."""

    def __init__(self, model: nn.Module, is_cold: torch.Tensor, factor: float):
        super().__init__()
        self.model = model
        scale = torch.ones(is_cold.shape[0], 1)
        scale[is_cold.bool()] = factor
        self.register_buffer("scale", scale)

    @torch.no_grad()
    def score_last(self, seqs: torch.Tensor) -> torch.Tensor:
        hidden, table = self.model(seqs)
        return hidden[:, -1] @ (table * self.scale).T


@torch.no_grad()
def vector_lengths(model: nn.Module, is_cold: torch.Tensor) -> dict:
    """Mean length of the item vectors of old and new videos, and the matching factor."""
    model.eval()
    norms = model.items().norm(dim=1)
    cold = is_cold.bool()
    warm = ~cold
    warm[0] = False  # padding
    warm_mean, cold_mean = float(norms[warm].mean()), float(norms[cold].mean())
    return {"warm_mean_length": warm_mean, "cold_mean_length": cold_mean,
            "matched_factor": warm_mean / cold_mean if cold_mean > 0 else float("nan")}


def exposure_at(model, split, evalset, factor: float, max_len: int, device, k: int = 10, batch_size: int = 1024) -> float:
    """Share of top-k slots going to new videos for ``evalset``'s users at this factor."""
    from src.exposure import cold_slots_per_user

    scaled = ScaledCold(model, torch.as_tensor(split.is_cold), factor).to(device)
    slots = cold_slots_per_user(scaled, evalset, split, max_len, device, k, batch_size)
    return float(slots.mean() / k) if len(slots) else float("nan")


def calibrate_factor(exposure_of, target: float, low: float = 0.5, high: float = 4.0, steps: int = 14) -> float:
    """The factor at which ``exposure_of(factor)`` reaches ``target``, by bisection.
    Exposure rises with the factor. If the target lies outside [low, high] the nearer end is returned."""
    if exposure_of(low) >= target:
        return low
    if exposure_of(high) <= target:
        return high
    for _ in range(steps):
        mid = (low + high) / 2
        if exposure_of(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def evaluate_factor(model, split, factor: float, max_len: int, device, ks=(10,), batch_size: int = 1024) -> dict:
    from src.exposure import cold_slots_per_user, histogram, summarise_exposure
    from src.metrics import summarise
    from src.train import evaluate

    scaled = ScaledCold(model, torch.as_tensor(split.is_cold), factor).to(device)
    k = max(ks)
    slots = cold_slots_per_user(scaled, split.test, split, max_len, device, k, batch_size)
    return {
        "factor": factor,
        "exposure": summarise_exposure(histogram(slots, k), split.n_cold, split.n_items),
        "cold_all": summarise(evaluate(scaled, split.cold, split, "cold_all", max_len, device, batch_size), ks),
        "warm_all": summarise(evaluate(scaled, split.test, split, "cold_all", max_len, device, batch_size), ks),
    }


def test_one(cfg: dict, split, features, checkpoint: Path, device, factors=FACTORS) -> dict:
    from src.models import build_model

    model = build_model(cfg, split, features).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    lengths = vector_lengths(model, torch.as_tensor(split.is_cold, device=device))
    rows = [dict(evaluate_factor(model, split, f, cfg["data"]["max_len"], device), label=f"{f:g}") for f in factors]
    matched = evaluate_factor(model, split, lengths["matched_factor"], cfg["data"]["max_len"], device)
    rows.append(dict(matched, label="matched"))
    # Chosen on the validation users (their feeds only), reported on the test users.
    max_len, target = cfg["data"]["max_len"], split.n_cold / split.n_items
    factor = calibrate_factor(lambda f: exposure_at(model, split, split.valid, f, max_len, device), target)
    rows.append(dict(evaluate_factor(model, split, factor, max_len, device), label="calibrated"))
    return {"lengths": lengths, "rows": rows, "calibration": {"target_exposure": target, "factor": factor}}


# ------------------------------------------------------------------ report

def mean_std(values) -> tuple[float, float | None]:
    values = [v for v in values if v is not None]
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def pm(values, scale=1.0, digits=4) -> str:
    mean, std = mean_std(values)
    return f"{scale * mean:.{digits}f}" + ("" if std is None else f" ± {scale * std:.{digits}f}")


def markdown(results: dict, names: dict[str, str]) -> str:
    lines = ["# Are new videos buried by the length of their vectors?", "",
             "Each model is evaluated as trained (factor 1) and again with every new video's vector multiplied "
             "by a factor. Nothing is retrained. \"matched\" makes the mean length of a new video's vector equal "
             "the mean length of an old video's. Exposure: share of top-10 slots going to new videos in a mixed "
             "feed. The two Recall@10 columns rank the target among all videos, old and new. Mean ± sample "
             "standard deviation over seeds.", ""]
    lines += side_by_side(results, names)
    for name, seeds in results.items():
        if not seeds:
            continue
        lengths = [s["lengths"] for s in seeds.values()]
        lines += [f"## {names.get(name, name)}", "",
                  f"Seeds: {len(seeds)}. Mean vector length: old videos {pm([x['warm_mean_length'] for x in lengths], digits=3)}, "
                  f"new videos {pm([x['cold_mean_length'] for x in lengths], digits=3)}. "
                  f"Matching factor: {pm([x['matched_factor'] for x in lengths], digits=2)}.", "",
                  "| Factor on new videos | New-video exposure (%) | Recall@10, new video among all | "
                  "Recall@10, old video among all |", "|---|---:|---:|---:|"]
        labels = [r["label"] for r in next(iter(seeds.values()))["rows"]]
        for i, label in enumerate(labels):
            rows = [s["rows"][i] for s in seeds.values()]
            lines.append(f"| {label} | {pm([r['exposure']['cold_slot_share'] for r in rows], 100, 2)} | "
                         f"{pm([r['cold_all']['recall@10'] for r in rows])} | "
                         f"{pm([r['warm_all']['recall@10'] for r in rows])} |")
        lines.append("")
    return "\n".join(lines)


def row_values(seeds: dict, label: str, path: tuple[str, str]) -> list[float]:
    out = []
    for s in seeds.values():
        for r in s["rows"]:
            if r["label"] == label:
                out.append(r[path[0]][path[1]])
    return out


def side_by_side(results: dict, names: dict[str, str]) -> list[str]:
    """Every model as trained and calibrated, in the same three columns, so they can be compared."""
    usable = {n: s for n, s in results.items() if s and all("calibration" in v for v in s.values())}
    if not usable:
        return []
    lines = ["## Like for like", "",
             "Calibrated: new videos' vectors are multiplied by the factor that gives them their share of the "
             "catalogue in the validation users' top-10 lists. The factor is found without any test target and "
             "the numbers below are for the test users.", "",
             "| Model | New videos' vectors | Factor | New-video exposure (%) | Recall@10, new video among all | "
             "Recall@10, old video among all |", "|---|---|---:|---:|---:|---:|"]
    for name, seeds in usable.items():
        for label, shown in (("1", "as trained"), ("calibrated", "calibrated")):
            factor = "1" if label == "1" else pm([v["calibration"]["factor"] for v in seeds.values()], digits=2)
            lines.append(f"| {names.get(name, name)} | {shown} | {factor} | "
                         f"{pm(row_values(seeds, label, ('exposure', 'cold_slot_share')), 100, 2)} | "
                         f"{pm(row_values(seeds, label, ('cold_all', 'recall@10')))} | "
                         f"{pm(row_values(seeds, label, ('warm_all', 'recall@10')))} |")
    return lines + [""]


def chart(results: dict, names: dict[str, str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from report import COLD_COLOR, INK, MUTED, SURFACE, WARM_COLOR, style_axis

    shown = {n: s for n, s in results.items() if s}
    fig, axes = plt.subplots(1, len(shown), figsize=(5.2 * len(shown), 4.2), facecolor=SURFACE, squeeze=False)
    for ax, (name, seeds) in zip(axes[0], shown.items()):
        n_fixed = len(FACTORS)
        factors = [r["factor"] for r in next(iter(seeds.values()))["rows"][:n_fixed]]
        exposure = [100 * mean_std([s["rows"][i]["exposure"]["cold_slot_share"] for s in seeds.values()])[0]
                    for i in range(n_fixed)]
        warm = [mean_std([s["rows"][i]["warm_all"]["recall@10"] for s in seeds.values()])[0] for i in range(n_fixed)]
        ax.plot(factors, exposure, "o-", color=COLD_COLOR, linewidth=2)
        ax.set_xlabel("Factor on new videos' vectors")
        ax.set_ylabel("New-video exposure (%)", color=COLD_COLOR)
        matched = mean_std([s["lengths"]["matched_factor"] for s in seeds.values()])[0]
        ax.axvline(matched, color=MUTED, linestyle="--", linewidth=1)
        ax.text(matched, ax.get_ylim()[1], " lengths matched", color=MUTED, fontsize=8, va="top")
        style_axis(ax)
        twin = ax.twinx()
        twin.plot(factors, warm, "s--", color=WARM_COLOR, linewidth=1.5)
        twin.set_ylabel("Recall@10 on old videos, among all", color=WARM_COLOR)
        twin.set_ylim(0, max(warm) * 1.25)
        for side in ("top", "left", "right"):
            twin.spines[side].set_visible(False)
        twin.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.set_title(names.get(name, name), loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


NAMES = {"id_content_dropout_000": "IDs + cover + title, ID dropout 0",
         "id_content_dropout_050": "IDs + cover + title, ID dropout 0.5",
         "id_content_dropout_100": "Cover + title, no IDs (ID dropout 1)"}


def main(argv=None) -> int:
    from src.config import load_config
    from src.data import build_split, load_interactions
    from src.features import load_feature_table

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(NAMES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.results) / "scale_test"
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "scale_test.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
    split = None
    for name in args.models:
        cfg = load_config(f"configs/{name}.yaml")
        results.setdefault(name, {})
        for seed in args.seeds:
            checkpoint = Path(args.checkpoints) / f"{name}_seed{seed}.pt"
            done = "calibration" in results[name].get(str(seed), {})
            if done or not checkpoint.exists():
                print(f"{name} seed {seed}: {'already done' if done else 'no checkpoint'}")
                continue
            if split is None:
                split = build_split(load_interactions(cfg["data"]["pairs"]), cfg["data"]["cold_item_frac"],
                                    cfg["data"]["min_warm_len"], cfg["data"].get("recent_frac", 0.10))
            features = load_feature_table(cfg["data"]["features_dir"], cfg["model"]["features"], split.item_ids)
            results[name][str(seed)] = test_one(cfg, split, features, checkpoint, device)
            saved.write_text(json.dumps(results, indent=1), encoding="utf-8")
            first, last = results[name][str(seed)]["rows"][0], results[name][str(seed)]["rows"][-1]
            print(f"{name} seed {seed}: exposure {first['exposure']['cold_slot_share']:.4f} as trained, "
                  f"{last['exposure']['cold_slot_share']:.4f} calibrated on validation "
                  f"(factor {last['factor']:.2f})")
    (out / "scale_test.md").write_text(markdown(results, NAMES), encoding="utf-8")
    print(f"wrote {out / 'scale_test.md'}")
    if any(results.values()):
        chart(results, NAMES, out / "scale_test.png")
        print(f"wrote {out / 'scale_test.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
