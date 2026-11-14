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


def curve(seeds: dict) -> list[dict]:
    """One point per factor (the fixed ones and the calibrated one), mean and std over seeds, by factor.

    "matched" is left out: it is a different factor in each seed and it is reported in the tables."""
    labels = [r["label"] for r in next(iter(seeds.values()))["rows"] if r["label"] != "matched"]
    points = []
    for label in labels:
        rows = [r for s in seeds.values() for r in s["rows"] if r["label"] == label]
        point = {"label": label, "factor": mean_std([r["factor"] for r in rows])[0]}
        for key, path in (("exposure", ("exposure", "cold_slot_share")), ("new", ("cold_all", "recall@10")),
                          ("old", ("warm_all", "recall@10"))):
            mean, std = mean_std([r[path[0]][path[1]] for r in rows])
            point[key], point[f"{key}_std"] = mean, std or 0.0
        points.append(point)
    return sorted(points, key=lambda p: p["factor"])


# Fixed per model, so a model keeps its colour whichever models are shown.
COLORS = {"id_content_dropout_050": "#2a78d6", "id_content_dropout_100": "#eb6834",
          "id_content_dropout_000": "#1baf7a"}
SHORT = {"id_content_dropout_000": "ID dropout 0", "id_content_dropout_050": "ID dropout 0.5",
         "id_content_dropout_100": "No IDs"}


def visible_range(shown: dict, key: str, x_max: float) -> tuple[float, float]:
    """Lowest and highest value of `key` the lines reach left of x_max (exposure in %)."""
    values = []
    for points in shown.values():
        xs = [100 * p["exposure"] for p in points]
        values += [p[key] for p, x in zip(points, xs) if x <= x_max]
        values.append(float(np.interp(x_max, xs, [p[key] for p in points])))
    return min(values), max(values)


def spread(ys: dict, gap: float) -> dict:
    """Move label positions apart so that neighbours are at least `gap` apart.

    Labels closer than `gap` form a group, spaced `gap` apart around the group's mean; others stay put."""
    order = sorted(ys, key=ys.get)
    groups = [[order[0]]]
    for name in order[1:]:
        if ys[name] - ys[groups[-1][-1]] < gap:
            groups[-1].append(name)
        else:
            groups.append([name])
    out = {}
    for group in groups:
        centre = sum(ys[n] for n in group) / len(group)
        for i, name in enumerate(group):
            out[name] = centre + (i - (len(group) - 1) / 2) * gap
    return out


def chart(results: dict, names: dict[str, str], path: Path, x_max: float = 25.0) -> None:
    """Old- and new-video recall against new videos' share of slots, as the factor grows.

    Each line is one model; moving right along it multiplies new videos' vectors by a bigger factor.
    Hollow dot: the model as trained. Large dot: the factor calibrated on validation users."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from report import INK, MUTED, SURFACE, style_axis

    shown = {n: curve(s) for n, s in results.items() if s}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    panels = (("old", "Old videos", "Recall@10, the old video a user took"),
              ("new", "New videos", "Recall@10, the new video a user took"))
    for ax, (key, title, ylabel) in zip(axes, panels):
        style_axis(ax)
        ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
        ax.axvline(10, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=1)
        for name, points in shown.items():
            color = COLORS.get(name, MUTED)
            xs = [100 * p["exposure"] for p in points]
            ys = [p[key] for p in points]
            ax.plot(xs, ys, color=color, linewidth=2, zorder=3, label=names.get(name, name))
            ax.errorbar(xs, ys, xerr=[100 * p["exposure_std"] for p in points],
                        yerr=[p[f"{key}_std"] for p in points], fmt="none", ecolor=color, alpha=0.45,
                        elinewidth=1, capsize=0, zorder=2)
            for p, x, y in zip(points, xs, ys):
                if p["label"] == "1":
                    ax.plot(x, y, "o", markersize=8, markerfacecolor=SURFACE, markeredgecolor=color,
                            markeredgewidth=2, zorder=5)
                elif p["label"] == "calibrated":
                    ax.plot(x, y, "o", markersize=11, color=color, markeredgecolor=SURFACE,
                            markeredgewidth=2, zorder=6)
                else:
                    ax.plot(x, y, "o", markersize=5, color=color, zorder=4)
        ax.set_xlim(-1, x_max)
        ax.set_xlabel("New videos' share of top-10 slots (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    for ax, (key, _, _) in zip(axes, panels):
        low, high = visible_range(shown, key, x_max)
        pad = 0.15 * (high - low)
        ax.set_ylim(0 if key == "new" else low - pad, high + pad)
    top = axes[0].get_ylim()[1]
    axes[0].text(10.3, top, "new videos' share\nof the catalogue", color=MUTED, fontsize=8, va="top")
    # Direct labels where the lines leave the left panel, pushed apart where lines end close together.
    ends = {name: float(np.interp(x_max, [100 * p["exposure"] for p in points], [p["old"] for p in points]))
            for name, points in shown.items()}
    low, high = axes[0].get_ylim()
    for name, y in spread(ends, 0.045 * (high - low)).items():
        axes[0].annotate(SHORT.get(name, name), (x_max, y), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=8.5, color=INK, annotation_clip=False)
    blue = shown.get("id_content_dropout_050")
    if blue:
        trained = next(p for p in blue if p["label"] == "1")
        calibrated = next(p for p in blue if p["label"] == "calibrated")
        arrow = dict(arrowstyle="-", color=MUTED, linewidth=0.8)
        axes[1].annotate("as trained", (100 * trained["exposure"], trained["new"]), xytext=(-6, 44),
                         textcoords="offset points", ha="center", fontsize=8, color=MUTED, arrowprops=arrow)
        axes[1].annotate(f"factor {calibrated['factor']:.2f},\nchosen on validation users",
                         (100 * calibrated["exposure"], calibrated["new"]), xytext=(-110, 40),
                         textcoords="offset points", fontsize=8, color=MUTED, arrowprops=arrow)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=len(labels),
               frameon=False, fontsize=9)
    fig.suptitle("Give every model the same share of new videos, then compare", x=0.01, ha="left",
                 color=INK, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 0.96, 0.95))
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
