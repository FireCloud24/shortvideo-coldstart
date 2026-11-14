"""The trade-off curve: accuracy on old videos against exposure of new videos.

Each model is one point: warm Recall@10 (how well it recommends videos it has
seen) against the share of top-10 slots it gives to new videos in a mixed feed.
The ID + content model is run at several ID dropout rates, which traces a curve
from "always use the ID" to "never use it" (the content-only model). Runs trained
with the popularity correction are drawn on the same axes, joined to the run with
the same ID dropout and no correction.

Reads results/runs/*/seed*/metrics.json only. Nothing is retrained.

    python scripts/tradeoff.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exposure import TIED_SCORES, label  # noqa: E402
from report import COLD_COLOR, INK, MUTED, SURFACE, WARM_COLOR, load_runs, mean_std, ordered, style_axis  # noqa: E402

SWEEP_FEATURES = ["cnclip_image", "cnclip_text"]  # the curve is drawn for this ID + content model


def point(name: str, runs: list[dict], k: int = 10) -> dict | None:
    """One model's place on the figure, averaged over seeds."""
    usable = [r for r in runs if r.get("exposure") and r["exposure"].get("cold_slot_share") is not None
              and r.get("metrics", {}).get("warm", {}).get(f"recall@{k}") is not None]
    if not usable:
        return None
    cfg = usable[0].get("config", {})
    model, train = cfg.get("model", {}), cfg.get("train", {})
    warm, warm_std = mean_std([r["metrics"]["warm"][f"recall@{k}"] for r in usable])
    share, share_std = mean_std([r["exposure"]["cold_slot_share"] for r in usable])
    return {
        "name": name, "seeds": len(usable), "warm": warm, "warm_std": warm_std,
        "exposure": share, "exposure_std": share_std,
        "kind": model.get("kind"), "features": list(model.get("features") or []),
        "id_dropout": model.get("id_dropout") if model.get("kind") == "sasrec_id_content" else None,
        "pop_correction": float(train.get("pop_correction") or 0.0),
    }


def classify(points: list[dict]) -> tuple[list[dict], list[tuple[dict, dict]], list[dict]]:
    """(curve sorted by ID dropout, [(corrected run, its uncorrected partner)], every other model)."""
    in_sweep = lambda p: p["kind"] == "sasrec_id_content" and p["features"] == SWEEP_FEATURES
    curve = sorted((p for p in points if in_sweep(p) and not p["pop_correction"]), key=lambda p: p["id_dropout"])
    by_dropout = {p["id_dropout"]: p for p in curve}
    corrected = [(p, by_dropout.get(p["id_dropout"])) for p in points if p["pop_correction"]]
    others = [p for p in points if not in_sweep(p) and not p["pop_correction"]]
    return curve, corrected, others


def pm(mean: float, std: float | None, scale: float = 1.0, digits: int = 4) -> str:
    return f"{scale * mean:.{digits}f}" + ("" if std is None else f" ± {scale * std:.{digits}f}")


def change(a: dict, b: dict) -> str:
    """'b against a' in words: relative change in warm Recall@10 and in exposure."""
    warm = f"{100 * (b['warm'] / a['warm'] - 1):+.1f}%"
    exposure = (f"{100 * (b['exposure'] / a['exposure'] - 1):+.1f}%" if a["exposure"] > 0
                else f"from zero to {100 * b['exposure']:.2f}%")
    return (f"Recall@10 on old videos {warm} ({a['warm']:.4f} to {b['warm']:.4f}), "
            f"exposure {exposure} ({100 * a['exposure']:.2f}% to {100 * b['exposure']:.2f}%)")


def comparisons(curve: list[dict]) -> list[str]:
    """The comparisons the README quotes, computed here so they are never typed by hand."""
    by = {p["id_dropout"]: p for p in curve}
    lines = ["Comparisons along the sweep:", ""]
    if 1.0 in by:
        no_ids = by[1.0]
        for rate in (0.3, 0.0):
            if rate in by:
                lines.append(f"- Adding IDs (dropout {rate:g}) to the model with no IDs (dropout 1): "
                             f"{change(no_ids, by[rate])}.")
        best = max(curve, key=lambda p: p["warm"])
        lines.append(f"- Removing IDs entirely from the best model on old videos (dropout {best['id_dropout']:g}): "
                     f"{change(best, no_ids)}.")
    inner = [p for p in curve if p["id_dropout"] <= 0.75]
    if len(inner) > 1:
        lines.append(f"- From dropout {inner[0]['id_dropout']:g} to {inner[-1]['id_dropout']:g}, Recall@10 on old "
                     f"videos stays between {min(p['warm'] for p in inner):.4f} and "
                     f"{max(p['warm'] for p in inner):.4f} while exposure goes from "
                     f"{100 * inner[0]['exposure']:.2f}% to {100 * inner[-1]['exposure']:.2f}%.")
    return lines


def tradeoff_markdown(points: list[dict]) -> str:
    curve, corrected, others = classify(points)
    lines = ["# Trade-off: accuracy on old videos against exposure of new videos", "",
             "Warm Recall@10: the held-out old video is in the top 10, ranked among all old videos. "
             "Exposure: share of top-10 slots that go to new videos when old and new are ranked together. "
             "Mean ± sample standard deviation over seeds.", ""]
    head = ["| Model | ID dropout | Seeds | Warm Recall@10 | New-video exposure (%) |", "|---|---:|---:|---:|---:|"]
    row = lambda p, name: (f"| {name} | {'' if p['id_dropout'] is None else p['id_dropout']} | {p['seeds']} | "
                           f"{pm(p['warm'], p['warm_std'])} | {pm(p['exposure'], p['exposure_std'], 100, 2)} |")
    if curve:
        lines += ["## ID dropout sweep (IDs + cover + title)", ""] + head
        lines += [row(p, "IDs + cover + title") for p in curve] + [""]
        lines += comparisons(curve) + [""]
    if corrected:
        lines += ["## Popularity correction", "",
                  "Trained with the log of each video's share of training interactions added to its score, "
                  "and scored without it. Change is against the run with the same ID dropout and no correction.", "",
                  "| ID dropout | Correction strength | Seeds | Warm Recall@10 | Change | New-video exposure (%) | "
                  "Change (points) |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for p, base in corrected:
            d_warm = "" if base is None else f"{100 * (p['warm'] / base['warm'] - 1):+.1f}%"
            d_exp = "" if base is None else f"{100 * (p['exposure'] - base['exposure']):+.2f}"
            lines.append(f"| {p['id_dropout']} | {p['pop_correction']} | {p['seeds']} | {pm(p['warm'], p['warm_std'])} | "
                         f"{d_warm} | {pm(p['exposure'], p['exposure_std'], 100, 2)} | {d_exp} |")
        lines.append("")
    if others:
        lines += ["## Other models", ""] + head + [row(p, label(p["name"])) for p in others] + [""]
    return "\n".join(lines)


def tradeoff_chart(points: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve, corrected, others = classify(points)
    fig, ax = plt.subplots(figsize=(8.5, 5.6), facecolor=SURFACE)
    x = lambda p: 100 * p["exposure"]
    for p in others:
        ax.scatter(x(p), p["warm"], s=28, color=MUTED, zorder=2)
        ax.annotate(label(p["name"]), (x(p), p["warm"]), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color=MUTED)
    if curve:
        ax.plot([x(p) for p in curve], [p["warm"] for p in curve], color=WARM_COLOR, linewidth=2, zorder=3)
        ax.errorbar([x(p) for p in curve], [p["warm"] for p in curve],
                    xerr=[100 * (p["exposure_std"] or 0) for p in curve], yerr=[p["warm_std"] or 0 for p in curve],
                    fmt="o", color=WARM_COLOR, ecolor=WARM_COLOR, markersize=6, capsize=2, zorder=4,
                    label="IDs + cover + title, by ID dropout in training")
        for p in curve:
            ax.annotate(f"{p['id_dropout']:g}", (x(p), p["warm"]), xytext=(6, 6), textcoords="offset points",
                        fontsize=9, color=INK)
    for i, (p, base) in enumerate(corrected):
        ax.scatter(x(p), p["warm"], s=70, marker="D", color=COLD_COLOR, zorder=5,
                   label="with popularity correction" if i == 0 else None)
        if base is not None:
            ax.annotate("", xy=(x(p), p["warm"]), xytext=(x(base), base["warm"]),
                        arrowprops=dict(arrowstyle="->", color=COLD_COLOR, linewidth=1.2), zorder=4)
    ax.set_xlabel("Share of top-10 slots going to new videos (%)")
    ax.set_ylabel("Recall@10 on old videos")
    ax.set_title("Accuracy on old videos against how often new videos are shown", loc="left", color=INK,
                 fontsize=12)
    ax.margins(x=0.22)  # room for the point labels on the right
    ax.set_xlim(left=-0.6)
    style_axis(ax)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
    if curve or corrected:
        ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)
    results = Path(args.results)
    runs = load_runs(results / "runs")
    points = [pt for pt in (point(name, runs[name]) for name in ordered(runs) if name not in TIED_SCORES)
              if pt is not None]
    (results / "tradeoff.md").write_text(tradeoff_markdown(points), encoding="utf-8")
    print(f"wrote {results / 'tradeoff.md'} ({len(points)} models)")
    if points:
        tradeoff_chart(points, results / "tradeoff.png")
        print(f"wrote {results / 'tradeoff.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
