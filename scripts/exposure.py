"""Exposure table and figure: share of top-10 slots that go to new videos, per model.

Reads the ``exposure`` block every run saves in its metrics.json (see src/exposure.py).
Nothing is retrained.

    python scripts/exposure.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import COLD_COLOR, INK, MUTED, SURFACE, load_runs, mean_std, ordered, style_axis  # noqa: E402
from src.exposure import bootstrap_share, summarise_exposure  # noqa: E402

# Plain names for the figures and tables. A model without an entry keeps its config name.
LABELS = {
    "popularity": "Popularity",
    "popularity_recent": "Recent popularity",
    "popularity_category": "Category popularity",
    "knn_cnclip_image": "Nearest covers (no training)",
    "knn_cnclip_image_text": "Nearest covers + titles (no training)",
    "sasrec_id": "IDs only",
    "content_cnclip_image": "Cover only",
    "content_cnclip_text": "Title only",
    "content_enclip_text": "Title only (English CLIP)",
    "content_cnclip_image_text": "Cover + title",
    "content_cnclip_image_enclip_text": "Cover + title (English CLIP)",
    "content_qwen_vl": "Qwen2-VL",
    "id_content_cnclip_image_text": "IDs + cover + title",
    "id_content_qwen_vl": "IDs + Qwen2-VL",
    "content_cnclip_text_lora": "Title only, LoRA (preliminary)",
    "content_cnclip_image_lora": "Cover only, LoRA (preliminary)",
}


def label(name: str) -> str:
    return LABELS.get(name, name)


def pooled_histogram(runs: list[dict]) -> list[int] | None:
    """Users by cold slots, added up over seeds."""
    hists = [r["exposure"]["users_by_cold_slots"] for r in runs if r.get("exposure")]
    if not hists or len({len(h) for h in hists}) != 1:
        return None
    return np.sum(np.asarray(hists), axis=0).tolist()


def exposure_rows(runs: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for name in ordered(runs):
        with_exposure = [r for r in runs[name] if r.get("exposure") and r["exposure"]["cold_slot_share"] is not None]
        if not with_exposure:
            continue
        hist = pooled_histogram(with_exposure)
        pooled = summarise_exposure(hist)
        low, high = bootstrap_share(hist)
        share_mean, share_std = mean_std([r["exposure"]["cold_slot_share"] for r in with_exposure])
        rows.append({
            "name": name, "seeds": len(with_exposure), "k": pooled["k"],
            "n_users": with_exposure[0]["exposure"]["n_users"],
            "share": share_mean, "share_std": share_std, "low": low, "high": high,
            "users_with_any": pooled["users_with_any_cold"],
            "users_with_3_plus": float(sum(hist[3:]) / sum(hist)),
            "catalogue_share": with_exposure[0]["exposure"].get("cold_catalogue_share"),
        })
    return rows


def pct(x: float | None, digits: int = 2) -> str:
    return "" if x is None else f"{100 * x:.{digits}f}%"


def exposure_markdown(rows: list[dict]) -> str:
    if not rows:
        return "# Exposure of new videos\n\nNo run has saved exposure yet.\n"
    k, catalogue = rows[0]["k"], rows[0]["catalogue_share"]
    lines = [
        "# Exposure of new videos", "",
        f"One top-{k} list per warm test user, drawn from all videos, old and new together, with the "
        "videos the user already has removed. The table says how many of those slots new videos got.", "",
        f"New videos are {pct(catalogue, 1)} of the catalogue. A model that spread its slots evenly over "
        "videos would give them that share." if catalogue is not None else "", "",
        f"| Model | Seeds | Share of top-{k} slots | 95% interval | Users shown at least one | "
        "Users shown three or more |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        std = "" if r["share_std"] is None else f" ± {100 * r['share_std']:.2f}"
        lines.append(f"| {label(r['name'])} | {r['seeds']} | {pct(r['share'])}{std} | "
                     f"{pct(r['low'])} to {pct(r['high'])} | {pct(r['users_with_any'], 1)} | "
                     f"{pct(r['users_with_3_plus'], 1)} |")
    lines += ["", "Share: mean over seeds (± sample standard deviation). Interval: bootstrap over users, "
                  "seeds pooled. The per-user columns pool the seeds.", ""]
    return "\n".join(lines)


def exposure_chart(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = rows[::-1]  # first model at the top
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    y = np.arange(len(rows))
    shares = np.array([100 * r["share"] for r in rows])
    err = np.array([[100 * (r["share"] - r["low"]) for r in rows], [100 * (r["high"] - r["share"]) for r in rows]])
    ax.barh(y, shares, color=COLD_COLOR, height=0.62, xerr=np.clip(err, 0, None), ecolor=INK, capsize=2)
    catalogue = rows[0]["catalogue_share"]
    if catalogue is not None:
        ax.axvline(100 * catalogue, color=MUTED, linestyle="--", linewidth=1)
        ax.text(100 * catalogue, len(rows) - 0.4, " share of the catalogue", color=MUTED, fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([label(r["name"]) for r in rows])
    ax.set_xlabel(f"Share of top-{rows[0]['k']} slots going to new videos (%)")
    ax.set_title("How much of the feed do new videos get?", loc="left", color=INK, fontsize=12)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)
    results = Path(args.results)
    rows = exposure_rows(load_runs(results / "runs"))
    (results / "exposure.md").write_text(exposure_markdown(rows), encoding="utf-8")
    print(f"wrote {results / 'exposure.md'} ({len(rows)} models)")
    if rows:
        exposure_chart(rows, results / "exposure.png")
        print(f"wrote {results / 'exposure.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
