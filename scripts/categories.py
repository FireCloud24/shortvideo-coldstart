"""Which kinds of new video does the cover describe well, and which badly?

For each video category: how many new videos and test examples it has, and the cold
Recall@10 of the cover-only model and of the title-only model, with 95% intervals.
The interval resamples videos, not examples, because all the examples for one video
succeed or fail together (a category with one much-watched video is not well measured).

Reads results/runs/<model>/seed*/cold_by_item.json and the category file that comes
with the dataset. Nothing is retrained.

    python scripts/categories.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exposure import label  # noqa: E402
from report import COLD_COLOR, INK, MUTED, SURFACE, WARM_COLOR, load_runs, style_axis  # noqa: E402

OTHER = "Other categories"


def per_item_recall(runs: list[dict], k: int = 10) -> dict[int, tuple[int, float]]:
    """{item: (examples, recall averaged over seeds)} from the saved per-item results."""
    n, total, seeds = {}, defaultdict(float), defaultdict(int)
    for r in runs:
        path = Path(r["_dir"]) / "cold_by_item.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            n[row["item"]] = row["n"]
            total[row["item"]] += row[f"recall@{k}"]
            seeds[row["item"]] += 1
    return {item: (n[item], total[item] / seeds[item]) for item in n}


def weighted_recall(n: np.ndarray, recall: np.ndarray) -> float:
    return float((n * recall).sum() / n.sum()) if n.sum() else float("nan")


def bootstrap_videos(n: np.ndarray, *recalls: np.ndarray, n_samples: int = 2000, seed: int = 0) -> list[tuple]:
    """95% intervals from resampling videos. One interval per recall array, then, when
    two arrays are given, one for the difference first - second on the same resamples."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(n), size=(n_samples, len(n)))
    weights = n[idx]
    denom = weights.sum(axis=1)
    samples = [(weights * r[idx]).sum(axis=1) / denom for r in recalls]
    if len(samples) == 2:
        samples.append(samples[0] - samples[1])
    return [tuple(float(v) for v in np.percentile(s, [2.5, 97.5])) for s in samples]


def category_rows(cover: dict, title: dict, categories: dict[int, str], top: int = 10,
                  n_samples: int = 2000) -> list[dict]:
    items = sorted(set(cover) & set(title))
    examples = defaultdict(int)
    for item in items:
        examples[categories.get(item, "Unknown")] += cover[item][0]
    keep = {c for c, _ in sorted(examples.items(), key=lambda kv: -kv[1])[:top]}
    groups = defaultdict(list)
    for item in items:
        c = categories.get(item, "Unknown")
        groups[c if c in keep else OTHER].append(item)

    rows = []
    for category, members in groups.items():
        n = np.array([cover[i][0] for i in members], dtype=np.float64)
        rc = np.array([cover[i][1] for i in members])
        rt = np.array([title[i][1] for i in members])
        ci_cover, ci_title, ci_diff = bootstrap_videos(n, rc, rt, n_samples=n_samples)
        rows.append({
            "category": category, "videos": len(members), "examples": int(n.sum()),
            "cover": weighted_recall(n, rc), "cover_ci": ci_cover,
            "title": weighted_recall(n, rt), "title_ci": ci_title,
            "diff": weighted_recall(n, rc) - weighted_recall(n, rt), "diff_ci": ci_diff,
        })
    return sorted(rows, key=lambda r: (r["category"] == OTHER, -r["diff"]))


def categories_markdown(rows: list[dict], cover_name: str, title_name: str, k: int = 10) -> str:
    lines = [
        "# New videos by category: where the cover is enough", "",
        f"Cold Recall@{k} (the new video the user took is in the top {k}, ranked among new videos), "
        f"for the {label(cover_name).lower()} model and the {label(title_name).lower()} model. "
        "Seeds are averaged per video. Intervals are 95%, from resampling videos within the category. "
        "Sorted by the cover's lead over the title.", "",
        "| Category | New videos | Examples | Cover only | 95% interval | Title only | 95% interval | "
        "Cover minus title | 95% interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ci = lambda pair: f"{pair[0]:.3f} to {pair[1]:.3f}"
    for r in rows:
        lines.append(f"| {r['category']} | {r['videos']:,} | {r['examples']:,} | {r['cover']:.3f} | {ci(r['cover_ci'])} | "
                     f"{r['title']:.3f} | {ci(r['title_ci'])} | {r['diff']:+.3f} | {ci(r['diff_ci'])} |")
    lines += ["", "A difference whose interval includes zero is not a difference I can claim.", ""]
    return "\n".join(lines)


def categories_chart(rows: list[dict], cover_name: str, title_name: str, path: Path, k: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = rows[::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.8), facecolor=SURFACE)
    for offset, key, color, name in ((0.17, "cover", COLD_COLOR, cover_name), (-0.17, "title", WARM_COLOR, title_name)):
        values = np.array([r[key] for r in rows])
        low = values - np.array([r[f"{key}_ci"][0] for r in rows])
        high = np.array([r[f"{key}_ci"][1] for r in rows]) - values
        ax.errorbar(values, y + offset, xerr=[np.clip(low, 0, None), np.clip(high, 0, None)], fmt="o", color=color,
                    ecolor=color, markersize=5, capsize=2, label=label(name))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['category']}  ({r['videos']:,} videos)" for r in rows])
    ax.set_xlabel(f"Recall@{k} on new videos, with 95% intervals")
    ax.set_title("Which new videos can be recommended from the cover alone?", loc="left", color=INK, fontsize=12)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    from src.features import read_categories

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--categories", default="data/raw/tags_to_summary.csv")
    p.add_argument("--cover", default="content_cnclip_image")
    p.add_argument("--title", default="content_cnclip_text")
    p.add_argument("--top", type=int, default=10, help="categories kept; the rest are pooled")
    args = p.parse_args(argv)

    results = Path(args.results)
    runs = load_runs(results / "runs")
    if args.cover not in runs or args.title not in runs:
        print(f"skipping categories: need runs of both {args.cover} and {args.title}")
        return 0
    if not Path(args.categories).exists():
        print(f"skipping categories: {args.categories} not found (run scripts/download.py)")
        return 0
    rows = category_rows(per_item_recall(runs[args.cover]), per_item_recall(runs[args.title]),
                         read_categories(args.categories), args.top)
    if not rows:
        print("skipping categories: no per-item results saved")
        return 0
    (results / "categories.md").write_text(categories_markdown(rows, args.cover, args.title), encoding="utf-8")
    categories_chart(rows, args.cover, args.title, results / "categories.png")
    print(f"wrote {results / 'categories.md'} and {results / 'categories.png'} ({len(rows)} categories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
