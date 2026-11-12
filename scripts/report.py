"""Build the results table, the warm/cold chart and the error slices.

Reads only what training saved under results/runs/ (and results/finetune/), so
every number in the output traces back to a metrics.json. Nothing is retrained.

    python scripts/report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Row order of the table; anything else found on disk is appended alphabetically.
ORDER = [
    "popularity", "popularity_recent", "popularity_category", "knn_cnclip_image", "knn_cnclip_image_text",
    "sasrec_id",
    "content_cnclip_image", "content_cnclip_text", "content_enclip_text",
    "content_cnclip_image_text", "content_cnclip_image_enclip_text", "content_qwen_vl",
    "id_content_cnclip_image_text", "id_content_qwen_vl",
    "content_cnclip_text_lora", "content_cnclip_image_lora",
]
SETTINGS = [("warm", "Warm"), ("warm_matched", "Warm, matched"), ("cold", "Cold"),
            ("cold_all", "Cold vs all items")]
CANDIDATES = {"warm": "n_warm_items", "warm_matched": "n_cold_items", "cold": "n_cold_items",
              "cold_all": "n_items"}

# (model A, model B, setting): paired bootstrap of NDCG@10(A) - NDCG@10(B) over evaluation examples.
PAIRS = [
    ("content_cnclip_image_text", "content_cnclip_image", "cold"),
    ("content_cnclip_image_text", "content_cnclip_image", "cold_all"),
    ("id_content_cnclip_image_text", "sasrec_id", "warm"),
    ("id_content_cnclip_image_text", "content_cnclip_image_text", "cold"),
    ("content_cnclip_image", "knn_cnclip_image", "cold"),
    ("content_enclip_text", "content_cnclip_text", "cold"),
    ("content_cnclip_text_lora", "content_cnclip_text", "cold"),
    ("content_cnclip_text_lora", "content_cnclip_text", "cold_all"),
    ("content_cnclip_image_lora", "content_cnclip_image", "cold"),
    ("content_cnclip_image_lora", "content_cnclip_image", "cold_all"),
]
BOOTSTRAP_SAMPLES = 2000

# Chart colours: colour-blind safe, at least 3:1 contrast on the surface colour.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
WARM_COLOR, COLD_COLOR = "#2a78d6", "#eb6834"


# ------------------------------------------------------------------ loading

def load_runs(runs_dir: Path) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for path in sorted(runs_dir.glob("*/seed*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["_dir"] = path.parent
        runs[result["name"]].append(result)
    return dict(runs)


def ordered(names) -> list[str]:
    names = list(names)
    return [n for n in ORDER if n in names] + sorted(n for n in names if n not in ORDER)


def mean_std(values: list[float]) -> tuple[float, float | None]:
    """Mean and sample standard deviation (None when there is a single seed)."""
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), None
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def fmt(values: list[float]) -> str:
    mean, std = mean_std(values)
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"


def metric(runs: list[dict], setting: str, name: str) -> list[float]:
    # Runs made before a setting existed simply have no number for it.
    return [r["metrics"].get(setting, {}).get(name) for r in runs]


# ------------------------------------------------------------------ paired comparisons

def per_example_ndcg(run_dir: Path, setting: str) -> np.ndarray | None:
    """NDCG@10 of every evaluation example, from the top-10 ranks saved by training."""
    path = run_dir / "top10_ranks.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        if setting not in saved:
            return None
        ranks = saved[setting].astype(np.float64)
    gains = np.zeros_like(ranks)
    np.divide(1.0, np.log2(ranks + 1.0), out=gains, where=ranks > 0)
    return gains


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> dict:
    """Mean of a - b with a 95% percentile interval, resampling examples with replacement.

    ``a`` and ``b`` hold one value per evaluation example, in the same order, so the
    difference is taken example by example before resampling (a paired comparison).
    """
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in shape: {a.shape} vs {b.shape}")
    diff = a - b
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_samples)])
    low, high = np.quantile(means, [0.025, 0.975])
    return {"diff": float(diff.mean()), "low": float(low), "high": float(high), "n": int(len(diff))}


def paired_rows(runs: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for a, b, setting in PAIRS:
        if a not in runs or b not in runs:
            continue
        by_seed_a = {r["seed"]: per_example_ndcg(r["_dir"], setting) for r in runs[a]}
        by_seed_b = {r["seed"]: per_example_ndcg(r["_dir"], setting) for r in runs[b]}
        seeds = sorted(s for s in by_seed_a if by_seed_a[s] is not None and by_seed_b.get(s) is not None)
        if not seeds:
            continue
        # Average each example over the seeds first, so seed noise is not counted as examples.
        mean_a = np.mean([by_seed_a[s] for s in seeds], axis=0)
        mean_b = np.mean([by_seed_b[s] for s in seeds], axis=0)
        rows.append({"a": a, "b": b, "setting": setting, "seeds": seeds, **paired_bootstrap(mean_a, mean_b)})
    return rows


def paired_markdown(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    labels = dict(SETTINGS)
    lines = ["", "## Paired comparisons", "",
             f"NDCG@10 of A minus B, per evaluation example (averaged over seeds), with a 95% bootstrap interval",
             f"over examples ({BOOTSTRAP_SAMPLES:,} resamples). An interval that excludes 0 means the difference is",
             "larger than the example-to-example noise. It says nothing about a different data split.", "",
             "| A | B | Setting | Examples | A − B | 95% interval | Excludes 0 |",
             "|---|---|---|---:|---:|---|---|"]
    for row in rows:
        excludes = "yes" if row["low"] > 0 or row["high"] < 0 else "no"
        lines.append(f"| `{row['a']}` | `{row['b']}` | {labels[row['setting']]} | {row['n']:,} | "
                     f"{row['diff']:+.4f} | {row['low']:+.4f} to {row['high']:+.4f} | {excludes} |")
    return lines


def training_length_markdown(runs: dict[str, list[dict]], names: list[str]) -> list[str]:
    lines = ["", "## Training length", "",
             "Early stopping ends a run after 5 epochs without a better validation NDCG@10. A best epoch close",
             "to the cap would mean the run was cut off while still improving.", "",
             "| Model | Epoch cap | Epochs run (per seed) | Best epoch (per seed) |", "|---|---:|---|---|"]
    for name in names:
        rs = sorted(runs[name], key=lambda r: r["seed"])
        if not any(r.get("training", {}).get("epochs_run") for r in rs):
            continue  # training-free baselines
        caps = {r.get("epochs_cap") or r.get("config", {}).get("train", {}).get("epochs", "n/a") for r in rs}
        cap = "/".join(sorted(str(c) for c in caps))
        lines.append(f"| `{name}` | {cap} | {'/'.join(str(r['training']['epochs_run']) for r in rs)} | "
                     f"{'/'.join(str(r['training']['best_epoch']) for r in rs)} |")
    return lines


def features_markdown(features_dir: Path) -> list[str]:
    metas = sorted(features_dir.glob("*.meta.json")) if features_dir.exists() else []
    if not metas:
        return []
    lines = ["", "## Content features", "",
             "Copied from the `.meta.json` written next to each cached feature file. \"Zero rows\" are items",
             "with no vector (for image features: no cover file); they all share the same all-zero vector.", "",
             "| Feature | Encoder | Dimensions | Items embedded | Zero rows | Seconds |", "|---|---|---:|---:|---:|---:|"]
    for path in metas:
        meta = json.loads(path.read_text(encoding="utf-8"))
        name = path.name[: -len(".meta.json")]
        embedded = meta.get("n_items_embedded")
        lines.append(f"| `{name}` | {meta.get('model', 'n/a')} | {meta['shape'][1]} | "
                     f"{embedded if embedded is not None else 'n/a'} | {meta.get('zero_rows', 'n/a')} | "
                     f"{meta.get('seconds', 'n/a')} |")
    return lines


# ------------------------------------------------------------------ results.md

def results_markdown(runs: dict[str, list[dict]], finetune_logs: list[dict], k: int = 10,
                     features_dir: Path | None = None) -> str:
    names = ordered(runs)
    splits = {json.dumps(r["split"], sort_keys=True) for rs in runs.values() for r in rs}
    split = next(iter(runs.values()))[0]["split"]
    lines = [
        "# Results",
        "",
        "Generated by `scripts/report.py` from `results/runs/*/seed*/metrics.json`. Do not edit by hand.",
        "",
        f"Metrics are Recall@{k} and NDCG@{k} with full ranking, mean ± sample standard deviation over seeds.",
        "",
    ]
    if len(splits) > 1:
        lines += ["> **Warning:** the runs below were not all trained on the same split, so they are "
                  "not comparable. Re-run them with one config.", ""]
    lines += ["## Split", "", "| | |", "|---|---:|"]
    lines += [f"| {key.replace('_', ' ')} | {value:,} |" for key, value in split.items()]
    lines += [
        "",
        "## Main table",
        "",
        "The settings rank the target among different numbers of candidates, so compare models **within a",
        "column group**. The one fair warm-against-cold comparison is \"Warm, matched\" against \"Cold\": both rank",
        "the target among the same number of items (a fixed random sample of warm items, the size of the cold",
        "set). \"Cold vs all items\" is the realistic setting: a new video competing with every established one.",
        "A uniformly random ranker scores about Recall@10 = 10 / candidates (last row).",
        "",
        "| Model | Seeds | " + " | ".join(f"{label} R@{k} | {label} N@{k}" for _, label in SETTINGS) + " | Train s |",
        "|---|---:|" + "---:|" * (2 * len(SETTINGS) + 1),
    ]
    for name in names:
        rs = runs[name]
        cells = []
        for setting, _ in SETTINGS:
            cells += [fmt(metric(rs, setting, f"recall@{k}")), fmt(metric(rs, setting, f"ndcg@{k}"))]
        seconds = mean_std([r["train_seconds"] for r in rs])[0]
        lines.append(f"| `{name}` | {len(rs)} | " + " | ".join(cells) + f" | {seconds:.0f} |")
    chance = " | ".join(f"{k / split[CANDIDATES[s]]:.4f} | " for s, _ in SETTINGS)  # R@k cell, empty N@k cell
    lines.append(f"| *random ranker (expected)* | | {chance}| |")

    if finetune_logs:
        lines += ["", "## Fine-tuning cost (RQ3)", "",
                  "| Stage | Steps | Seconds | Peak GPU MB | Trainable encoder params | First loss | Last loss |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for log in finetune_logs:
            peak = log["peak_gpu_mb"] if log.get("peak_gpu_mb") is not None else "n/a"
            lines.append(f"| `{log['name']}` | {log['steps']:,} | {log['seconds']:.0f} | {peak} | "
                         f"{log['n_trainable_encoder_params']:,} | {log['first_loss']:.3f} | {log['last_loss']:.3f} |")
        lines += ["", "First and last loss are the mean of the first and last 10 steps (in-batch softmax).",
                  "Warm-started = the recommender on top of the encoder began from a trained content model.", "",
                  "| Stage | Warm-started | Encoder lr | Head lr | Mean loss per epoch |", "|---|---|---:|---:|---|"]
        for log in finetune_logs:
            per_epoch = ", ".join(f"{x:.3f}" for x in log.get("epoch_mean_loss", [])) or "n/a"
            lines.append(f"| `{log['name']}` | {'yes' if log.get('warm_started') else 'no'} | "
                         f"{log.get('lr', log['args']['lr'])} | {log.get('head_lr', log['args']['lr'])} | {per_epoch} |")
    lines += training_length_markdown(runs, names)
    lines += paired_markdown(paired_rows(runs))
    if features_dir is not None:
        lines += features_markdown(features_dir)
    seeds = sorted({r["seed"] for rs in runs.values() for r in rs})
    heads = sorted({str(r.get("git_head"))[:7] for rs in runs.values() for r in rs})
    lines += ["", f"Seeds: {seeds}. Code version(s) (git): {', '.join(heads)}.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ chart helpers

def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbar_panel(ax, labels, means, stds, color, title, xlabel):
    y = np.arange(len(labels))[::-1]
    err = [s if s is not None else 0.0 for s in stds]
    from matplotlib.ticker import MaxNLocator

    has_err = any(s is not None for s in stds)  # no error bars when there is a single seed
    ax.barh(y, means, height=0.55, color=color, xerr=err if has_err else None,
            error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_yticks(y, labels)
    for yi, m, e in zip(y, means, err):
        ax.text(m + e + 0.01 * max(max(means), 1e-9), yi, f" {m:.3f}", va="center", ha="left", color=INK, fontsize=9)
    ax.set_xlim(0, max(max(means), 1e-9) * 1.25)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    style_axis(ax)


def warm_cold_chart(runs: dict[str, list[dict]], path: Path, k: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ordered(runs)
    split = next(iter(runs.values()))[0]["split"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(names) + 1.8), sharey=True, facecolor=SURFACE)
    panels = [("warm", WARM_COLOR, f"Warm items (ranked among {split['n_warm_items']:,})"),
              ("cold", COLD_COLOR, f"Cold-start items (ranked among {split['n_cold_items']:,})")]
    for ax, (setting, color, title) in zip(axes, panels):
        stats = [mean_std(metric(runs[n], setting, f"ndcg@{k}")) for n in names]
        means = [0.0 if np.isnan(m) else m for m, _ in stats]
        hbar_panel(ax, names, means, [s for _, s in stats], color, title, f"NDCG@{k} (mean, bars: ± 1 std over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ error slices

def load_item_metadata(raw_dir: Path) -> dict[str, dict[int, str]] | None:
    """Slice labels per original item id, or None when the raw data is not on disk."""
    from src.features import read_titles

    files = {name: raw_dir / name for name in
             ("tags_to_summary.csv", "MicroLens-100k_title_en.csv", "MicroLens-100k_likes_and_views.txt")}
    if not all(p.exists() for p in files.values()):
        return None
    category = {int(i): c.strip() for i, _, c in
                (line.partition(",") for line in files["tags_to_summary.csv"].read_text(encoding="utf-8").splitlines()
                 if line.strip())}
    words = {i: len(t.split()) for i, t in read_titles(files["MicroLens-100k_title_en.csv"]).items()}
    likes = {}
    for line in files["MicroLens-100k_likes_and_views.txt"].read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip().isdigit():
            likes[int(parts[0])] = float(parts[1])

    def length_bucket(n):
        return "1-5 words" if n <= 5 else "6-10 words" if n <= 10 else "11-20 words" if n <= 20 else "21+ words"

    return {"category": category,
            "title length": {i: length_bucket(n) for i, n in words.items()},
            "_likes": likes}


def like_buckets(likes: dict[int, float], items: set[int]) -> dict[int, str]:
    """Quartiles of platform-wide likes, computed over the cold items only."""
    have = sorted(i for i in items if i in likes)
    if len(have) < 4:
        return {}
    cuts = np.quantile([likes[i] for i in have], [0.25, 0.5, 0.75])
    labels = ["Q1 (fewest likes)", "Q2", "Q3", "Q4 (most likes)"]
    return {i: labels[int(np.searchsorted(cuts, likes[i], side="left"))] for i in have}


def slice_tables(runs: dict[str, list[dict]], meta: dict, k: int = 10, top_categories: int = 8):
    """{dimension: {slice: {model: recall}}} plus example counts per slice."""
    per_item = {}  # model -> list of rows (all seeds)
    for name, rs in runs.items():
        rows = []
        for r in rs:
            path = r["_dir"] / "cold_by_item.json"
            if path.exists():
                rows += json.loads(path.read_text(encoding="utf-8"))
        if rows:
            per_item[name] = rows
    if not per_item:
        return {}, {}
    cold_items = {row["item"] for rows in per_item.values() for row in rows}
    dims = {"category": meta["category"], "title length": meta["title length"],
            "platform popularity": like_buckets(meta["_likes"], cold_items)}

    first = next(iter(per_item.values()))
    cat_counts = defaultdict(int)
    for row in first:
        cat_counts[meta["category"].get(row["item"], "Unknown")] += row["n"]
    keep = {c for c, _ in sorted(cat_counts.items(), key=lambda kv: -kv[1])[:top_categories]}

    tables, counts = {}, {}
    for dim, labels in dims.items():
        table, count = defaultdict(dict), defaultdict(int)
        for name, rows in per_item.items():
            hits, total = defaultdict(float), defaultdict(int)
            for row in rows:
                label = labels.get(row["item"], "Unknown")
                if dim == "category" and label not in keep:
                    label = "Other categories"
                hits[label] += row[f"recall@{k}"] * row["n"]
                total[label] += row["n"]
            for label in total:
                table[label][name] = hits[label] / total[label]
                if name == next(iter(per_item)):
                    count[label] = total[label] // max(len(runs[name]), 1)
        tables[dim], counts[dim] = dict(table), dict(count)
    return tables, counts


def slice_order(dim: str, labels, counts) -> list[str]:
    if dim == "category":
        return sorted(labels, key=lambda label: (label == "Other categories", -counts.get(label, 0)))
    return sorted(labels, key=lambda s: (int(s.split("-")[0].split("+")[0]) if s[0].isdigit() else 0, s))


def slices_markdown(tables, counts, names, k: int = 10) -> str:
    lines = ["# Error slices: cold-start items", "",
             "Generated by `scripts/report.py` from `results/runs/*/seed*/cold_by_item.json`.",
             f"Cold Recall@{k} (ranked among cold items), pooled over seeds, by attribute of the target video.",
             "\"Platform popularity\" is the like count published with the dataset, split into quartiles of the cold items.", ""]
    for dim, table in tables.items():
        lines += [f"## By {dim}", "", "| Slice | Examples | " + " | ".join(f"`{n}`" for n in names) + " |",
                  "|---|---:|" + "---:|" * len(names)]
        for label in slice_order(dim, table, counts[dim]):
            cells = [f"{table[label][n]:.4f}" if n in table[label] else "n/a" for n in names]
            lines.append(f"| {label} | {counts[dim].get(label, 0):,} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def slices_chart(tables, counts, model: str, path: Path, k: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = [d for d in tables if tables[d]]
    rows = max(len(tables[d]) for d in dims)
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 0.42 * rows + 1.8), facecolor=SURFACE)
    for ax, dim in zip(np.atleast_1d(axes), dims):
        labels = slice_order(dim, tables[dim], counts[dim])
        means = [tables[dim][label].get(model, 0.0) for label in labels]
        hbar_panel(ax, labels, means, [None] * len(labels), COLD_COLOR, f"By {dim}", f"Cold Recall@{k}")
    fig.suptitle(f"Where cold-start recommendation fails: {model}", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def best_cold_model(runs, k: int = 10) -> str:
    return max(runs, key=lambda n: np.nan_to_num(mean_std(metric(runs[n], "cold", f"ndcg@{k}"))[0], nan=-1.0))


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--raw-dir", default="data/raw")
    args = p.parse_args(argv)

    results = Path(args.results_dir)
    runs = load_runs(results / "runs")
    if not runs:
        print(f"no runs found under {results / 'runs'}; train something first (scripts/train.py)")
        return 1
    logs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((results / "finetune").glob("*.json"))]

    (results / "results.md").write_text(results_markdown(runs, logs, features_dir=results / "features"),
                                        encoding="utf-8")
    warm_cold_chart(runs, results / "warm_vs_cold.png")
    print(f"wrote {results / 'results.md'} and {results / 'warm_vs_cold.png'} ({len(runs)} models)")

    meta = load_item_metadata(Path(args.raw_dir))
    if meta is None:
        print(f"skipping error slices: item metadata not found in {args.raw_dir}")
        return 0
    tables, counts = slice_tables(runs, meta)
    if tables:
        (results / "error_slices.md").write_text(slices_markdown(tables, counts, ordered(runs)), encoding="utf-8")
        best = best_cold_model(runs)
        slices_chart(tables, counts, best, results / "error_slices.png")
        print(f"wrote {results / 'error_slices.md'} and {results / 'error_slices.png'} (chart: {best})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
