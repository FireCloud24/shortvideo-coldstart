"""Pick the learning rate for one model family from the search runs.

The full run trains one representative model of each family (IDs only, content only,
IDs + content) with seed 0 at each learning rate in a small fixed grid, into
results/lr_search/<family>_lr<rate>/. This script reads those runs and picks the rate
with the best *validation* NDCG@10. Test and cold-start numbers are never looked at.

    python scripts/pick_lr.py sasrec_id            # prints the chosen rate
    python scripts/pick_lr.py --table              # writes results/lr_search/chosen.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FAMILIES = {  # family name -> (representative config, plain name)
    "sasrec_id": "IDs only",
    "content_cnclip_image_text": "Content only (cover + title)",
    "id_content_cnclip_image_text": "IDs + content (cover + title)",
}
DEFAULT_LR = 0.001


def family_of(config_name: str) -> str | None:
    """Which family's learning rate a config uses, from its name."""
    if config_name.startswith("id_content"):
        return "id_content_cnclip_image_text"
    if config_name.startswith("content"):
        return "content_cnclip_image_text"
    if config_name.startswith("sasrec_id"):
        return "sasrec_id"
    return None  # training-free baselines


def search_runs(search_dir: Path, family: str, k: int = 10) -> list[dict]:
    rows = []
    for path in sorted(search_dir.glob(f"{family}_lr*/seed0/metrics.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"lr": float(r["config"]["train"]["lr"]),
                     "valid_ndcg": r["metrics"]["valid"][f"ndcg@{k}"],
                     "best_epoch": r["training"]["best_epoch"], "epochs_run": r["training"]["epochs_run"],
                     "cap": r.get("epochs_cap"), "seconds": r.get("train_seconds")})
    return sorted(rows, key=lambda row: row["lr"])


def pick(rows: list[dict]) -> float:
    """Best validation NDCG@10; a tie goes to the smaller rate. No runs -> the default."""
    if not rows:
        return DEFAULT_LR
    return max(rows, key=lambda row: (row["valid_ndcg"], -row["lr"]))["lr"]


def table_markdown(search_dir: Path) -> str:
    lines = ["# Learning rate per model family", "",
             "One representative model per family, seed 0, at each rate in the grid. The rate with the best "
             "validation NDCG@10 is used for every model of that family. Test and cold-start results "
             "played no part in the choice.", "",
             "| Family | Learning rate | Validation NDCG@10 | Best epoch | Epochs run | Chosen |",
             "|---|---:|---:|---:|---:|:---:|"]
    for family, plain in FAMILIES.items():
        rows = search_runs(search_dir, family)
        chosen = pick(rows)
        for row in rows:
            capped = " (cap)" if row["cap"] and row["best_epoch"] >= row["cap"] else ""
            lines.append(f"| {plain} | {row['lr']:g} | {row['valid_ndcg']:.4f} | {row['best_epoch']}{capped} | "
                         f"{row['epochs_run']} | {'yes' if row['lr'] == chosen else ''} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", help="a config name, e.g. content_cnclip_image")
    p.add_argument("--search-dir", default="results/lr_search")
    p.add_argument("--table", action="store_true")
    args = p.parse_args(argv)
    search_dir = Path(args.search_dir)
    if args.table:
        search_dir.mkdir(parents=True, exist_ok=True)
        (search_dir / "chosen.md").write_text(table_markdown(search_dir), encoding="utf-8")
        print(f"wrote {search_dir / 'chosen.md'}")
        return 0
    if not args.config:
        p.error("give a config name or --table")
    family = family_of(args.config)
    print(f"{pick(search_runs(search_dir, family)) if family else DEFAULT_LR:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
