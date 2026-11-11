"""Train and evaluate one configured model for one or more seeds.

    python scripts/train.py --config configs/sasrec_id.yaml --seeds 0 1 2
    python scripts/train.py --config configs/sasrec_id.yaml --set train.epochs=1 out_dir=results/smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data import build_split, load_interactions  # noqa: E402
from src.train import run  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    split = build_split(load_interactions(cfg["data"]["pairs"]), cfg["data"]["cold_item_frac"],
                        cfg["data"]["min_warm_len"])
    print(f"split: {split.summary()}")
    for seed in args.seeds:
        run(cfg, seed, split=split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
