"""Adapt the item encoder with the recommendation loss (see src/finetune.py).

LoRA on a Chinese-CLIP tower (GPU), then re-embed and train as usual. The
recommender on top of the tower starts from a trained content model, which
scripts/train.py saves when train.save_model is set:

    python scripts/train.py --config configs/content_cnclip_text.yaml --seeds 0 --set train.save_model=true
    python scripts/finetune.py --mode lora --modality text \
        --init-from checkpoints/content_cnclip_text_seed0.pt
    python scripts/embed.py --model cnclip --what text --lora-dir checkpoints/lora_text --tag lora
    python scripts/train.py --config configs/content_cnclip_text_lora.yaml --seeds 0 1 2

The image tower is the same with --modality image (batch 8, gradient checkpointing).

A 2-layer adapter on cached features is kept for memory-limited machines. It is
not part of the reported comparison: the downstream model already trains an MLP
on the same features, so an adapter adds nothing new.

    python scripts/finetune.py --mode adapter --feature cnclip_image
    python scripts/train.py --config configs/content_cnclip_image.yaml --seeds 0 1 2 \
        --set name=content_cnclip_image_adapter "model.features=[cnclip_image_adapter]"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features as feats  # noqa: E402
from src import finetune as ft  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import build_split, load_interactions  # noqa: E402

# Items per step = batch_size * (max_len + 1) at most; images cost far more memory than titles.
DEFAULT_BATCH = {"text": 32, "image": 8, "adapter": 256}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["lora", "adapter"], required=True)
    p.add_argument("--modality", choices=["text", "image"], default="text", help="lora: which tower to adapt")
    p.add_argument("--feature", default="cnclip_image", help="adapter: cached feature to adapt")
    p.add_argument("--model-name", default=feats.CNCLIP_DEFAULT)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--init-from", default=None,
                   help="state dict of a trained sasrec_content model to start the projection and trunk from")
    p.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of the projection and trunk")
    p.add_argument("--no-grad-checkpointing", action="store_true",
                   help="image LoRA: keep all activations in memory (faster, needs more GPU memory)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-len", type=int, default=10, help="history length during fine-tuning")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate of the LoRA matrices / adapter")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--log-dir", default="results/finetune")
    args = p.parse_args(argv)

    cfg = load_config(overrides=["name=finetune"])
    m, d = cfg["model"], cfg["data"]
    split = build_split(load_interactions(d["pairs"]), d["cold_item_frac"], d["min_warm_len"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = Path(args.raw_dir)

    if args.mode == "lora":
        from transformers import AutoImageProcessor, AutoTokenizer, ChineseCLIPModel

        name = f"lora_{args.modality}"
        base = ChineseCLIPModel.from_pretrained(args.model_name)
        grad_checkpointing = False
        if args.modality == "image" and not args.no_grad_checkpointing:
            grad_checkpointing = ft.enable_grad_checkpointing(base)
        clip = ft.add_lora(base, args.modality, r=args.lora_r)
        if args.modality == "text":
            titles = feats.read_titles(raw / "MicroLens-100k_title_en.csv")
            by_index = [""] + [titles.get(int(i), "") for i in split.item_ids[1:]]
            encoder = ft.TextItemEncoder(clip, AutoTokenizer.from_pretrained(args.model_name), by_index)
        else:
            covers = feats.find_covers(raw / "covers")
            by_index = [None] + [covers.get(int(i)) for i in split.item_ids[1:]]
            encoder = ft.ImageItemEncoder(clip, AutoImageProcessor.from_pretrained(args.model_name), by_index)
        feat_dim = clip.config.projection_dim
        batch_size = args.batch_size or DEFAULT_BATCH[args.modality]
    else:
        name = f"{args.feature}_adapter"
        table = feats.load_feature_table(d["features_dir"], [args.feature], split.item_ids)
        encoder = ft.CachedItemEncoder(table)
        feat_dim = table.shape[1]
        batch_size = args.batch_size or DEFAULT_BATCH["adapter"]

    # The trunk is built with the downstream model's max_len so that its weights can be loaded;
    # fine-tuning still feeds histories of args.max_len items (they use the last positions).
    trunk_len = d["max_len"] if args.init_from else args.max_len
    model = ft.InBatchRecommender(encoder, feat_dim, m["dim"], trunk_len, m["n_layers"], m["n_heads"], m["dropout"])
    if args.init_from:
        model.load_downstream(torch.load(args.init_from, map_location="cpu"))
        print(f"{name}: projection and trunk loaded from {args.init_from}")
    else:
        print(f"{name}: WARNING no --init-from; the projection and trunk start from random weights")
    print(f"{name}: fine-tuning on {device}, batch {batch_size}, history {args.max_len}, "
          f"lr {args.lr} (encoder) / {args.head_lr} (projection and trunk)")
    cost = ft.finetune(model, split, args.max_len, args.epochs, batch_size, args.lr, device, args.seed,
                       max_steps=args.max_steps, head_lr=args.head_lr)

    record = {"name": name, "mode": args.mode, "args": vars(args), "batch_size": batch_size,
              "warm_started": bool(args.init_from), **cost}
    if args.mode == "lora":
        record["grad_checkpointing"] = grad_checkpointing
    if args.mode == "lora":
        out = Path(args.out_dir) / name
        clip.save_pretrained(out)  # LoRA weights only
        record["adapter_dir"] = str(out)
        print(f"saved LoRA adapter to {out}; now run scripts/embed.py with --lora-dir {out} --tag lora")
    else:
        n_rows = int(split.item_ids.max()) + 1
        arr = ft.export_adapted_features(encoder, split.item_ids, n_rows)
        path = feats.save_features(d["features_dir"], name, arr, {"model": f"{args.feature} + residual adapter",
                                                                  "seconds": cost["seconds"]})
        record["features"] = str(path)
        print(f"wrote {path}")
    ft.write_cost_log(Path(args.log_dir) / f"{name}.json", record)
    print(f"cost: {cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
