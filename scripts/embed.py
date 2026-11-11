"""Precompute content embeddings once and cache them as .npy (run on a GPU).

    python scripts/embed.py --model cnclip                # -> cnclip_image.npy, cnclip_text.npy
    python scripts/embed.py --model qwen                  # -> qwen_vl.npy
    python scripts/embed.py --model cnclip --limit 64     # quick check on 64 items (writes *_limit64.npy)
    python scripts/embed.py --model cnclip --what text --lora-dir checkpoints/lora_text --tag lora
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features as F  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["cnclip", "qwen"], required=True)
    p.add_argument("--what", nargs="+", choices=["image", "text"], default=["image", "text"],
                   help="cnclip only: which towers to run")
    p.add_argument("--model-name", default=None, help="Hugging Face model id (defaults per --model)")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="only embed the first N items (smoke test)")
    p.add_argument("--lora-dir", default=None, help="cnclip only: LoRA adapter saved by scripts/finetune.py")
    p.add_argument("--tag", default=None, help="suffix for the output name, e.g. 'lora' -> cnclip_text_lora")
    args = p.parse_args(argv)

    raw = Path(args.raw_dir)
    titles = F.read_titles(raw / "MicroLens-100k_title_en.csv")
    covers = F.find_covers(raw / "covers")
    n_rows = max(titles) + 1
    ids = sorted(titles)[: args.limit]
    missing = [i for i in ids if i not in covers]
    print(f"{len(ids)} items, {len(ids) - len(missing)} with a cover, {len(missing)} without")
    suffix = (f"_{args.tag}" if args.tag else "") + (f"_limit{args.limit}" if args.limit else "")

    def save(name, item_ids, vectors, started, model_name, extra=None):
        meta = {"model": model_name, "n_items_embedded": len(item_ids),
                "seconds": round(time.time() - started, 1), "limit": args.limit,
                "lora_dir": args.lora_dir, **(extra or {})}
        path = F.save_features(args.features_dir, name + suffix, F.scatter_rows(item_ids, vectors, n_rows), meta)
        print(f"wrote {path} {vectors.shape} in {meta['seconds']}s")

    if args.model == "cnclip":
        enc = F.CnClipEncoder(args.model_name or F.CNCLIP_DEFAULT, lora_dir=args.lora_dir)
        bs = args.batch_size or 128
        if "text" in args.what:
            t0 = time.time()
            vecs = F.embed_in_batches(enc.encode_texts, [titles[i] for i in ids], bs)
            save("cnclip_text", ids, vecs, t0, enc.model_name, {"title_language": "en"})
        if "image" in args.what:
            t0 = time.time()
            with_cover = [i for i in ids if i in covers]
            if not with_cover:
                raise SystemExit(f"no covers under {raw / 'covers'}; run scripts/download.py without --skip-covers")
            vecs = F.embed_in_batches(enc.encode_images, [covers[i] for i in with_cover], bs)
            save("cnclip_image", with_cover, vecs, t0, enc.model_name, {"items_without_cover": len(missing)})
    else:
        enc = F.QwenVLEncoder(args.model_name or F.QWEN_DEFAULT)
        bs = args.batch_size or 8
        t0 = time.time()
        pairs = [(titles[i], covers.get(i)) for i in ids]
        vecs = F.embed_in_batches(lambda b: enc.encode_pairs([t for t, _ in b], [c for _, c in b]), pairs, bs)
        save("qwen_vl", ids, vecs, t0, enc.model_name, {"items_without_cover": len(missing)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
