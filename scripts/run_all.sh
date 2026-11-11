#!/usr/bin/env bash
# Run every experiment in order, cheapest first, so partial results exist early.
# A failing step is reported and the rest still runs.
#
#   nohup bash scripts/run_all.sh > run_all.log 2>&1 &
#   tail -f run_all.log
#
# Assumes scripts/download.py has already been run.

SEEDS="${SEEDS:-0 1 2}"
export PYTHONUNBUFFERED=1  # so progress reaches the log file line by line

step() {
  echo
  echo "=== $(date +%H:%M:%S) $* ==="
  "$@" || echo "!!! FAILED: $*"
}

train() {  # train <config name> [extra args...]
  local name="$1"; shift
  step python scripts/train.py --config "configs/${name}.yaml" --seeds $SEEDS "$@"
}

# 1. Baselines (no features needed)
step python scripts/train.py --config configs/popularity.yaml --seeds 0
train sasrec_id

# 2. Frozen Chinese-CLIP features
step python scripts/embed.py --model cnclip
train content_cnclip_image
train content_cnclip_text
train content_cnclip_image_text
train id_content_cnclip_image_text

# 3. Adaptation: LoRA on the text tower, and the cheap adapter on image features
step python scripts/finetune.py --mode lora --modality text
step python scripts/embed.py --model cnclip --what text --lora-dir checkpoints/lora_text --tag lora
train content_cnclip_text_lora
step python scripts/finetune.py --mode adapter --feature cnclip_image
train content_cnclip_image --set name=content_cnclip_image_adapter "model.features=[cnclip_image_adapter]"

# 4. Qwen2-VL (the slow embedding step)
step python scripts/embed.py --model qwen
train content_qwen_vl
train id_content_qwen_vl

# 5. Report
step python scripts/report.py
echo
echo "=== $(date +%H:%M:%S) ALL DONE ==="
