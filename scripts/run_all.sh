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

# The run can be resumed: finished seeds, feature files and LoRA adapters are skipped.
# If PERSIST_DIR is set (for example a folder on Google Drive), results, features and
# checkpoints are copied there after every step, so a cut-off session loses one step at most.
PERSIST_DIR="${PERSIST_DIR:-}"

persist() {
  [ -n "$PERSIST_DIR" ] || return 0
  mkdir -p "$PERSIST_DIR/features" "$PERSIST_DIR/checkpoints"
  cp -ru results "$PERSIST_DIR"/ 2>/dev/null
  cp -u data/features/*.npy data/features/*.meta.json "$PERSIST_DIR/features/" 2>/dev/null
  [ -d checkpoints ] && cp -ru checkpoints/. "$PERSIST_DIR/checkpoints/" 2>/dev/null
  [ -f run_all.log ] && cp run_all.log "$PERSIST_DIR/run_all.log" 2>/dev/null
  return 0
}

step() {
  echo
  echo "=== $(date +%H:%M:%S) $* ==="
  "$@" || echo "!!! FAILED: $*"
  persist
}

train() {  # train <config name> [extra args...]
  local name="$1"; shift
  step python scripts/train.py --config "configs/${name}.yaml" --seeds $SEEDS --skip-existing "$@"
}

once() {  # once <file or folder that marks the step as done> <command...>
  local marker="$1"; shift
  if [ -e "$marker" ]; then
    echo
    echo "=== $(date +%H:%M:%S) already done ($marker), skipping: $* ==="
  else
    step "$@"
  fi
}

# 1. Baselines that need no content features
step python scripts/train.py --config configs/popularity.yaml --seeds 0 --skip-existing
step python scripts/train.py --config configs/popularity_recent.yaml --seeds 0 --skip-existing
step python scripts/train.py --config configs/popularity_category.yaml --seeds 0 --skip-existing
train sasrec_id

# 2. Frozen features: Chinese-CLIP (image and text) and English CLIP (text; the titles are English)
once data/features/cnclip_image.npy python scripts/embed.py --model cnclip
once data/features/enclip_text.npy python scripts/embed.py --model enclip
step python scripts/train.py --config configs/knn_cnclip_image.yaml --seeds 0 --skip-existing
step python scripts/train.py --config configs/knn_cnclip_image_text.yaml --seeds 0 --skip-existing
# save_model: the seed-0 weights are the starting point for LoRA in step 3
train content_cnclip_image --set train.save_model=true
train content_cnclip_text --set train.save_model=true
train content_enclip_text
train content_cnclip_image_text
train content_cnclip_image_enclip_text
train id_content_cnclip_image_text

# 3. Adaptation (RQ3): LoRA inside a tower, starting from the trained content model.
#    The image tower is slow (every step decodes and encodes covers), so it gets a step budget.
once checkpoints/lora_text python scripts/finetune.py --mode lora --modality text --epochs 3   --init-from checkpoints/content_cnclip_text_seed0.pt
once data/features/cnclip_text_lora.npy python scripts/embed.py --model cnclip --what text --lora-dir checkpoints/lora_text --tag lora
train content_cnclip_text_lora
once checkpoints/lora_image python scripts/finetune.py --mode lora --modality image --epochs 1 --max-steps "${IMAGE_LORA_STEPS:-3000}"   --init-from checkpoints/content_cnclip_image_seed0.pt
once data/features/cnclip_image_lora.npy python scripts/embed.py --model cnclip --what image --lora-dir checkpoints/lora_image --tag lora
train content_cnclip_image_lora

# 4. Qwen2-VL (the slow embedding step)
once data/features/qwen_vl.npy python scripts/embed.py --model qwen
train content_qwen_vl
train id_content_qwen_vl

# 5. Report. The feature .meta.json files hold no data, only how each feature file was made.
mkdir -p results/features
cp data/features/*.meta.json results/features/ 2>/dev/null || echo "!!! no feature meta files to copy"
step python scripts/report.py
echo
echo "=== $(date +%H:%M:%S) ALL DONE ==="
# Keep the log with the results, so every runtime in the README can be traced.
[ -f run_all.log ] && cp run_all.log results/run_all.log
persist
