#!/usr/bin/env bash
# Run the experiments: the core result first, the slow extras only with EXTRAS=1.
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

# MAX_HOURS: start no new step after this many hours, so a session with a time limit
# (Kaggle: 12 hours) ends cleanly with everything saved. Run again to continue.
MAX_HOURS="${MAX_HOURS:-}"
OUT_OF_TIME=0

step() {
  if [ -n "$MAX_HOURS" ] && [ "$SECONDS" -ge "$(awk "BEGIN{print int($MAX_HOURS*3600)}")" ]; then
    [ "$OUT_OF_TIME" = 0 ] && echo && echo "=== $(date +%H:%M:%S) OUT OF TIME after ${MAX_HOURS}h: skipping the remaining steps ==="
    OUT_OF_TIME=1
    return 0
  fi
  echo
  echo "=== $(date +%H:%M:%S) $* ==="
  "$@" || echo "!!! FAILED: $*"
  persist
}

always() {  # a step that runs even when out of time (the final tables)
  echo
  echo "=== $(date +%H:%M:%S) $* ==="
  "$@" || echo "!!! FAILED: $*"
  persist
}

# Several training processes share the GPU: the models are small, so one run leaves most of
# it idle. Each seed is its own process; with two GPUs (Kaggle T4 x2) they alternate.
PARALLEL="${PARALLEL:-1}"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"; [ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1

together() {  # together <command 1> -- <command 2> -- ...: run the commands at once, fail if any fails
  local -a cmd=() pids=()
  local i=0 failed=0 arg
  for arg in "$@" --; do
    if [ "$arg" = "--" ]; then
      [ "${#cmd[@]}" -gt 0 ] || continue
      if [ "$PARALLEL" = 1 ]; then
        CUDA_VISIBLE_DEVICES=$((i % NGPU)) "${cmd[@]}" &
        pids+=($!)
      else
        "${cmd[@]}" || failed=1
      fi
      i=$((i + 1)); cmd=()
    else
      cmd+=("$arg")
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return $failed
}

train() {  # train <config name> [extra key=value overrides...]
  # Every model uses the learning rate chosen for its family by search_lr. One process per seed.
  local name="$1"; shift
  local lr; lr="$(python scripts/pick_lr.py "$name")"
  local -a all=()
  local seed
  for seed in $SEEDS; do
    all+=(python scripts/train.py --config "configs/${name}.yaml" --seeds "$seed" --skip-existing
          --set "train.lr=${lr}" "$@" --)
  done
  step together "${all[@]}"
}

search_lr() {  # search_lr <family config>: seed 0 at each rate in the grid, judged on validation
  local family="$1"
  local -a all=()
  local lr
  for lr in $LR_GRID; do
    all+=(python scripts/train.py --config "configs/${family}.yaml" --seeds 0 --skip-existing
          --set "train.lr=${lr}" out_dir=results/lr_search "name=${family}_lr${lr}" --)
  done
  step together "${all[@]}"
}
LR_GRID="${LR_GRID:-0.001 0.003 0.01}"

once() {  # once <file or folder that marks the step as done> <command...>
  local marker="$1"; shift
  if [ -e "$marker" ]; then
    echo
    echo "=== $(date +%H:%M:%S) already done ($marker), skipping: $* ==="
  else
    step "$@"
  fi
}

report_all() {  # report_all <step|always>: every table and figure from what is on disk
  mkdir -p results/features
  # The feature .meta.json files hold no data, only how each feature file was made.
  cp data/features/*.meta.json results/features/ 2>/dev/null || echo "!!! no feature meta files to copy"
  "$1" python scripts/pick_lr.py --table
  "$1" python scripts/report.py
  "$1" python scripts/exposure.py
  "$1" python scripts/tradeoff.py
  "$1" python scripts/categories.py
}

# ---------------------------------------------------------------- the core result
# 1. Baselines that need no content features, and the IDs-only model
step python scripts/train.py --config configs/popularity.yaml --seeds 0 --skip-existing
step python scripts/train.py --config configs/popularity_recent.yaml --seeds 0 --skip-existing
step python scripts/train.py --config configs/popularity_category.yaml --seeds 0 --skip-existing
search_lr sasrec_id
train sasrec_id

# 2. Frozen Chinese-CLIP features (cover and title), the training-free content baselines,
#    and the learning rate for the two families that use content
once data/features/cnclip_image.npy python scripts/embed.py --model cnclip
step python scripts/train.py --config configs/knn_cnclip_image.yaml --seeds 0 --skip-existing
step python scripts/train.py --config configs/knn_cnclip_image_text.yaml --seeds 0 --skip-existing
search_lr content_cnclip_image_text
search_lr id_content_cnclip_image_text

# 3. The trade-off curve first, because it is the finding: the ID + content model at six ID
#    dropout rates (1.0 is content only), then the popularity correction at two of them.
for rate in 000 025 050 075 090 100; do
  train "id_content_dropout_${rate}"
done
train id_content_dropout_000_popcorrect
train id_content_dropout_050_popcorrect
report_all step

# 4. The other points on the figure and the category comparison (cover against title).
#    save_model: the seed-0 weights are the starting point for LoRA in the extras.
train content_cnclip_image_text
train id_content_cnclip_image_text
train content_cnclip_image train.save_model=true
train content_cnclip_text train.save_model=true
report_all step

# ---------------------------------------------------------------- extras (EXTRAS=1)
# Not needed for the finding: English CLIP titles, LoRA inside a tower (preliminary) and
# Qwen2-VL. They are slow, so they run only when asked for.
if [ "${EXTRAS:-0}" = 1 ]; then
  once data/features/enclip_text.npy python scripts/embed.py --model enclip
  train content_enclip_text
  train content_cnclip_image_enclip_text

  # LoRA: the image tower is slow (every step decodes and encodes covers), so it gets a step budget.
  once checkpoints/lora_text python scripts/finetune.py --mode lora --modality text --epochs 3 --init-from checkpoints/content_cnclip_text_seed0.pt
  once data/features/cnclip_text_lora.npy python scripts/embed.py --model cnclip --what text --lora-dir checkpoints/lora_text --tag lora
  train content_cnclip_text_lora
  once checkpoints/lora_image python scripts/finetune.py --mode lora --modality image --epochs 1 --max-steps "${IMAGE_LORA_STEPS:-3000}" --init-from checkpoints/content_cnclip_image_seed0.pt
  once data/features/cnclip_image_lora.npy python scripts/embed.py --model cnclip --what image --lora-dir checkpoints/lora_image --tag lora
  train content_cnclip_image_lora

  once data/features/qwen_vl.npy python scripts/embed.py --model qwen
  train content_qwen_vl
  train id_content_qwen_vl
fi

# ---------------------------------------------------------------- final tables
report_all always
echo
if [ "$OUT_OF_TIME" = 1 ]; then
  echo "=== $(date +%H:%M:%S) STOPPED FOR TIME: run again to continue ==="
else
  echo "=== $(date +%H:%M:%S) ALL DONE ==="
fi
# Keep the log with the results, so every runtime in the README can be traced.
[ -f run_all.log ] && cp run_all.log results/run_all.log
persist
