#!/usr/bin/env bash
# The vector-length test: retrain two ID + content models with their weights saved,
# then run scripts/scale_test.py on them. About half an hour on a T4.
#
#   bash scripts/run_scale_test.sh
#
# Assumes scripts/download.py has been run. The models are trained into
# results/scale_test/runs, so the main results under results/runs are not touched.
# The learning rate is the one already chosen in results/lr_search.

SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-id_content_dropout_000 id_content_dropout_050}"
export PYTHONUNBUFFERED=1

[ -e data/features/cnclip_image.npy ] || python scripts/embed.py --model cnclip || exit 1

for name in $MODELS; do
  lr="$(python scripts/pick_lr.py "$name")"
  echo "=== $(date +%H:%M:%S) $name (learning rate $lr) ==="
  pids=()
  for seed in $SEEDS; do
    python scripts/train.py --config "configs/${name}.yaml" --seeds "$seed" --skip-existing \
      --set "train.lr=${lr}" train.save_model=true out_dir=results/scale_test/runs &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || echo "!!! FAILED: $name"; done
done

echo "=== $(date +%H:%M:%S) scale test ==="
python scripts/scale_test.py --models $MODELS --seeds $SEEDS
echo "=== $(date +%H:%M:%S) SCALE TEST DONE ==="
