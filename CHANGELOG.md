# Changelog

Process notes for this project: re-runs, reviews, problems found and changes of plan.
The README holds only the current findings.

## 2026-11-13

Added, with tests, ahead of the full re-run:

- Exposure: for every run, how many top-10 slots in a mixed feed (old and new videos
  together) go to new videos, saved per run as a histogram of users.
- ID dropout as a dial on the ID + content model, at 0, 0.25, 0.5, 0.75, 0.9 and 1.0.
  At 1.0 the IDs are never used, in training or at serving, so that run is the
  content-only model.
- A popularity correction: train on the score plus the log of each video's share of
  training interactions, rank on the score alone. Run at ID dropout 0 and 0.5.
- Per-category results for new videos with intervals that resample videos.

None of these has results yet. The numbers in `results/` are still from the first run,
which stopped at the 30-epoch cap, and they do not include exposure.

Stopped tracking the old planning file.

## 2026-11-12

A review of the first run found:

- Every run stopped at the 30-epoch cap, so the models were under-trained. The cap is
  now 100 with patience 5.
- The "cold" setting ranks a new video only among other new videos. The realistic
  number is the one that ranks it among all videos.
- The warm and cold numbers were not comparable (different numbers of candidates). A
  matched warm setting was added.
- There were no training-free baselines. Recent popularity, category popularity and
  nearest-content baselines were added.
- English titles were going into Chinese-CLIP's Chinese text tower. An English CLIP
  text tower was added for comparison.
- The LoRA experiment was under-trained.
- Some README numbers traced to files that were not committed.

The first Colab re-run was lost when the free GPU session ended with nothing saved.
The run script now resumes and copies results to Google Drive after every step.
