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

## 2026-11-13, later: second Colab run stopped

The re-run started on a Colab T4 and was stopped after about two hours. Colab then
ended the GPU session for usage limits.

- The IDs-only model stopped early at epochs 42 to 49, as intended.
- The cover-only and title-only models reached the 100-epoch cap with their best
  epoch at 99 or 100. They were still improving, so 100 epochs at a learning rate of
  0.001 is not enough for them. Those runs are discarded.
- Timing on the T4: 8.1 s for a training epoch, 1.2 s for a validation pass. My first
  guess, that validation was the slow part, was wrong.

Changes made because of this:

- The epoch cap is 300.
- The learning rate is chosen per model family (IDs only, content only, IDs + content)
  from 0.001, 0.003 and 0.01, by validation NDCG@10 of one representative model with
  seed 0. One shared rate suited the ID model and starved the content models.
- Training uses mixed precision on a GPU. The speed-up has not been measured yet.
- The full run can be given a time limit and continues in a later session.
- A Kaggle notebook was added, because the run does not fit in free Colab sessions.
