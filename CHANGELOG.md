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

The full run was also reordered so the finding comes first: baselines, IDs only, the ID
dropout sweep and the popularity correction, then the remaining content models. English
CLIP titles, LoRA and Qwen2-VL are extras that run only when asked for. Seeds now train
as separate processes at the same time on the GPU.

## 2026-11-14: full re-run finished; does the tax still hold?

The core run finished on a Colab Pro T4 in about 6 hours 50 minutes (code version
f3260e4, 0 failed steps). `results/` now holds only this run. The first run's
results are gone from the repository and remain in git history.

Every trained model stopped early, with best epochs between 16 and 107 against a cap
of 300. Learning rate 0.003 was chosen for all three families
(`results/lr_search/chosen.md`).

**The tax holds in direction and is much smaller than the first run said.**
Numbers from `results/tradeoff.md`:

- First run: adding IDs to the cover + title model made Recall@10 on old videos 78%
  better and cut new videos' exposure by 62%. That 78% was an artefact of content
  models stopped at 30 epochs.
- This run: 0.0884 to 0.1016 on old videos (15% better), and exposure of new videos
  from 5.52% to 1.12% of top-10 slots (80% lower).
- The curve is not a smooth trade-off. From ID dropout 0 to 0.75, Recall@10 on old
  videos stays between 0.0987 and 0.1026 while exposure rises from 0.00% to 2.57%.
  The cost comes at the end: with no IDs at all, 0.0899 and 5.38%.

**Popularity is not the cause.** The popularity correction lowered Recall@10 on old
videos by 17.5% (dropout 0) and 19.4% (dropout 0.5) and did not raise exposure
(0.00% to 0.00%; 1.68% to 1.38%). Only weakening the IDs moved exposure. One
correction strength (1.0) was tried.

Open points:

- Untested third explanation: old videos are scored from content plus an ID vector and
  new videos from content alone, so new videos may lose on the size of their vectors
  and not on information. The exact 0.00% exposure at dropout 0 points that way.
- Category popularity gives every new video in a category the same score, so its
  10.00% exposure and zero recall come from how ties are broken. It should not be on
  the exposure figure.
- The figure titles were written before the results and need to match them.
- English CLIP titles, LoRA and Qwen2-VL were not run (extras, off by default).
