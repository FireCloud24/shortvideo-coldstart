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

## 2026-11-14, later: the vector-length test

Question: are new videos buried because their vectors are shorter than old videos'
(content alone against content plus an ID part), and not because IDs know more?

Test (`scripts/scale_test.py`, numbers in `results/scale_test/scale_test.md`): two
ID + content models were retrained with their weights saved (ID dropout 0 and 0.5,
three seeds, Colab T4, code version c65b7ca), then every new video's vector was
multiplied by a factor with nothing retrained.

- New videos' vectors are shorter: 2.34 against 3.42 (dropout 0) and 4.89 against 6.72
  (dropout 0.5).
- Exposure is very sensitive to that length. Dropout 0.5: 1.68% of top-10 slots as
  trained, 19.80% at a factor of 1.25, 40.38% with mean lengths matched (factor 1.37).
  Dropout 0: 0.00% as trained, 10.64% with lengths matched (factor 1.47).
- With the dropout 0.5 model at a factor of 1.25, Recall@10 for the new video a user
  took, ranked among all videos, goes from 0.0082 to 0.0292, and Recall@10 for old
  videos among all videos goes from 0.1009 to 0.0897.
- The dropout 0 model gains exposure but not accuracy on new videos (0.0045 with
  lengths matched). Its content part was never trained to stand alone.

Reading: most of the burying is a matter of score scale, which neither of the two
planned explanations covers. ID dropout decides whether the exposed new videos are
the right ones.

Limits: the factors were tried on the test users, not chosen on validation. Matching
mean lengths overshoots the catalogue share (40% against 10%), so it is a dial, not a
fix. Recall for old videos here is measured among all videos, which is not the same
column as the main table's (among old videos only). The content-only model was not
measured among all videos, so a like-for-like comparison with it is still missing.

The first Colab attempt at this test failed because the dataset server was
unreachable; a retry worked.

## 2026-11-14, later still: the factor chosen on validation, and a like-for-like table

The two gaps left by the vector-length test are closed
(`results/scale_test/scale_test.md`, section "Like for like"; code version 384d586).

- The factor is now chosen on the validation users: the one at which new videos get
  their share of the catalogue (10%) in those users' top-10 lists. It is then applied
  to the test users. Factors found: 1.42 (ID dropout 0), 1.16 (dropout 0.5), 1.06 (no
  IDs).
- The model with no IDs was retrained with its weights saved and measured in the same
  three columns as the others. Its best epochs (107, 62, 58) match the overnight run.
- Calibrated, the dropout 0.5 model gives new videos 9.91% of slots, finds the new
  video a user took 0.0202 of the time and the old one 0.0953 of the time. The model
  with no IDs, as trained: 5.38%, 0.0126 and 0.0866. Calibrated: 11.28%, 0.0197 and
  0.0836.
- The model with no IDs also gains from calibration although its vectors are almost
  the same length (matching factor 1.06). Vector length does not explain that part.
- The calibrated exposure on test users is 6.92% to 11.28% against the 10% target.
  Validation and test users are the same people one step apart.

The README's two limits about factors tried on test users and the missing comparison
were removed, and these two were added in their place.
