# PLAN: Cold-start short-video recommendation with Chinese multimodal foundation models

Working repo name: `shortvideo-coldstart`

## Goal

Answer one question with careful experiments:

> **When a new short video has no viewing history, can embeddings from Chinese open-source
> multimodal models (Chinese-CLIP, Qwen-VL) recommend it as well as a model that has seen its
> history, and how much does light fine-tuning (LoRA) help?**

This is the core problem of every short-video feed (Douyin, Kuaishou, WeChat Channels, Bilibili):
thousands of new videos arrive every minute with no clicks yet. Recommendation algorithm engineer
is one of the largest hiring categories at Chinese tech companies, and the multimodal angle
matches supervisors working on multimodal LLMs and on adapting foundation models.

## Data

**MicroLens-100K** (Westlake University, Hangzhou; Ni et al., 2023, arXiv:2309.15379):
100,000 users, 19,738 videos, 719,405 interactions, with cover images and titles (Chinese and English).

- Repo: https://github.com/westlake-repl/MicroLens. Downloads: https://recsys.westlake.edu.cn/
- The dataset forbids redistributing modified copies: download in code, never commit the data.
- Cite the paper in the README.

## Research questions

1. **RQ1 Baseline gap.** How much worse are ID-only sequential recommenders on cold-start items
   than on warm items?
2. **RQ2 Frozen content.** Do frozen image/text embeddings from Chinese-CLIP and Qwen-VL close the
   cold-start gap? Image vs title vs both?
3. **RQ3 Adaptation.** Does LoRA fine-tuning of the encoder (or a small trained projection) help
   beyond frozen features, and at what compute cost?
4. **RQ4 Errors.** Which cold-start videos still fail (by category, title length, popularity)?

## Experimental design

- **Split.** Time-ordered. Hold out the last 10% of items by first-appearance time as **cold-start
  items** (never seen in training). Leave-one-out per user for warm evaluation. State the split
  in the README; it decides everything.
- **Models.**
  - Popularity baseline.
  - SASRec with item IDs (warm baseline; cannot score cold items).
  - SASRec with item embedding = projection of frozen content features (Chinese-CLIP image, Chinese-CLIP text, Qwen-VL fused).
  - SASRec with ID + content (gated sum).
  - Best content model with LoRA on the text/vision encoder (if Colab memory allows; otherwise a 2-layer adapter).
- **Metrics.** Recall@10, NDCG@10 on full ranking (no sampled negatives), reported separately for
  warm and cold items, mean ± std over 3 seeds.
- **Compute budget.** Free Colab T4. Precompute embeddings once (about 20k covers and titles), save
  to `.npy`, train SASRec on cached features.

## Tasks (one commit each)

| # | Task | Acceptance criteria |
|---|---|---|
| T1 | Repo skeleton: `src/`, `scripts/`, `tests/`, `configs/`, MIT licence, `.gitignore` excluding `data/` | `python -m pytest` runs (even with 0 tests) |
| T2 | `scripts/download.py`: fetch MicroLens-100K interactions, titles, covers into `data/raw/`, with SHA-256 manifest | Re-running skips existing files; manifest written |
| T3 | `src/data.py`: build user sequences, time-ordered cold/warm item split, leave-one-out | Unit tests on a 10-user toy file: no cold item in train, split sizes match config |
| T4 | `src/metrics.py`: Recall@K, NDCG@K with full ranking | Tests against hand-computed examples |
| T5 | Popularity + SASRec (ID) in PyTorch, config-driven | Toy test overfits a tiny dataset; Colab run logs warm metrics |
| T6 | `scripts/embed.py`: Chinese-CLIP (image + text) and Qwen-VL embeddings to `data/features/*.npy` | Shape and NaN checks; runtime logged |
| T7 | Content and ID+content SASRec variants | Cold-item metrics reported (ID model shows its cold gap) |
| T8 | LoRA/adapter fine-tuning of the best encoder | Runs on T4 within memory; compared at equal epochs |
| T9 | `scripts/report.py`: results table (mean ± std, 3 seeds), warm vs cold bar chart, error slices | `results/results.md` and PNGs generated from saved runs only |
| T10 | README written as a short paper: question, data, design, results, limitations, how to reproduce | Every number traces to a file in `results/` |
| T11 | Colab notebook running T2 to T9 end to end | Fresh runtime completes |

## Definition of done

- Tests pass; Colab notebook reproduces the results table.
- README results come from `results/`, with seeds and the split described.
- Limitations stated honestly (one dataset, likes as implicit feedback, covers not full video).

## Stretch (only after done)

- Use 4–8 sampled video frames instead of the cover.
- Compare against the MicroLens authors' published baselines on the same split.
- A tiny demo: paste a new video title and cover, see which users it would be shown to.
