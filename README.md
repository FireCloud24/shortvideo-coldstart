# Cold-start short-video recommendation with Chinese multimodal foundation models

> **Status: code complete and unit-tested; the experiments have not been run yet.**
> This README contains no result numbers. They are produced by the Colab notebook and written to
> [`results/results.md`](results/results.md); the Results section below is filled in from that file only.

## 1. Question

A short-video feed receives thousands of new videos a minute. A new video has no views, likes or
watch history, so a recommender that knows items only by their ID has nothing to go on. This
project asks:

> **When a new short video has no viewing history, can embeddings from Chinese open-source
> multimodal models (Chinese-CLIP, Qwen-VL) recommend it as well as a model that has seen its
> history, and how much does light fine-tuning (LoRA) help?**

| | Research question |
|---|---|
| **RQ1** | How much worse are ID-only sequential recommenders on cold-start items than on warm items? |
| **RQ2** | Do frozen image/text embeddings from Chinese-CLIP and Qwen-VL close that gap? Image, title, or both? |
| **RQ3** | Does adapting the encoder (LoRA, or a small adapter) help beyond frozen features, and at what compute cost? |
| **RQ4** | Which cold-start videos still fail (by category, title length, popularity)? |

## 2. Data

**MicroLens-100K** (Ni et al., 2023): 100,000 users, 19,738 short videos, 719,405 interactions,
with a cover image, a title, a category and platform like/view counts per video.

- The data is downloaded by `scripts/download.py` from the authors' server and is **never
  committed**: the authors prohibit redistributing modified copies. `data/` is git-ignored.
- A SHA-256 manifest of what was downloaded is written to `data/raw/manifest.json`.
- The 100K release ships **English titles only** (`MicroLens-100k_title_en.csv`); see Limitations.

## 3. Experimental design

### The split (this decides every number)

1. Order items by the time of their first interaction. The **newest 10% of items (1,974) are
   cold-start items**. Every interaction with them is removed from training, so no model sees a
   cold item's ID or any behaviour on it.
2. The remaining interactions form each user's *warm sequence*. Leave-one-out: the last warm item
   is the warm test target, the one before it the validation target, the rest is training.
3. Each removed interaction (user, cold item, time) is one **cold test example**: the input is the
   user's warm interactions before that time, the target is the cold item.

The split is implemented in [`src/data.py`](src/data.py) and checked by `tests/test_data.py`
(no cold item in training or in any model input; targets never appear in their own input).

### Models

All sequence models share one SASRec trunk (2 layers, 64 dimensions); only the source of the item
vectors changes ([`src/models.py`](src/models.py)).

| Config | Item vector | Can it score a cold item? |
|---|---|---|
| `popularity` | none (training interaction counts) | No: cold items have a count of zero |
| `sasrec_id` | learned ID embedding | No: the cold item's embedding was never trained |
| `content_cnclip_image` / `_text` / `_image_text` | trained projection of **frozen** Chinese-CLIP features | Yes |
| `content_qwen_vl` | trained projection of a frozen Qwen2-VL embedding of cover + title | Yes |
| `id_content_*` | content vector + gate × ID vector (ID dropped for cold items, and randomly in training) | Yes |
| `content_cnclip_text_lora` / `_image_lora` | features re-extracted after LoRA fine-tuning of the encoder | Yes |

### Metrics

Recall@10 and NDCG@10 with **full ranking** (no sampled negatives), items the user already has
removed from the candidates, ties counted against the model ([`src/metrics.py`](src/metrics.py)).
Reported as mean ± sample standard deviation over 3 seeds, in three settings:

- **Warm**: warm test target ranked among all warm items.
- **Cold**: cold target ranked among all cold items ("which new video do we show this user?").
- **Cold vs all items**: cold target ranked among warm and cold items together.

The settings have different numbers of candidates, so models are compared within a setting.
Model selection (early stopping) uses the warm validation set only.

### Adaptation (RQ3)

`scripts/finetune.py` trains LoRA matrices inside one Chinese-CLIP tower with the next-item loss
(in-batch softmax), on warm training sequences only. The adapted encoder then re-embeds every item
and the downstream model is trained with the **same config and number of epochs** as its frozen
counterpart. Time, peak GPU memory and trainable parameters are logged to `results/finetune/`.

## 4. Results

*Not run yet.* After running the notebook this section will quote `results/results.md`
(main table), `results/warm_vs_cold.png` and `results/error_slices.md`, and nothing else.

## 5. Limitations

- **One dataset**, from one platform, already filtered to active users and items.
- **Likes as implicit feedback**: an interaction is a positive signal only; there are no explicit negatives or watch times.
- **Covers, not video**: one cover frame stands in for the whole clip.
- **English titles into a Chinese text encoder**: MicroLens-100K provides machine-translated
  English titles. Chinese-CLIP's text tower was trained on Chinese, so its title results are a
  lower bound on what original Chinese titles would give.
- **Cold items are simulated**: they are the newest items of a finished dataset. All other items'
  full histories are available, unlike a live system.
- **No cold validation set**: hyperparameters and the stopping epoch are chosen on warm data.
- **LoRA is fine-tuned with one seed**; the ± only covers the downstream model's seeds.
- **Small models and budget**: one free Colab T4, a 64-dimensional SASRec, no hyperparameter search.

## 6. Reproduce

Open [`notebooks/colab_end_to_end.ipynb`](notebooks/colab_end_to_end.ipynb) in Colab with a T4 GPU
and run all cells. Or, step by step:

```bash
pip install -r requirements.txt
python -m pytest                                      # 48 tests, about 30 s on CPU
python scripts/download.py                            # ~710 MB into data/raw/
python scripts/embed.py --model cnclip                # GPU
python scripts/embed.py --model qwen                  # GPU
python scripts/train.py --config configs/sasrec_id.yaml --seeds 0 1 2   # repeat per config
python scripts/finetune.py --mode lora --modality text                   # GPU
python scripts/embed.py --model cnclip --what text --lora-dir checkpoints/lora_text --tag lora
python scripts/train.py --config configs/content_cnclip_text_lora.yaml --seeds 0 1 2
python scripts/report.py                              # writes results/results.md and the charts
```

## 7. Repository layout

```
configs/    base.yaml plus one small file per model
scripts/    download.py  embed.py  train.py  finetune.py  report.py
src/        data.py (split)  metrics.py  models.py  train.py  features.py  finetune.py
tests/      unit tests on toy data and a tiny offline Chinese-CLIP
notebooks/  colab_end_to_end.ipynb
results/    written by the scripts: runs/<model>/seed<k>/metrics.json, results.md, charts
```

## References

- Ni, Y., Cheng, Y., Liu, X., Fu, J., Li, Y., He, X., Zhang, Y., Yuan, F. (2023). *A Content-Driven Micro-Video Recommendation Dataset at Scale.* arXiv:2309.15379. <https://github.com/westlake-repl/MicroLens>
- Kang, W.-C., McAuley, J. (2018). *Self-Attentive Sequential Recommendation.* ICDM.
- Yang, A. et al. (2022). *Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese.* arXiv:2211.01335.
- Wang, P. et al. (2024). *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.* arXiv:2409.12191.
- Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.

## Licence

Code: MIT. The MicroLens data is not part of this repository and remains under its authors' terms.
