# Exposure of new videos

One top-10 list per warm test user, drawn from all videos, old and new together, with the videos the user already has removed. The table says how many of those slots new videos got.

New videos are 10.0% of the catalogue. A model that spread its slots evenly over videos would give them that share.

| Model | Seeds | Share of top-10 slots | 95% interval | Users shown at least one | Users shown three or more |
|---|---:|---:|---:|---:|---:|
| Popularity | 1 | 0.00% | 0.00% to 0.00% | 0.0% | 0.0% |
| Recent popularity | 1 | 0.00% | 0.00% to 0.00% | 0.0% | 0.0% |
| Nearest covers (no training) | 1 | 11.50% | 11.43% to 11.57% | 69.0% | 10.7% |
| Nearest covers + titles (no training) | 1 | 10.71% | 10.65% to 10.77% | 66.3% | 8.5% |
| IDs only | 3 | 0.00% ± 0.00 | 0.00% to 0.00% | 0.0% | 0.0% |
| Cover only | 3 | 6.19% ± 1.05 | 6.16% to 6.22% | 42.5% | 4.0% |
| Title only | 3 | 8.21% ± 0.75 | 8.18% to 8.24% | 56.4% | 4.9% |
| Cover + title | 3 | 5.52% ± 0.27 | 5.49% to 5.55% | 39.5% | 2.8% |
| IDs + cover + title, ID dropout 0.3 | 3 | 1.12% ± 0.14 | 1.10% to 1.13% | 8.8% | 0.4% |
| IDs + cover + title, ID dropout 0 | 3 | 0.00% ± 0.00 | 0.00% to 0.00% | 0.0% | 0.0% |
| ID dropout 0, popularity correction | 3 | 0.00% ± 0.00 | 0.00% to 0.00% | 0.0% | 0.0% |
| IDs + cover + title, ID dropout 0.25 | 3 | 0.73% ± 0.04 | 0.72% to 0.74% | 5.8% | 0.1% |
| IDs + cover + title, ID dropout 0.5 | 3 | 1.68% ± 0.27 | 1.66% to 1.70% | 13.4% | 0.5% |
| ID dropout 0.5, popularity correction | 3 | 1.38% ± 0.05 | 1.37% to 1.40% | 10.7% | 0.6% |
| IDs + cover + title, ID dropout 0.75 | 3 | 2.57% ± 0.48 | 2.55% to 2.59% | 19.4% | 1.0% |
| IDs + cover + title, ID dropout 0.9 | 3 | 2.77% ± 0.25 | 2.75% to 2.80% | 20.8% | 1.2% |
| IDs + cover + title, ID dropout 1 (no IDs) | 3 | 5.38% ± 0.46 | 5.35% to 5.40% | 38.8% | 2.7% |

Category popularity is left out: it gives every new video in a category the same score, so its exposure depends only on how ties are broken.

Share: mean over seeds (± sample standard deviation). Interval: bootstrap over users, seeds pooled. The per-user columns pool the seeds.
