# Trade-off: accuracy on old videos against exposure of new videos

Warm Recall@10: the held-out old video is in the top 10, ranked among all old videos. Exposure: share of top-10 slots that go to new videos when old and new are ranked together. Mean ± sample standard deviation over seeds.

## ID dropout sweep (IDs + cover + title)

| Model | ID dropout | Seeds | Warm Recall@10 | New-video exposure (%) |
|---|---:|---:|---:|---:|
| IDs + cover + title | 0 | 3 | 0.0987 ± 0.0005 | 0.00 ± 0.00 |
| IDs + cover + title | 0.25 | 3 | 0.1008 ± 0.0011 | 0.73 ± 0.04 |
| IDs + cover + title | 0.3 | 3 | 0.1016 ± 0.0014 | 1.12 ± 0.14 |
| IDs + cover + title | 0.5 | 3 | 0.1026 ± 0.0017 | 1.68 ± 0.27 |
| IDs + cover + title | 0.75 | 3 | 0.1016 ± 0.0016 | 2.57 ± 0.48 |
| IDs + cover + title | 0.9 | 3 | 0.0962 ± 0.0002 | 2.77 ± 0.25 |
| IDs + cover + title | 1.0 | 3 | 0.0899 ± 0.0035 | 5.38 ± 0.46 |

Comparisons along the sweep:

- Adding IDs (dropout 0.3) to the model with no IDs (dropout 1): Recall@10 on old videos +13.0% (0.0899 to 0.1016), exposure -79.3% (5.38% to 1.12%).
- Adding IDs (dropout 0) to the model with no IDs (dropout 1): Recall@10 on old videos +9.9% (0.0899 to 0.0987), exposure -100.0% (5.38% to 0.00%).
- Removing IDs entirely from the best model on old videos (dropout 0.5): Recall@10 on old videos -12.4% (0.1026 to 0.0899), exposure +219.8% (1.68% to 5.38%).
- From dropout 0 to 0.75, Recall@10 on old videos stays between 0.0987 and 0.1026 while exposure goes from 0.00% to 2.57%.

## Popularity correction

Trained with the log of each video's share of training interactions added to its score, and scored without it. Change is against the run with the same ID dropout and no correction.

| ID dropout | Correction strength | Seeds | Warm Recall@10 | Change | New-video exposure (%) | Change (points) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0 | 3 | 0.0815 ± 0.0010 | -17.5% | 0.00 ± 0.00 | +0.00 |
| 0.5 | 1.0 | 3 | 0.0827 ± 0.0013 | -19.4% | 1.38 ± 0.05 | -0.30 |

## Other models

| Model | ID dropout | Seeds | Warm Recall@10 | New-video exposure (%) |
|---|---:|---:|---:|---:|
| Popularity |  | 1 | 0.0025 | 0.00 |
| Recent popularity |  | 1 | 0.0104 | 0.00 |
| Nearest covers (no training) |  | 1 | 0.0202 | 11.50 |
| Nearest covers + titles (no training) |  | 1 | 0.0254 | 10.71 |
| IDs only |  | 3 | 0.0924 ± 0.0010 | 0.00 ± 0.00 |
| Cover only |  | 3 | 0.0796 ± 0.0008 | 6.19 ± 1.05 |
| Title only |  | 3 | 0.0640 ± 0.0024 | 8.21 ± 0.75 |
| Cover + title |  | 3 | 0.0884 ± 0.0013 | 5.52 ± 0.27 |
