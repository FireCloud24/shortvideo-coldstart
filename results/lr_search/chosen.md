# Learning rate per model family

One representative model per family, seed 0, at each rate in the grid. The rate with the best validation NDCG@10 is used for every model of that family. Test and cold-start results played no part in the choice.

| Family | Learning rate | Validation NDCG@10 | Best epoch | Epochs run | Chosen |
|---|---:|---:|---:|---:|:---:|
| IDs only | 0.001 | 0.0590 | 35 | 40 |  |
| IDs only | 0.003 | 0.0614 | 25 | 30 | yes |
| IDs only | 0.01 | 0.0609 | 34 | 39 |  |
| Content only (cover + title) | 0.001 | 0.0548 | 109 | 114 |  |
| Content only (cover + title) | 0.003 | 0.0581 | 72 | 77 | yes |
| Content only (cover + title) | 0.01 | 0.0520 | 31 | 36 |  |
| IDs + content (cover + title) | 0.001 | 0.0607 | 39 | 44 |  |
| IDs + content (cover + title) | 0.003 | 0.0644 | 50 | 55 | yes |
| IDs + content (cover + title) | 0.01 | 0.0621 | 17 | 22 |  |
