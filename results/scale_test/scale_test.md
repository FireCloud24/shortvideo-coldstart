# Are new videos buried by the length of their vectors?

Each model is evaluated as trained (factor 1) and again with every new video's vector multiplied by a factor. Nothing is retrained. "matched" makes the mean length of a new video's vector equal the mean length of an old video's. Exposure: share of top-10 slots going to new videos in a mixed feed. The two Recall@10 columns rank the target among all videos, old and new. Mean ± sample standard deviation over seeds.

## IDs + cover + title, ID dropout 0

Seeds: 3. Mean vector length: old videos 3.420 ± 0.144, new videos 2.335 ± 0.136. Matching factor: 1.47 ± 0.03.

| Factor on new videos | New-video exposure (%) | Recall@10, new video among all | Recall@10, old video among all |
|---|---:|---:|---:|
| 1 | 0.00 ± 0.00 | 0.0000 ± 0.0000 | 0.0987 ± 0.0005 |
| 1.25 | 0.72 ± 0.08 | 0.0002 ± 0.0001 | 0.0985 ± 0.0005 |
| 1.5 | 13.52 ± 2.62 | 0.0057 ± 0.0019 | 0.0931 ± 0.0017 |
| 2 | 57.42 ± 10.70 | 0.0299 ± 0.0099 | 0.0658 ± 0.0082 |
| 3 | 90.69 ± 5.37 | 0.0578 ± 0.0076 | 0.0312 ± 0.0087 |
| matched | 10.64 ± 2.36 | 0.0045 ± 0.0023 | 0.0942 ± 0.0019 |

## IDs + cover + title, ID dropout 0.5

Seeds: 3. Mean vector length: old videos 6.723 ± 0.693, new videos 4.888 ± 0.355. Matching factor: 1.37 ± 0.05.

| Factor on new videos | New-video exposure (%) | Recall@10, new video among all | Recall@10, old video among all |
|---|---:|---:|---:|
| 1 | 1.68 ± 0.27 | 0.0082 ± 0.0018 | 0.1009 ± 0.0013 |
| 1.25 | 19.80 ± 1.03 | 0.0292 ± 0.0025 | 0.0897 ± 0.0016 |
| 1.5 | 62.93 ± 3.68 | 0.0564 ± 0.0050 | 0.0638 ± 0.0041 |
| 2 | 96.03 ± 0.69 | 0.0784 ± 0.0054 | 0.0280 ± 0.0029 |
| 3 | 99.25 ± 0.52 | 0.0814 ± 0.0045 | 0.0101 ± 0.0054 |
| matched | 40.38 ± 8.73 | 0.0424 ± 0.0050 | 0.0780 ± 0.0034 |
