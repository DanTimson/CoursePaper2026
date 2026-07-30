# Total operation analysis

This report treats recorded `*_calc` values as exact finite-instance distance-call counts. The log-linear fits are descriptive/extrapolative models, not implementation-level lower bounds.

## Inputs

- `/home/remotedt/CoursePaper2026/results/bigann100k.jsonl`
- `/home/remotedt/CoursePaper2026/results/bigann10k.jsonl`
- `/home/remotedt/CoursePaper2026/results/bigann10m.jsonl`
- `/home/remotedt/CoursePaper2026/results/bigann1m.jsonl`

## Exact measured inequality

For every partition run, the exact condition is:

`merge_calc < mono_build_calc - partition_build_calc`.

Measured comparisons: **67**; partition+merge wins: **10**.

| dataset | M | efc | algorithm | P | merge params | build saving/pt | merge/pt | net/pt | result |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| bigann10k | 16 | 200 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 316.295 | 359.150 | -42.855 | LOSS |
| bigann10k | 16 | 200 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 316.295 | 292.123 | 24.172 | WIN |
| bigann10k | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 316.295 | 766.939 | -450.643 | LOSS |
| bigann10k | 16 | 200 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 316.295 | 1082.064 | -765.768 | LOSS |
| bigann10k | 16 | 200 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 316.295 | 103.375 | 212.920 | WIN |
| bigann100k | 16 | 32 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 51.782 | 357.996 | -306.215 | LOSS |
| bigann100k | 16 | 32 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 51.782 | 272.491 | -220.710 | LOSS |
| bigann100k | 16 | 32 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 51.782 | 327.911 | -276.129 | LOSS |
| bigann100k | 16 | 32 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 51.782 | 104.608 | -52.827 | LOSS |
| bigann100k | 16 | 64 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 106.637 | 421.884 | -315.247 | LOSS |
| bigann100k | 16 | 64 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 106.637 | 329.698 | -223.061 | LOSS |
| bigann100k | 16 | 64 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 106.637 | 637.104 | -530.467 | LOSS |
| bigann100k | 16 | 64 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 106.637 | 128.152 | -21.515 | LOSS |
| bigann100k | 32 | 64 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 107.509 | 133.744 | -26.235 | LOSS |
| bigann100k | 16 | 200 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 265.380 | 472.767 | -207.388 | LOSS |
| bigann100k | 16 | 200 | CGTM | 2 | merge_lambda=4, jump_ef=20, local_ef=5, next_step_k=3, next_step_ef=6, search_M=3, search_ef=40, merge_ef_construction=-1 | 265.380 | 523.826 | -258.447 | LOSS |
| bigann100k | 16 | 200 | CGTM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 593.974 | -328.594 | LOSS |
| bigann100k | 16 | 200 | IGTM | 2 | merge_lambda=4, jump_ef=20, local_ef=5, next_step_k=3, next_step_ef=3, search_M=3, search_ef=40, merge_ef_construction=-1 | 265.380 | 451.364 | -185.985 | LOSS |
| bigann100k | 16 | 200 | IGTM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 589.439 | -324.059 | LOSS |
| bigann100k | 16 | 200 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 265.380 | 374.243 | -108.863 | LOSS |
| bigann100k | 16 | 200 | IGTM | 2 | merge_lambda=4, jump_ef=64, local_ef=5, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 265.380 | 710.162 | -444.783 | LOSS |
| bigann100k | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=10, merge_ef_construction=-1 | 265.380 | 471.855 | -206.475 | LOSS |
| bigann100k | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=20, merge_ef_construction=-1 | 265.380 | 661.809 | -396.429 | LOSS |
| bigann100k | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 1016.000 | -750.620 | LOSS |
| bigann100k | 16 | 200 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 1530.996 | -1265.616 | LOSS |
| bigann100k | 16 | 200 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=16 | 265.380 | 236.773 | 28.607 | WIN |
| bigann100k | 16 | 200 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=32 | 265.380 | 397.435 | -132.056 | LOSS |
| bigann100k | 16 | 200 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=48 | 265.380 | 547.397 | -282.017 | LOSS |
| bigann100k | 16 | 200 | TWO_MERGE | 2 | merge_lambda=1, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 76.201 | 189.179 | WIN |
| bigann100k | 16 | 200 | TWO_MERGE | 2 | merge_lambda=10, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 275.477 | -10.097 | LOSS |
| bigann100k | 16 | 200 | TWO_MERGE | 2 | merge_lambda=2, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 102.176 | 163.204 | WIN |
| bigann100k | 16 | 200 | TWO_MERGE | 2 | merge_lambda=20, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 476.546 | -211.167 | LOSS |
| bigann100k | 16 | 200 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 265.380 | 147.731 | 117.649 | WIN |
| bigann1m | 16 | 32 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 62.580 | 437.785 | -375.205 | LOSS |
| bigann1m | 16 | 32 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 62.580 | 323.016 | -260.436 | LOSS |
| bigann1m | 16 | 32 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 62.580 | 428.245 | -365.665 | LOSS |
| bigann1m | 16 | 32 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 62.580 | 136.650 | -74.070 | LOSS |
| bigann1m | 16 | 64 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 126.416 | 519.330 | -392.914 | LOSS |
| bigann1m | 16 | 64 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 126.416 | 396.646 | -270.229 | LOSS |
| bigann1m | 16 | 64 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 126.416 | 846.167 | -719.751 | LOSS |
| bigann1m | 16 | 64 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 126.416 | 169.952 | -43.536 | LOSS |
| bigann1m | 16 | 200 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 276.033 | 575.070 | -299.036 | LOSS |
| bigann1m | 16 | 200 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 276.033 | 447.264 | -171.231 | LOSS |
| bigann1m | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=10, merge_ef_construction=-1 | 276.033 | 577.916 | -301.883 | LOSS |
| bigann1m | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=20, merge_ef_construction=-1 | 276.033 | 802.324 | -526.291 | LOSS |
| bigann1m | 16 | 200 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 276.033 | 1990.723 | -1714.690 | LOSS |
| bigann1m | 16 | 200 | TWO_MERGE | 2 | merge_lambda=1, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 276.033 | 101.605 | 174.428 | WIN |
| bigann1m | 16 | 200 | TWO_MERGE | 2 | merge_lambda=10, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 276.033 | 346.704 | -70.670 | LOSS |
| bigann1m | 16 | 200 | TWO_MERGE | 2 | merge_lambda=20, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 276.033 | 569.996 | -293.963 | LOSS |
| bigann1m | 16 | 200 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 276.033 | 193.967 | 82.067 | WIN |
| bigann10m | 16 | 32 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 51.078 | 505.912 | -454.834 | LOSS |
| bigann10m | 16 | 32 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 51.078 | 356.092 | -305.013 | LOSS |
| bigann10m | 16 | 32 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 51.078 | 513.665 | -462.587 | LOSS |
| bigann10m | 16 | 32 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 51.078 | 161.288 | -110.209 | LOSS |
| bigann10m | 16 | 64 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 110.877 | 601.729 | -490.851 | LOSS |
| bigann10m | 16 | 64 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 110.877 | 443.504 | -332.626 | LOSS |
| bigann10m | 16 | 64 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 110.877 | 1027.909 | -917.032 | LOSS |
| bigann10m | 16 | 64 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 110.877 | 203.505 | -92.628 | LOSS |
| bigann10m | 16 | 200 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | 241.997 | 664.635 | -422.639 | LOSS |
| bigann10m | 16 | 200 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 241.997 | 502.385 | -260.388 | LOSS |
| bigann10m | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=10, merge_ef_construction=-1 | 241.997 | 664.841 | -422.844 | LOSS |
| bigann10m | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=20, merge_ef_construction=-1 | 241.997 | 914.450 | -672.454 | LOSS |
| bigann10m | 16 | 200 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 241.997 | 2379.301 | -2137.305 | LOSS |
| bigann10m | 16 | 200 | TWO_MERGE | 2 | merge_lambda=1, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 241.997 | 126.289 | 115.708 | WIN |
| bigann10m | 16 | 200 | TWO_MERGE | 2 | merge_lambda=10, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 241.997 | 400.166 | -158.169 | LOSS |
| bigann10m | 16 | 200 | TWO_MERGE | 2 | merge_lambda=20, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 241.997 | 634.567 | -392.570 | LOSS |
| bigann10m | 16 | 200 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 241.997 | 233.802 | 8.194 | WIN |

## Fitted build models

Model: `build_calc / N = alpha + beta * ln(N)`.

| family | M | efc | alpha | beta | R² | RMSE/pt | N range |
|---|---:|---:|---:|---:|---:|---:|---|
| bigann | 16 | 32 | -305.614 | 79.548 | 0.9983 | 6.257 | 100,000–10,000,000 |
| bigann | 16 | 64 | -738.871 | 166.610 | 0.9989 | 10.439 | 100,000–10,000,000 |
| bigann | 16 | 200 | -1573.069 | 379.765 | 0.9989 | 32.304 | 10,000–10,000,000 |

## Fitted merge crossovers

For equal leaves, the fitted build saving per point is `build_beta * ln(P)`. The reported N* solves `merge_alpha + merge_beta * ln(N*) = build_beta * ln(P)`.

| family | M | efc | algorithm | P | params | merge alpha | merge beta | budget/pt | N* | R² |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| bigann | 16 | 32 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | -9.849 | 32.120 | 55.139 | 8 | 0.9979 |
| bigann | 16 | 32 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 66.399 | 18.154 | 55.139 | 1 | 0.9857 |
| bigann | 16 | 32 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -133.989 | 40.336 | 55.139 | 109 | 0.9979 |
| bigann | 16 | 32 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -35.856 | 12.308 | 55.139 | 1,625 | 0.9943 |
| bigann | 16 | 64 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | -25.218 | 39.053 | 115.485 | 37 | 0.9977 |
| bigann | 16 | 64 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 48.531 | 24.713 | 115.485 | 15 | 0.9897 |
| bigann | 16 | 64 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -335.356 | 84.862 | 115.485 | 203 | 0.9984 |
| bigann | 16 | 64 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -58.856 | 16.363 | 115.485 | 42,395 | 0.9960 |
| bigann | 16 | 200 | CGTM | 2 | merge_lambda=4, jump_ef=15, local_ef=5, next_step_k=3, next_step_ef=6, search_M=5, search_ef=40, merge_ef_construction=-1 | -42.411 | 44.244 | 263.233 | 1,000 | 0.9972 |
| bigann | 16 | 200 | IGTM | 2 | merge_lambda=4, jump_ef=5, local_ef=7, next_step_k=3, next_step_ef=3, search_M=5, search_ef=40, merge_ef_construction=-1 | 16.911 | 30.566 | 263.233 | 3,161 | 0.9925 |
| bigann | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=10, merge_ef_construction=-1 | -7.421 | 41.906 | 263.233 | 638 | 0.9967 |
| bigann | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=20, merge_ef_construction=-1 | 34.937 | 54.860 | 263.233 | 64 | 0.9958 |
| bigann | 16 | 200 | NGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -229.306 | 108.166 | 263.233 | 95 | 1.0000 |
| bigann | 16 | 200 | SIGM | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -647.521 | 188.981 | 263.233 | 124 | 0.9987 |
| bigann | 16 | 200 | TWO_MERGE | 2 | merge_lambda=1, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -48.899 | 10.876 | 263.233 | 2,906,259,078,187 | 0.9999 |
| bigann | 16 | 200 | TWO_MERGE | 2 | merge_lambda=10, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -33.286 | 27.076 | 263.233 | 57,032 | 0.9933 |
| bigann | 16 | 200 | TWO_MERGE | 2 | merge_lambda=20, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | 86.308 | 34.314 | 263.233 | 173 | 0.9890 |
| bigann | 16 | 200 | TWO_MERGE | 2 | merge_lambda=4, jump_ef=40, local_ef=10, next_step_k=6, next_step_ef=6, search_M=40, search_ef=40, merge_ef_construction=-1 | -70.916 | 19.001 | 263.233 | 43,388,612 | 0.9991 |

## Interpretation limits

- A measured WIN/LOSS is exact for that dataset ordering, random graph realization, code revision, and configuration.
- N* is model-dependent and should be reported with the fitted range and residual error; it is not a proof of asymptotic behavior.
- A theorem that merge must eventually lose would additionally need a justified positive lower bound on merge work per point that grows with ln(N). The current implementation's stopping conditions are data-dependent, so the configuration values alone do not provide that bound.
