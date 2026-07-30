# Fast Navigable Graph Construction by Merge

Compare strategies for building an HNSW index by
**merging** partition sub-indices against **full rebuild**, on the distance-
computation axis, across four decades of scale.

## Question

Given two HNSW indices, what is the cheapest way to produce one merged index —
and how does the answer change with dataset size? Full rebuild, sequential
insertion, and several graph-merge algorithms are all *merge strategies* for this
task; the paper compares their **merge cost** in distance computations (language-
and thread-independent), alongside the **search quality** (d_s at matched recall,
and wall-clock QPS) of the index each produces, from 10⁴ to 10⁷ vectors.

The comparison is on the merge operation itself, not on total from-scratch
construction cost: partitioning's build saving is a fixed Θ(N log P) per point
while a merge is Θ(N log N), so no merge asymptotically undercuts a full rebuild
on *total* cost — the interesting axis is the cost and quality of the merge
step, where the strategies differ by more than an order of magnitude.

## Strategies compared

All are treated as **merge strategies** in the sense of Jin et al. (SIGMOD'26):
producing one index from two, by whatever means.

| strategy    | what it does                                              | source |
|-------------|-----------------------------------------------------------|--------|
| **Rebuild** | discard both indices, build one from scratch over all N   | baseline |
| **SIGM**    | seed from one index, insert the other's points (insertion)| baseline |
| **NGM/IGTM/CGTM** | two-way neighbour search between the two graphs      | Ponomarenko, arXiv:2505.16064 |
| **HNSW-Merger** | one-way top-λ forward search + backward direct-connect | Jin et al., SIGMOD'26 |

## Key findings

- **Merge cost, all scales.** On the merge-cost axis, HNSW-Merger is 18–37×
  cheaper than a full rebuild and the traversal merges (IGTM/CGTM/NGM) 2–13×, at
  every scale from 10⁴ to 10⁷. The ordering — HNSW-Merger < IGTM < CGTM ≈ NGM <
  SIGM < Rebuild — is stable across four decades
  (`docs/figures/_scale/merge_strategies_grid.png`,
  `docs/figures/_scale/scale_trend.png`).
- **Reviewer's claim confirmed.** On the merge column the traversal merges beat
  SIGM (sequential insertion, «перестроение» in Ponomarenko's table) at every
  scale — the point the course review turned on.
- **Iso-quality: λ is a clean dial.** HNSW-Merger's λ trades merge cost for search
  quality (d_s at matched recall) monotonically, with diminishing returns past
  λ = 4 — the basis for the default λ = 4
  (`docs/figures/bigann*/iso_quality_r95.png`).
- **Density is edge placement, not edge count.** At identical mean degree (~13.7),
  NGM searches ~15 % cheaper than IGTM/CGTM: its two-way full search places
  better-positioned edges. Merges never fragment connectivity.
  
## Layout

```
ngmbench/              live experiment package (C++ backend driver)
  cli_cpp.py           entry point: expands a sweep config, runs, logs JSONL
  index/hnswmerger.py  shells out to the patched HNSWMerger ./exps / ./builds
  cache.py             JSONL results log + skip-cache
  config.py            dataclasses for run configs
  prepare_bigann.py    extract SIFT prefixes (10k/100k/1M/10M) from BIGANN
cpp/                   patched HNSWMerger sources (see cpp/README.md)
config/                sweep + structural-dump configs
scripts/
  make_figures.py      all paper figures from the JSONL logs
  analyse_trends.py    λ frontier, ef_construction sensitivity, density vs d_s
  graph_structure.py   degree distribution + connectivity from level-0 dumps
  xval_python_ref.py   cross-validate the C++ ports against Ponomarenko's Python
  patch_hnswmerger_gist.py  add a GIST1M workload to an upstream clone
results/bigann*.jsonl  the scale-sweep logs (the paper's data)
results/legacy/        pre-BIGANN SIFT1M / GIST1M runs
docs/figures/          generated figures
tests/                 parser tests for the C++ stdout
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. build the C++ backend (see cpp/README.md), then point HNSWMERGER_BIN at it:
cp .env.example .env      # edit HNSWMERGER_BIN to your HNSW-Merger clone
set -a; . .env; set +a    # export it for this shell
# 2. fetch data
python -m ngmbench.prepare_bigann --src data/bigann --out data/sift_scales \
       --scales 10000 100000 1000000 10000000 --k 100
# 3. run a scale
python -m ngmbench.cli_cpp --config config/bigann100k_sweep.json
# 4. figures + trends
python scripts/make_figures.py --results results/bigann10k.jsonl results/bigann100k.jsonl results/bigann1m.jsonl results/bigann10m.jsonl --out docs/figures
python scripts/analyse_trends.py --results results/bigann10k.jsonl results/bigann100k.jsonl results/bigann1m.jsonl results/bigann10m.jsonl \
       --density docs/figures/structure/graph_structure.csv --out docs/figures/trends
```

## References

- A. Ponomarenko. *Three Algorithms for Merging Hierarchical Navigable Small
  World Graphs.* arXiv:2505.16064.
- C. Jin, Y. Zhang, J. Liu, J. Wang. *Efficient Vector Index Merging in Vector
  Databases.* Proc. ACM Manag. Data 4(1), SIGMOD'26, art. 31.
  <https://doi.org/10.1145/3786645>