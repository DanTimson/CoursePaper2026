# Fast Navigable Graph Construction by Merge — experiment harness

A pipeline for comparing **divide-and-conquer HNSW construction by graph merge**
(NGM / IGTM / CGTM, from Ponomarenko, *Three Algorithms for Merging HNSW
Graphs*, arXiv:2505.16064) against **sequential insertion (SIGM)** and
**NN-Descent**, on SIFT1M / GIST1M-style data.

The harness builds indices, caches every stage to disk (so reruns and resumes
are cheap), records metrics to a JSONL log, and serves a Streamlit playground
over the results.

This is **phase 1**: pure-Python construction using the reference
implementation. The primary within-merge-family metric is **distance-computation
count** (language-independent). Wall-clock at full SIFT1M/GIST1M scale and the
C++ ports are **phase 2** (see *Extending* below).

## What gets compared, and how

The task is to contrast two ways of building a proximity-graph index:

- **Merge / divide-and-conquer** — partition the data, build a sub-index per
  partition, then merge sub-indices pairwise (NGM/IGTM/CGTM) up a tree. Produces
  a hierarchical, navigable HNSW.
- **NN-Descent** — iterative local refinement (neighbours-of-neighbours).
  Produces a flat approximate k-NN graph, no hierarchy or navigability
  guarantee.

They are not the same object, so the comparison is on a common basis:

| axis | applies to | meaning |
|------|------------|---------|
| `total_calc` (construction distance comps) | merge family + SIGM | shared Python distance fn, counted exactly |
| construction wall-clock | all builders incl. NN-Descent | leaf build + merge seconds |
| `recall@k` under greedy search | all builders | quality of the resulting index |
| graph quality (`deg_*`, `frac_reachable_L0`, `n_layers`) | merge family + SIGM | structural diagnostics |

NN-Descent's distance count is recorded as `null`: `pynndescent` is
Numba-compiled and does not expose a count comparable to the pure-Python merge
family, so for it the honest axis is recall vs wall-clock.

The conceptual contrast (the analytical core of the paper): both methods avoid
naive sequential-insertion cost, but merge does it by *partition-then-stitch*
while NN-Descent does it by *neighbour-of-neighbour refinement*; merge must
preserve hierarchy, an entry point, and bounded degree across the seam, which
NN-Descent has nothing analogous to.

## Layout

```
ngmbench/
  distance.py        # CountingDistance — counts every L2 evaluation
  data.py            # .fvecs/.ivecs IO, synthetic data, partitioning (random|kmeans)
  vendor_api.py      # import shim for the reference impl (vendor_repo/)
  index/
    base.py          # build leaf, id-shift, (de)serialize, graph-quality stats
    merge.py         # divide-and-conquer driver (balanced | sequential order)
    nndescent.py     # NN-Descent baseline via pynndescent
  evaluate.py        # recall@k + search-time distance counting
  config.py          # config dataclasses + stable content hashing
  cache.py           # content-addressed stage cache + JSONL results log
  pipeline.py        # load -> partition -> leaves -> merge -> evaluate -> record
  cli.py             # expand a sweep config into runs
app/playground.py    # Streamlit results browser
config/              # synthetic_quick.json, synthetic_demo.json, sift1m.json
scripts/             # setup_vendor.sh, smoke.py
tests/test_smoke.py  # correctness + determinism checks
vendor_repo/         # reference impl (gitignored; see Setup)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[playground]"        # or: pip install -r requirements.txt
bash scripts/setup_vendor.sh          # clones the reference impl into vendor_repo/
```

`vendor_repo/` holds the upstream reference implementation and is **gitignored**:
the upstream repo ships no license, so it is treated as all-rights-reserved and
re-cloned on demand rather than committed. Ask the author before redistributing,
or switch it to a git submodule.

## Quick start

```bash
python -m ngmbench.cli --config config/synthetic_quick.json   # ~1-2 min
streamlit run app/playground.py -- --results results.jsonl
```

A pre-generated `sample_results.jsonl` is included if you want to open the
playground before running anything:
`streamlit run app/playground.py -- --results sample_results.jsonl`.

`config/synthetic_demo.json` is a fuller grid (3 algos × 2 orders × {2,4,8}
partitions + a kmeans ablation + baselines). It is heavier; it also demonstrates
resume — if interrupted, re-running the same command skips cached runs.

## Running SIFT1M / GIST1M

1. Download the TEXMEX corpus (e.g. `sift.tar.gz`, `gist.tar.gz`) and extract the
   `.fvecs` / `.ivecs` files under `data/`.
2. Point `config/sift1m.json` at them. Start with `base_limit` set (a subset);
   when you truncate the base set the loader drops the shipped groundtruth and
   recomputes exact groundtruth for the subset automatically.
3. Run headless and resumable (long jobs survive disconnects):

```bash
tmux new -s sift
python -m ngmbench.cli --config config/sift1m.json
```

**Scale caveats (phase 1, pure Python):**
- The metric that holds at any scale is `total_calc`. Full-scale *wall-clock*
  needs the C++ ports (phase 2).
- Full **SIFT1M** in pure Python is slow but feasible on a machine with enough
  RAM; remove `base_limit` only when you are ready to wait.
- Full **GIST1M** (960-dim) in pure Python is effectively infeasible on memory
  and time — keep it subset-only until the C++ ports land.
- The GPU is **not used** in phase 1; nothing here is CUDA/ROCm-bound.

## Design notes

- **CGTM id constraint, handled.** The reference `CGTM` classifies a vertex by
  `curr_idx < len(hnsw_a.data)`, which only holds when the left operand's ids
  start at 0. The driver relabels each pairwise merge so B's ids occupy
  `[nA, nA+nB)` (carrying a local→global id map), so any merge order — including
  balanced binary — is correct for all three algorithms.
- **Determinism.** Level assignment (`random.random()`) and CGTM's
  `random.choice` are seeded per build and before each merge, so distance counts
  and recall are reproducible and consistent across cache hits.
- **State-saving.** `leaves` and `index` are separately cached, content-addressed
  stages. A sweep over algorithms on one partitioning builds the leaves once.
  Each bundle stores its own per-phase distance counts and timings, so a cache
  hit returns those numbers without recomputation.

## Extending: adding the C++ ports (phase 2)

Add a builder adapter that returns the same record shape as the others
(`build_calc`/`merge_calc` may be `null` if the port reports only wall-clock),
wire it into `pipeline.run`, and add a `builder` value in the sweep config. The
Elastic C++ port and the Chinese `HNSW-Merger`
(`dl.acm.org/doi/pdf/10.1145/3786645`, repo link inside the PDF) are the targets;
they slot in alongside `nndescent.py` without touching data/partition/eval/cache.

## Phase 2 — HNSWMerger (C++) for full-scale runs

Phase 1 is pure Python (distance counts, subsets). For full SIFT1M/GIST1M
wall-clock, the harness drives the [HNSWMerger](https://github.com/Kimchuls/HNSWMerger)
C++ tool (Apache-2.0) as an upstream dependency — it reimplements NGM/IGTM/CGTM
plus the Elasticsearch approach, rebuild/insert baselines, and its own HNSW-Merger
algorithm in one codebase. We clone and build it; we never fork it (except the
one GIST patch below).

```bash
git clone https://github.com/Kimchuls/HNSWMerger.git
cd HNSWMerger/HNSW-Merger
make build && make exp          # needs g++ + OpenMP; produces ./builds and ./exps
# GIST1M is not a built-in workload — patch + recompile:
python /path/to/ngmbench/scripts/patch_hnswmerger_gist.py test_config.h
make build && make exp
```

Then point `config/sift1m_cpp.json` at the two binaries and your `.fvecs`/`.ivecs`,
and run:

```bash
python -m ngmbench.cli_cpp --config config/sift1m_cpp.json
```

The adapter (`ngmbench/index/hnswmerger.py`) partitions by contiguous id-range
(HNSWMerger's own scheme), builds one leaf index per partition with `./builds`,
then drives divide-and-conquer pairwise merges with `./exps` (since NGM/IGTM/CGTM
are strictly two-index). It parses the C++ stdout into the shared record schema:
`build_calc` and `merge_calc` (HNSWMerger reports `distance calls` on both build
and merge — so distance count is a cross-implementation metric, not Python-only),
`build_seconds`/`merge_seconds`, and a `recall_curve` over the efs sweep. Records
land in the same results log, so the playground shows Python and C++ rows together.

Two honest caveats: the printed recall label `R@100` is hardcoded — the value is
actually recall@`k` (denominator `nq*k`), so it's correct, only mislabeled.
And C++ partitions are contiguous id-ranges, not the Python side's random/k-means
splits; keep those as a Python-side ablation rather than claiming identical
partitions across the two implementations.



## Figures

```bash
python scripts/make_figures.py \
  --results results_cpp.jsonl results_sift.jsonl results_gist_cpp.jsonl results_gist.jsonl \
  --out docs/figures
```

Writes a figure set per dataset to `docs/figures/<dataset>/` (`sift1m/`, `gist1m/`):
`merge_cost`, `partition_scaling`, `recall_vs_qps`, `construction_time`,
`recall_vs_buildtime` (PNG + PDF) and `summary.csv`. Rows are grouped by their
`dataset` field, so SIFT and GIST never mix. On a `run_key` collision the **last**
occurrence wins (a re-run supersedes an earlier row, across files in argument
order), so you don't have to hand-prune old rows — though deleting a superseded
results file is still cleaner. QPS uses the per-dataset query-set size
(SIFT 10000, GIST 1000), read from each row's `nq` when present.

### GIST1M runs

GIST is dim 960 (~7.5x SIFT), so a full {2,4,8} merge sweep is a multi-hour job —
confirm with the advisor whether full-scale wall-clock is needed or a subset is
acceptable before committing to the whole grid.

```bash
# 1. fetch GIST (~2.6 GB)
scripts/get_data.sh --gist

# 2. add the GIST workload to HNSWMerger and recompile (GIST isn't built in)
python scripts/patch_hnswmerger_gist.py /path/to/HNSW-Merger
(cd /path/to/HNSW-Merger && make build && make exp)

# 3. C++ merge family on GIST (edit config/gist1m_cpp.json binary + data paths first)
python -m ngmbench.cli_cpp --config config/gist1m_cpp.json     # -> results_gist_cpp.jsonl

# 4. NN-Descent on GIST (Python)
python -m ngmbench.cli --config config/gist1m.json            # -> results_gist.jsonl

# 5. regenerate all figures
python scripts/make_figures.py \
  --results results_cpp.jsonl results_sift.jsonl results_gist_cpp.jsonl results_gist.jsonl
```

### Data hazards (handled by make_figures.py)

- **`build_calc` / `build_seconds` = 0 on cached-leaf rows.** HNSWMerger reuses
  leaf indexes across algorithms at a partition count, so only the first
  algorithm to run records the build cost. The script imputes the shared build
  per partition count; newer runs also carry it forward via a `.meta.json`
  sidecar next to each leaf, so fresh data won't need imputing.
- **Compare merge algorithms with `merge_calc`, not `total_calc`** — the build
  cost is shared across algorithms at a partition count and would swamp the signal.
- **`TWO_MERGE` reports `merge_calc = 0`** (not routed through the distance
  counter) — dropped from distance plots, kept for time/recall.
- **`INSERT`/baseline recall** is populated via a query-only run on newer adapter
  versions; older rows may be null and are skipped where recall is required.

Reference merge implementation: A. Ponomarenko,
`github.com/aponom84/merging-navigable-graphs` (arXiv:2505.16064). NN-Descent via
`pynndescent`. Datasets: TEXMEX SIFT1M / GIST1M.
