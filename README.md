# Total-cost merge experiment

This patch turns the direct build-budget sweep into a complete end-to-end
single-thread experiment:

\[
T_A(N,P)=\sum_j B(S_j)+C_A(N,P)
\]

with the build term always taken from the independent `BUILD_ONLY` sweep.

## Files to copy

```text
ngmbench/index/hnswmerger.py
ngmbench/cli_cpp.py
scripts/plot_total_cost_experiment.py
config/total_cost_bigann10k.json
config/total_cost_bigann100k.json
config/total_cost_bigann1m.json
config/total_cost_bigann1m_pilot.json
tests/test_total_cost_experiment.py
```

The two Python package files replace the current repository versions. The rest
are additions.

## What changed in the runner

- Records every pairwise merge step: input sizes, output size, distance count,
  wall time, merge-tree level, and lambda used.
- With top-level `cleanup_merged: true`, deletes intermediate merged indexes
  after they are consumed and deletes the final merged index after its recall
  curve has been recorded. Existing configs without this key keep old behavior.
- Keeps cached leaf indexes.
- Set `NGMBENCH_KEEP_MERGED=1` to retain final merged indexes even when cleanup
  is enabled.
- Supports HNSW-Merger `merge_lambda_mode`:
  - `fixed`
  - `adaptive`, using the log-size interpolation described in Section 7.2 of
    Jin et al.
- Preserves an optional human-readable `label` from the sweep config.

For the current `M=16`, `lambda0=4` balanced P=16 tree, adaptive lambda uses
approximately `4, 7, 10, 13` by merge-tree level. The large-first chain raises
lambda more gradually as the accumulator grows.

## Strategies

The full matrix uses:

- IGTM tuned, balanced
- CGTM tuned, balanced
- NGM `search_ef=10`, balanced
- HNSW-Merger fixed lambda=4, balanced
- HNSW-Merger fixed lambda=4, large-first
- HNSW-Merger adaptive lambda, large-first

`SIGM` is deliberately excluded from the P>2 end-to-end matrix. It does not
merge the graph structures of all input leaves; it retains the first graph and
reinserts the remaining vectors. Charging it for P independently built leaves
would count work that its actual construction path never needs. Keep SIGM as a
separate insertion baseline rather than mixing it into the graph-merge total.

## Run order

From the repository root:

```bash
set -a
. .env
set +a

cp ngmbench/index/hnswmerger.py ngmbench/index/hnswmerger.py.before-total-cost
cp ngmbench/cli_cpp.py ngmbench/cli_cpp.py.before-total-cost

pytest -q tests/test_total_cost_experiment.py
```

Run the inexpensive scales first:

```bash
python -m ngmbench.cli_cpp --config config/total_cost_bigann10k.json
python -m ngmbench.cli_cpp --config config/total_cost_bigann100k.json
```

For 1M, start with the resumable pilot:

```bash
python -m ngmbench.cli_cpp --config config/total_cost_bigann1m_pilot.json
```

Then complete only the missing P=16 traversal-merge rows:

```bash
python -m ngmbench.cli_cpp --config config/total_cost_bigann1m.json
```

Both 1M configs write to the same JSONL. Existing `run_key` rows are skipped.

## Generate figures

Use the direct build-budget CSV already produced by the build-only sweep:

```bash
python scripts/plot_total_cost_experiment.py \
  --build-budget docs/figures/build_budget/build_budget.csv \
  --merge-results \
    results/total_cost_bigann10k.jsonl \
    results/total_cost_bigann100k.jsonl \
    results/total_cost_bigann1m.jsonl \
  --strict-build-check \
  --out docs/figures/total_cost
```

If your build-budget CSV is currently elsewhere, pass its actual path.

## Main outputs

- `p2_components_by_scale`
  - for each strategy: two-leaf build, merge, and total versus 10K/100K/1M,
    normalized by monolithic build.
- `p2_total_ratio_by_scale`
  - all P=2 total curves together.
- `components_by_partitions_<dataset>`
  - build, merge, and total versus P for every strategy.
- `total_ratio_by_partitions`
  - direct break-even chart; horizontal 1.0 is monolithic construction.
- `merge_budget_utilization`
  - \(C_A/[B(N)-L(N,P)]\); below 1 wins.
- `required_merge_speedup`
  - the algorithm-independent speedup over rebuild required merely to fit the
    measured build budget.
- `quality_recall_at_ef100`
  - recall degradation or improvement as repeated merges accumulate.
- `quality_ds_at_recall095`
  - search distance work at matched recall, where the curve reaches 0.95.
- `merge_cost_by_level_p16`
  - where repeated-merge work is spent in the tree/chain.
- `total_cost_summary.csv`
  - all normalized and absolute quantities.
- `merge_steps.csv`
  - every pairwise merge operation.

## Interpretation discipline

The main work comparison is:

```text
directly measured sum of leaf builds
+ directly measured sum of pairwise merges
------------------------------------------------
directly measured monolithic build
```

The merge-run `build_calc` is checked against the independent build-budget
table but is not used as the authoritative source.

The operation-count result and quality result should be reported together.
Fixed lambda=4 is expected to be the cheapest HNSW-Merger variant, but repeated
merges may reduce quality. The adaptive large-first variant is the paper-aligned
multi-index comparison.

## Exploratory FastKCNA backend

The external, non-vendored FastKCNA preparation/runner, tiny compatibility smoke,
and prepared (not executed) overnight commands are documented in
[`cpp/FASTKCNA.md`](cpp/FASTKCNA.md). Uninstrumented upstream counters remain
diagnostic. Canonical construction accounting is documented in
[`cpp/FASTKCNA_DISTANCE_ACCOUNTING.md`](cpp/FASTKCNA_DISTANCE_ACCOUNTING.md).
The separate stock-index Recall@10/exact-search-cost evaluator and prepared
non-construction final commands are documented in
[`cpp/FASTHNSW_QUALITY_EVALUATION.md`](cpp/FASTHNSW_QUALITY_EVALUATION.md).


## CX-NND-004: literal per-layer NNDescent HNSW

`LayerwiseNNDescentHNSW` (`L-NND-HNSW`) is the untuned literal baseline in
[`cpp/LAYERWISE_NND_HNSW.md`](cpp/LAYERWISE_NND_HNSW.md).  Build the C++ tools
against the already instrumented pinned backend first:

```bash
export FASTKCNA_ROOT=/absolute/path/to/FastKCNA
make -C cpp layerwise FASTKCNA_ROOT="$FASTKCNA_ROOT"
make -C cpp fast_hnsw_quality_eval
```

The corrected stock-semantic SIFT10K smoke (`threads=1`) uses initial
`M=16` diversification on every layer, with final storage capacities 32/16.
It produced build total `45,296,135`, phases
`31,437,698 / 0 / 13,828,040 / 30,397 / 0` in the order candidate,
construction-search, prune, reverse-repair, other, layer totals
`41,887,888 / 3,402,217 / 6,024 / 6 / 0`, and occupancies
`10,000 / 617 / 41 / 3 / 1`.  Index SHA-256 is
`206f61574c0126e19ab63cc81edbe11ee3200e6af49c2476d0cd1d4e06f35cf3`.
The Recall@10 / exact `d_s` curve is `(0.89322,250.1038)`,
`(0.99683,691.0581)`, `(0.99963,1114.3036)`,
`(0.99994,1776.8142)`, `(0.99997,2772.8303)` for ef
`10,50,100,200,400`; bracketed `d_s@0.95 = 491.75407655631676`.
The earlier 32-neighbor initial-selection smoke is superseded.

The following large commands are **prepared for a human and were not run in
Prime**.  Each construction command is one foreground process and can be
placed after `prlimit ... --`.  The shell extracts the recorded key only to
pass that exact key back to quality selection; no transient key is product
logic.

### SIFT100K

```bash
export FASTKCNA_ROOT=/absolute/path/to/FastKCNA OMP_NUM_THREADS=1
.venv/bin/python -m ngmbench.cli_layerwise_nnd \
  --config config/layerwise_nnd_hnsw_canonical_sift100k.json --threads 1
BUILD_RESULTS=results/layerwise_nnd_hnsw_canonical_sift100k.jsonl
RUN_KEY="$(.venv/bin/python -c 'import json,sys; r=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; m=[x for x in r if x.get("dataset")=="sift100k" and x.get("canonical") is True]; assert m; print(m[-1]["run_key"])' "$BUILD_RESULTS")"
.venv/bin/python -m ngmbench.cli_layerwise_nnd_quality \
  --config config/layerwise_nnd_hnsw_quality_sift100k.json \
  --construction-results "$BUILD_RESULTS" --construction-run-key "$RUN_KEY"
```

### SIFT1M

```bash
export FASTKCNA_ROOT=/absolute/path/to/FastKCNA OMP_NUM_THREADS=1
.venv/bin/python -m ngmbench.cli_layerwise_nnd \
  --config config/layerwise_nnd_hnsw_canonical_sift1m.json --threads 1
BUILD_RESULTS=results/layerwise_nnd_hnsw_canonical_sift1m.jsonl
RUN_KEY="$(.venv/bin/python -c 'import json,sys; r=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; m=[x for x in r if x.get("dataset")=="sift1m" and x.get("canonical") is True]; assert m; print(m[-1]["run_key"])' "$BUILD_RESULTS")"
.venv/bin/python -m ngmbench.cli_layerwise_nnd_quality \
  --config config/layerwise_nnd_hnsw_quality_sift1m.json \
  --construction-results "$BUILD_RESULTS" --construction-run-key "$RUN_KEY"
```
