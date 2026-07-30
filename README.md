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
