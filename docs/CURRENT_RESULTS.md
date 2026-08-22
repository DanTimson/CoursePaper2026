# CoursePaper2026 — Current Results

Status: **current reviewed checkpoint**
Date: **2026-08-21**

This document summarizes the current experimental result of the constructor-comparison track and points to the canonical evidence and generated figures in the repository.

It should be read together with:

- `docs/CURRENT_METHODOLOGY_REVIEW.md`
- `docs/RESULTS_FRAMING_LAYERWISE_FASTHNSW.md`
- `cpp/HNSWMERGER_PROVENANCE.md`
- `docs/figures/constructors/constructor_figures_manifest.json`

## 1. Research question

The current constructor-comparison track asks:

> Can an HNSW graph constructed by applying NN-Descent independently on each preassigned HNSW layer reduce construction distance-evaluation cost relative to conventional monolithic HNSW insertion, and what search-efficiency cost does that construction strategy incur?

A second comparison point is FastHNSW/FastKCNA pg2, which represents a more aggressively refined NN-Descent-derived construction path.

The comparison is intentionally made in **dataset-distance evaluations**, not total CPU work.

## 2. Methods compared

### Monolithic HNSW

Conventional sequential HNSW construction with:

- `M = 16`
- `ef_construction = 200`
- single-threaded construction
- squared-L2 metric

Construction distance counts come from HNSWMerger's metric wrapper. Its counter semantics have been source-verified against the pinned upstream HNSWMerger source and the exact local source diff retained under `cpp/patches/`.

### Layerwise NN-Descent HNSW

Literal per-layer NN-Descent baseline implementing the advisor's suggestion.

Frozen canonical parameters:

- `K = 500`
- `L = 500`
- `S = 12`
- `R = 100`
- iterations = 6
- seed = 2024
- delta = 0.002
- controls = 100
- recall stop = 0.98
- `M = 16`
- threads = 1

Each nontrivial HNSW layer is built independently using the pinned FastKCNA/KGraph backend, followed by a minimal HNSW-compatible diversification/reciprocity conversion.

Construction-search cost is exactly zero by design.

### FastHNSW pg2

Pinned FastKCNA/FastHNSW constructor using the canonical pg2 path and canonical metric-boundary construction accounting.

This is not the same algorithm as the Layerwise baseline.

## 3. Main 1M result

At 1,000,000 vectors:

| Method | Construction dataset-distance evaluations | d_s@0.95 |
|---|---:|---:|
| Monolithic HNSW | 3,722,580,816 | 1218.35 |
| Layerwise NN-Descent HNSW | 3,314,236,807 | 1389.41 |
| FastHNSW pg2 | 9,838,110,472 | 1200.04 |

Relative to monolithic HNSW, Layerwise construction uses:

> **10.97% fewer dataset-distance evaluations**

but requires:

> **14.04% more query distance evaluations at 95% Recall@10**

under the measured search-quality comparison.

FastHNSW pg2 requires substantially more construction distance evaluations than either method but recovers search efficiency close to monolithic HNSW.

The correct interpretation is therefore a construction/search-efficiency trade-off, not a universal ordering of total implementation cost.

## 4. Counter comparability

The Layerwise and HNSWMerger counters instrument different backend-specific metric choke-points but measure the same physical unit:

> one invocation of the underlying squared-L2 dataset metric.

HNSWMerger increments immediately before delegating once to its real metric function. The Layerwise/FastKCNA oracle increments after the corresponding metric evaluation completes.

For successfully completed runs these events are in one-to-one correspondence, so the construction distance totals are directly comparable.

This does **not** imply equality of total CPU, memory, allocation, or synchronization cost between implementations.

## 5. Layerwise scaling over the measured range

Canonical Layerwise construction measurements:

| N | Total distance evaluations | Evaluations/vector |
|---:|---:|---:|
| 10K | 45,296,135 | 4,529.6 |
| 100K | 381,770,489 | 3,817.7 |
| 1M | 3,314,236,807 | 3,314.2 |

Observed decade ratios:

- 10K → 100K: **8.43x** work for 10x more points
- 100K → 1M: **8.68x** work for 10x more points

These three points correspond to a finite-range log-log slope of approximately **0.93**.

This is reported only as a description of the measured range. It is **not** interpreted as an asymptotic complexity estimate.

The earlier `N^1.14` expectation is not used as a target that these measurements must reconcile with.

## 6. Why small-layer clamping does not explain the trend

The same per-vector decline is already present in the base layer.

From 100K to 1M:

- L0 cost grows from about 352.8M to 3.069B
- this is about **8.70x** work for **10x** more vectors
- base-layer `K = L = 500` at both scales
- six iterations are used at both scales

Therefore the finite-range sub-linear trend cannot be attributed to small upper-layer clamping.

The exact mechanism behind the lower metric-evaluation rate per vector remains intentionally open unless it can be demonstrated directly from already-recorded counters.

## 7. Layerwise 1M construction decomposition

| Phase | Distance evaluations |
|---|---:|
| NN-Descent candidate evaluation | 2,175,905,937 |
| Construction search | 0 |
| Neighbour prune/diversification | 1,115,701,223 |
| Reverse repair | 22,629,647 |
| Other construction | 0 |
| **Total** | **3,314,236,807** |

The phase mix changes gradually with scale rather than showing a discontinuity.

Candidate generation remains dominant, neighbour pruning/diversification grows somewhat in share, and reverse repair remains below 1% of the total even at 1M.

## 8. Generated canonical figures

The deterministic constructor-results generator writes the current figure set to:

`docs/figures/constructors/`

### Construction scaling

- `constructor_build_scaling.png`
- `constructor_build_scaling.pdf`

Shows measured construction distance evaluations across scale.

No power-law trend line is fitted.

### Construction cost per vector

- `constructor_build_per_vector.png`
- `constructor_build_per_vector.pdf`

Shows the finite-range decline in distance evaluations per vector.

### Layerwise phase decomposition

- `layerwise_phase_per_vector.png`
- `layerwise_phase_per_vector.pdf`

Shows the contribution of candidate generation, pruning/diversification, and repair across scale.

### 1M construction/search trade-off

- `constructor_tradeoff_1m.png`
- `constructor_tradeoff_1m.pdf`

Shows construction distance cost against matched-recall query distance cost.

### 1M Recall@10 versus query distance work

- `recall_vs_ds_1m.png`
- `recall_vs_ds_1m.pdf`

Shows the measured recall/distance curves underlying `d_s@0.95`.

## 9. Generated canonical tables

The deterministic generator writes:

- `constructor_summary.csv`
- `layerwise_scaling.csv`
- `quality_curves_1m.csv`
- `constructor_figures_manifest.json`

The manifest records:

- exact selected constructor evidence
- construction run keys
- exact Layerwise/FastHNSW quality linkage
- monolithic historical-quality source linkage
- input SHA-256 hashes
- generated derived values

### Monolithic 1M linkage detail

The monolithic scaling table uses the direct canonical `BUILD_ONLY` row, while the separately recorded monolithic quality result retains its historical source run.

These are matched by:

- frozen construction configuration
- identical construction count: `3,722,580,816`

This is **configuration + construction-count equivalence**, not literal identity of the two run keys.

Layerwise and FastHNSW quality use exact construction-run linkage.

## 10. Canonical-selection discipline

Current constructor figures must not select a cheapest, latest, or otherwise opportunistic row.

Canonical evidence is selected by explicit:

- result family / namespace
- dataset and scale
- frozen parameters
- construction run identity where applicable
- quality-to-construction linkage
- construction count / provenance consistency checks

Ambiguous evidence is an error.

The older merge-figure helpers have also been hardened to prevent adaptive or alternate traversal rows from silently masquerading as canonical results.

## 11. Test state

At this checkpoint:

> **53 tests pass**

The stale `test_operation_model.py` path has been retired from the active suite rather than resurrecting a deleted fitted-model subsystem that the current methodology does not rely on.

Measured `BUILD_ONLY` and measured total-cost experiments remain current.

## 12. Current interpretation

The strongest current result is:

> Literal per-layer NN-Descent HNSW construction is viable and, at 1M vectors under the tested frozen configuration, uses modestly fewer dataset-distance evaluations than conventional monolithic HNSW construction. That construction saving is accompanied by a measurable loss in search efficiency at matched recall. FastHNSW pg2 spends substantially more construction distance work but recovers search efficiency close to monolithic HNSW.

This supports treating NN-Descent-based HNSW construction as a **construction/search-quality trade-off space**, rather than as a method with one universally superior cost.

## 13. Claims not supported by the current evidence

Do not claim that:

- NN-Descent asymptotically scales as `N^0.93`
- the Layerwise method performs 10.97% less total construction work
- small-layer clamping causes the finite-range sub-linear trend
- the two backends share one literal instrumentation choke-point
- FastHNSW is globally worse because it uses more construction distance evaluations
- the present three-point scale study establishes asymptotic behavior

## 14. Bounded open question

One mechanism remains worth a small existing-data check during final results writing:

> Why does the fixed-six-iteration, fixed-K/L/S base-layer NN-Descent construction path perform fewer metric evaluations per input vector over the measured 10K–1M range?

Only inspect existing per-iteration/candidate counters.

If the mechanism does not fall directly out of existing evidence, leave it open rather than launching a new experiment campaign solely to explain the slope.

## 15. Current next step

The constructor-comparison implementation, methodology review, canonical-selection maintenance, active test suite, and deterministic figure generation are now in a stable state.

The next work should be:

1. assemble the supervisor-facing/current-results summary from this packet;
2. decide whether the constructor comparison is already sufficient as the experimental core;
3. run new experiments only if the review or supervisor feedback identifies a concrete missing comparison.
