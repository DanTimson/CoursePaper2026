# Results framing — Layerwise NN-Descent, FastHNSW, and monolithic HNSW

Status: **reviewed writing/figure guidance**
Date: **2026-08-20**

This document records wording that is safe to reuse in the paper/results packet.

## Scaling paragraph

Layerwise construction cost was measured at three dataset sizes on SIFT/BIGANN prefixes (10K, 100K, and 1M vectors) under the frozen canonical configuration. Total dataset-distance evaluations grew from 45.30M to 381.77M to 3.314B: a factor of 8.43x for the first tenfold increase in N and 8.68x for the second. Equivalently, distance evaluations per vector declined from approximately 4,530 to 3,818 to 3,314 over the measured range. These two decade ratios correspond to a finite-range log-log slope of about 0.93; this is reported only as a compact description of the three measured points and is **not** interpreted as an asymptotic complexity estimate.

The decline is already present in the base-layer construction path. From 100K to 1M, L0 construction cost grows by about 8.70x for a tenfold increase in N while base-layer K, L, and the six-iteration limit remain fixed. The observed trend therefore cannot be attributed to upper-layer size/iteration clamping. Its exact mechanism is left open unless it can be demonstrated directly from already-recorded candidate/iteration counters.

The phase mix changes gradually rather than discontinuously across scale: candidate generation remains dominant but its share decreases, neighbour pruning/diversification rises correspondingly, and reverse repair remains below 1% even though its share increases. This supports reporting the observed trend as a coherent finite-range measurement without promoting it to a general complexity law.

## Scaling figure caption

> **Figure N.** Layerwise construction cost versus dataset size on SIFT/BIGANN prefixes (10K, 100K, 1M), single-threaded, frozen canonical configuration. Points show measured total dataset-distance evaluations. No power-law trend line is fitted because three scales do not support an asymptotic-complexity estimate. The observed decade ratios are 8.43x (10K->100K) and 8.68x (100K->1M); per-vector cost declines from approximately 4,530 to 3,314 across the range.

## 1M cross-method comparison

At 1M vectors:

| method | construction dataset-distance evaluations | d_s@0.95 |
|---|---:|---:|
| Monolithic HNSW | 3.723B | 1218.35 |
| Layerwise NN-Descent HNSW | 3.314B | 1389.41 |
| FastHNSW pg2 | 9.838B | 1200.04 |

Recommended headline:

> Under the tested frozen configuration at 1M vectors, Layerwise construction required 3.314B dataset-distance evaluations against 3.723B for monolithic HNSW, or **10.97% fewer dataset-distance evaluations**.

Counter-comparability wording:

> The two construction counters instrument separate backend-specific metric choke-points but measure the same physical unit: one invocation of the underlying squared-L2 dataset metric. HNSWMerger increments immediately before delegating once to its real metric function, whereas the Layerwise/FastKCNA oracle increments after the corresponding metric call completes. For successfully completed runs, these events are in one-to-one correspondence.

Interpretive caveat:

> The equality of the measured unit does not imply equality of total construction work. Layerwise spends its counted distance budget on NN-Descent candidate evaluation, diversification/pruning, and reverse repair, with zero HNSW construction-search evaluations by design. Monolithic HNSW produces its distance evaluations through sequential insertion, including graph traversal and neighbour-selection comparisons. The comparison therefore establishes a difference in dataset-distance evaluations, not a complete CPU/memory cost ordering between the implementations.

The same caveat applies when comparing FastHNSW with either method.

## Layerwise 1M phase decomposition

| phase | distance evaluations |
|---|---:|
| NN-Descent candidate | 2,175,905,937 |
| construction search | 0 |
| neighbour prune/diversification | 1,115,701,223 |
| reverse repair | 22,629,647 |
| other construction | 0 |
| **total** | **3,314,236,807** |

## Bounded open question

The only scaling mechanism worth a small follow-up during results writing is:

> Why does the fixed-six-iteration, fixed-K/L/S base-layer NN-Descent construction path perform fewer metric evaluations per vector as N increases over 10K-1M?

Inspect existing per-iteration/candidate counters only. If duplication, overlap/reuse, or surviving-pair counts make the mechanism directly demonstrable, summarize it briefly. If establishing the mechanism would require new instrumentation or a new run campaign, leave it explicitly open.

## Avoid

Do not write:

- `NN-Descent scales as N^0.93`
- `the measured asymptotic complexity is N^0.93`
- `Layerwise does 10.97% less construction work`
- `small-layer clamping explains the sub-linear trend`
- `both backends use one common instrumentation choke-point`
- `both counters increment after completed metric calls`

Do write:

- `the three measured points correspond to a finite-range slope of about 0.93`
- `Layerwise requires 10.97% fewer dataset-distance evaluations at 1M under the tested frozen configuration`
- `the exact mechanism behind the base-layer per-vector decline remains open unless existing counters demonstrate it`
