# Current methodology review — NN-Descent / FastHNSW / HNSW comparison

Status: **reviewed checkpoint**
Date: **2026-08-20**

This note records the methodological decisions that should be treated as the current baseline by future review/implementation agents. It consolidates the independent review of the Layerwise NN-Descent, FastHNSW/FastKCNA, and monolithic HNSW comparison and supersedes earlier conversational assumptions where they conflict with this document.

## 1. What is closed

### 1.1 Construction-distance unit is comparable across backends

The headline 1M comparison is:

| method | construction dataset-distance evaluations | matched-recall search cost |
|---|---:|---:|
| Monolithic HNSW | 3,722,580,816 | d_s@0.95 = 1218.3467 |
| Layerwise NN-Descent HNSW | 3,314,236,807 | d_s@0.95 = 1389.4099 |
| FastHNSW pg2 | 9,838,110,472 | d_s@0.95 = 1200.0433 |
| raw FastKCNA pg0 | 2,009,202,367 | not used as a final HNSW quality point |

Layerwise therefore uses **10.97% fewer dataset-distance evaluations** than monolithic HNSW at 1M under the tested frozen configuration.

This statement is intentionally about **dataset-distance evaluations**, not total CPU work or total construction work.

The counters instrument separate backend-specific metric choke-points but count the same physical event:

- HNSWMerger wraps its dataset metric function, increments its atomic counter once, then delegates exactly once to the real metric.
- The FastKCNA/Layerwise oracle increments once after the corresponding `DIST_TYPE::apply` metric evaluation completes.

For successfully completed runs, these events are in one-to-one correspondence. The before/after ordering differs, but the unit is the same: one invocation of the underlying squared-L2 dataset metric.

Do **not** describe the two implementations as sharing one literal instrumentation choke-point.

### 1.2 HNSWMerger counter provenance is source-verified

The local HNSWMerger checkout used for the current work was captured against upstream:

- upstream repository: `https://github.com/Kimchuls/HNSWMerger`
- base/local HEAD: `88f6b7179f651cf5f284bd4e52249a4b1323b5af`
- local HEAD matched `origin/main` at capture time
- the only tracked working-tree differences were `HNSW-Merger/experiment.cpp` and `HNSW-Merger/test_config.h`
- `hnswalg.h`, where the distance counter is implemented, was **not modified locally**

The exact captured diff is retained in:

`cpp/patches/hnswmerger_local_vs_origin_88f6b71.patch`

Capture metadata is retained in:

`cpp/patches/hnswmerger_local_vs_origin_88f6b71_metadata.txt`

SHA-256:

- patch: `761e8e3059f570ae22c800662c85e6de3431b0a44011b78a2539471c6cd36717`
- metadata: `c523226d037d4bf6977e3d7d369074efbefb44e5b5d3eea077310b43bca2bf36`

Loose historical `.patch` files in the external checkout are **not** authoritative provenance; the full captured working-tree diff is.

### 1.3 HNSWMerger licensing note

Earlier notes describing upstream HNSWMerger as unlicensed are stale. The current upstream repository exposes an **Apache-2.0** license.

This removes the former licensing rationale for avoiding vendoring. However, vendoring is **not required** for scientific reproducibility if the repo retains:

1. the upstream URL,
2. the exact base commit,
3. the exact local diff,
4. rebuild instructions,
5. binary hashes for canonical runs where recorded.

Avoid introducing a full vendor copy unless it serves another concrete maintenance goal.

### 1.4 Scaling interpretation is bounded

Canonical Layerwise construction measurements:

| N | total distance evaluations | evaluations/vector |
|---:|---:|---:|
| 10K | 45,296,135 | 4,529.6 |
| 100K | 381,770,489 | 3,817.7 |
| 1M | 3,314,236,807 | 3,314.2 |

Observed decade ratios:

- 10K -> 100K: **8.43x** work for 10x more points
- 100K -> 1M: **8.68x** work for 10x more points

These correspond to a finite-range log-log slope of about **0.93**. This is only a compact description of the three measured points and **must not be interpreted as an asymptotic complexity estimate**.

Do not foreground a claim such as `NN-Descent scales as N^0.93`.

Likewise, the earlier `N^1.14` expectation is retired as a load-bearing argument. It may only be mentioned if a specific source/configuration is recovered and clearly described as a different empirical context.

### 1.5 Small-layer clamping is not the explanation for the sub-linear finite-range trend

The decisive observation is the base layer:

- 100K L0 construction total: about 352.8M
- 1M L0 construction total: about 3.069B
- ratio: about 8.70x for a 10x increase in points

At both scales L0 uses fixed `K=L=500` and six iterations. Therefore the per-point decline is already present where hierarchy-level clamping does not apply.

Do not attribute the trend to upper-layer clamping or iteration clamping.

The bounded open question is:

> Why does the fixed-six-iteration, fixed-K/L/S base-layer NN-Descent construction path perform fewer metric evaluations per input vector as N grows over the measured range?

Candidate explanations (duplication, overlap/reuse, surviving candidate-pair counts, etc.) should only be stated if they fall directly out of existing counters. Otherwise leave the mechanism open.

### 1.6 Cross-method interpretation

The correct headline form is:

> Under the tested frozen configuration at 1M vectors, Layerwise construction requires 10.97% fewer **dataset-distance evaluations** than monolithic HNSW.

Do **not** write:

> Layerwise performs 10.97% less construction work.

The two methods generate the same measured unit through structurally different workloads.

Layerwise 1M phase totals:

- NN-Descent candidate evaluation: 2,175,905,937
- construction search: 0
- neighbour pruning/diversification: 1,115,701,223
- reverse repair: 22,629,647
- other construction: 0

Monolithic HNSW produces its distance evaluations through sequential insertion, including graph traversal and neighbour-selection comparisons. No phase decomposition equivalent to the Layerwise phase table has been established, so avoid unsupported claims about which monolithic subphase is "the bulk".

The same wording discipline applies to FastHNSW comparisons.

## 2. Canonical evidence rules

Future figure/table/report code must select canonical evidence explicitly.

Required discipline:

1. Prefer explicitly named canonical result files/namespaces.
2. Match dataset, scale, algorithm, frozen parameters, and run identity explicitly.
3. Where quality is tied to a built index, use the exact construction run key/index SHA or equivalent provenance field.
4. Never use `min()`, "latest row", "best row", or cheapest matching row as a substitute for canonical identity.
5. Keep exploratory results in explicitly separate namespaces/files.
6. Keep legacy material under `config/legacy/` and `results/legacy/` out of current figures unless a figure explicitly studies legacy behavior.
7. Keep the separately recorded monolithic quality evidence separate from historical construction rows rather than rewriting history.

Known canonical families include:

- `fastkcna_canonical_*`
- `fasthnsw_quality_*`
- `layerwise_nnd_hnsw_canonical_*`
- `layerwise_nnd_hnsw_quality_*`
- `hnsw_monolithic_quality_bigann1m.jsonl`

Existing older HNSWMerger result logs may contain multiple experiment generations; downstream selection must be configuration-specific rather than cheapest-row selection.

## 3. Test/maintenance boundary

The recent NN-Descent/quality work passed the bounded active suite while full pytest collection still encountered the pre-existing `tests/test_operation_model.py` / missing `ngmbench.operation_model` issue.

Current conceptual split:

- measured `BUILD_ONLY` + measured merge total-cost experiments remain current;
- fitted/model-derived `operation_model` material is secondary/derived and was not part of the validated NN-Descent work.

Before calling the repository fully test-clean, do one of the following deliberately:

1. restore and maintain `ngmbench.operation_model` as active functionality, **or**
2. retire/move its stale test and associated obsolete code/material to legacy.

Do not leave default full test collection broken indefinitely.

## 4. What is intentionally still open

These are not blockers for the current result:

- exact mechanism behind the Layerwise L0 per-vector decline;
- total CPU/wall-clock equivalence across different construction backends;
- whether further NN-Descent tuning can move the construction/search-quality frontier.

No new experiment campaign is justified solely to close the first item.

## 5. Current next-step order

1. Apply maintenance arising from this review:
   - explicit canonical selection in figure/report code;
   - preserve HNSWMerger base+diff provenance;
   - make the intended active test suite fully green.
2. Extend deterministic figure/table generation to include:
   - Layerwise,
   - FastHNSW,
   - monolithic HNSW,
   - the matched-recall comparison.
3. Generate a current-results packet from canonical JSONL evidence.
4. During results writing, perform only a bounded existing-counter inspection of the L0 per-point decline.
5. Run new experiments only if figure/review work exposes a genuine unresolved scientific gap.

## 6. Status statement for future agents

The independent methodology review is considered **analytically complete**.

Do not reopen counter-unit equivalence, the retired `N^1.14` expectation, or the small-layer-clamping explanation without new contradictory evidence.

The remaining work is maintenance and presentation, not reconciliation against a target exponent.
