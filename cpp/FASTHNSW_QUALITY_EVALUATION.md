# FastHNSW stock-index quality evaluation (CX-NND-003)

## Scope and status

`fast_hnsw_quality_eval` reads an already-built FastKCNA `pg_type=2` index with
stock hnswlib.  It does not build a graph, rewrite an index, or involve the
HNSWMerger loader.  The Python runner accepts only a successful, explicitly
selected CX-NND-002 canonical construction record, verifies its index SHA-256,
and writes a separate `fasthnsw-quality` JSONL record.

No external checkout source is changed by CX-NND-003.  The stock reader headers
are vendored, licensed source under `cpp/vendor/hnswlib`; the FastKCNA checkout
remains external.

## Stock format and source evidence

The observed FastKCNA checkout is revision
`e2f2d79d3de92419e7feea2f1a79d9efc5746f1d`.  It does **not** contain or link an
hnswlib checkout.  Instead, its unmodified
`code/include/graph_utils.h::save_hnsw` hand-writes the stock serialization:

1. `offsetLevel0`, maximum/current count, level-0 element size, label/data
   offsets, max level and entry point;
2. `maxM`, `maxM0`, `M`, level multiplier and construction ef;
3. packed level-0 links, float vector and `size_t` label per element;
4. one `uint32` upper-link byte count and its bytes per element.

This order and the x86-64 types are exactly those read/written by stock
`nmslib/hnswlib` v0.8.0 commit
`3f3429661187e4c24a490a0f148fc6bc89042b3d`.  CoursePaper pins that complete
header subtree; `cpp/vendor/hnswlib/UPSTREAM` records the source and the Apache
2.0 license is retained.  Important header SHA-256 values are:

- `hnswalg.h`: `0fb2c1b1d3aae1ea959b536f96844f3f3ae6a2f32ce063f2c31f93d84a7e5c32`;
- `hnswlib.h`: `54403db81f55fd28246114c5c9f683e663a2a55ad64903b56578c73e6db5ad2c`;
- `space_l2.h`: `c599d024657896412250b4fde1a95e39f8bc55cccd263f36c8948fa35779edf1`.

This is a pinned, byte-compatible stock reader; it is not described as a header
that FastKCNA compiled against, because no such header exists in that checkout.
The HNSWMerger fork is deliberately not used: its header inserts distance-block
fields and extended per-node blocks into serialization, which is why it rejects
the stock file.

Build the repository-owned binaries with:

```bash
make -C cpp clean all
make -C cpp test
```

## Cross-evaluator comparability

| Semantic item | existing HNSWMerger evaluator | new stock-hnswlib evaluator |
|---|---|---|
| metric | float squared L2 over all 128 SIFT coordinates | pinned stock `L2Space`, float squared L2 over the same 128 coordinates |
| one counted operation | one completed call through the wrapped dataset metric function | one completed call through `CountingL2Space` to the stock dataset metric function |
| search algorithm entry point | fork `HierarchicalNSW<float>::searchKnn(query, k)` | stock v0.8.0 `HierarchicalNSW<float>::searchKnn(query, k)` |
| k | 10 | 10 |
| ef/search-effort field | `setEf(ef)`; result curve key `ef` | `setEf(ef)`; machine key `ef_search`, normalized curve key `ef` |
| query set | `data/sift_scales/sift_query.fvecs`, 10,000 x 128 | the identical file, with exact row-count/header validation |
| GT set | scale-specific `sift100k_gt.ivecs` or `sift1m_gt.ivecs`, 10,000 x 100; first 10 | the identical scale-specific file, exact shape/range validation, first 10 |
| Recall@10 definition | sum of returned-label occurrences present in the query's GT top-10 set, divided by `nq*10`; rank ignored | exactly the same Python computation; returned duplicates are intentionally not deduplicated and short rows fail |
| search-distance aggregation | counter delta / `nq`; three deterministic repeats are averaged after the C++ mean was printed to four decimals | exact integer total / `nq` for one deterministic serial pass; exact total is retained |
| `d_s@0.95` derivation | measured recall/d_s points sorted by recall, linear interpolation inside a measured bracket | the shared `ngmbench.quality.ds_at_recall`; same rule, no extrapolation or endpoint substitution |

The existing evaluator's three runs are timing repetitions, not three different
query sets or a different cost unit.  Stock search is deterministic here, so a
single pass retains the scientific quantity while avoiding redundant work.  The
new exact mean is not inferred from ef, visited nodes, or time.

During inspection, `scripts/make_figures.py` contained a second later helper that
silently clamped an unbracketed target.  CX-NND-003 removes that override and
uses the already-present strict/no-extrapolation convention, also used by
`scripts/analyse_trends.py`.  Regression tests show unchanged values for all
sampled existing curves that bracket 0.95.  An unbracketed common sweep now
correctly yields `null`/unavailable.

## Counter boundary and threading

`cpp/fast_hnsw_counting_space.h` implements a `SpaceInterface<float>` that owns
stock `L2Space`.  Its trampoline calls the stock distance function first and
increments a `uint64_t` only after that call completes.  The evaluator resets the
counter immediately before its serial query loop and reads it immediately after.
Index load, layout/label/vector checks, result serialization, GT loading and
recall computation are outside that interval.  The canonical path has one query
thread and no OpenMP query loop.  No multithreaded canonical mode is exposed.

The C++ success line is prefixed `COURSEPAPER_FASTHNSW_QUERY ` and uses schema
`coursepaper.fasthnsw.query`, version 1, instrumentation
`stock-hnswlib-v0.8.0-counted-l2-v1`.  It includes the externally verified index
SHA token, ef, k, query count, exact total, exact-derived mean, identity results,
and all returned labels.  A validation/load/search failure exits nonzero and
emits no success record.

## Canonical selection and identity checks

Before launching search, `ngmbench.cli_fasthnsw` requires exactly one explicit
JSONL/run-key match and checks:

- namespace and builder are `fastkcna-canonical`, algorithm is `fasthnsw`,
  `pg_type == 2`, canonical counts are available, and exit status is zero;
- positive `build_calc`, `merge_calc == 0`, `total_calc == build_calc`, additive
  construction phase/layer counts, and the pinned FastKCNA revision;
- the index exists and its freshly computed SHA-256 equals the construction
  record;
- live base path, stat identity, n and dimension equal the construction source;
- base/query/GT fvecs/ivecs byte sizes and every row header; query and GT counts;
- all GT labels are in `[0,n)` and GT top-10 labels are unique.

The C++ reader additionally checks the 64-bit little-endian ABI, loaded element
count, stock offsets and float-vector payload dimension, every external label as
a duplicate-free permutation of `[0,n)`, and byte-compares eight deterministic
stored vectors (including first/last labels) with their source base rows.
Fresh SHA-256 values for base, query and GT are retained in quality provenance.

The stock file has no metric tag.  Thus L2 cannot be proven from index bytes
alone; it is established by FastKCNA's pinned L2 construction source and by this
reader explicitly selecting stock `L2Space`.  Dimension is strongly derived
from stock offsets/vector payload size.  The canonical construction record has
no base-file SHA, so its recorded exact path/size/mtime are checked before a new
base SHA is recorded.  These are the strongest non-mutating checks available.

## Result integration

The enriched record is a deep copy of the canonical construction record, so
`build_calc`, zero merge cost, construction phase/layer decomposition, counter
schema, FastKCNA revision/executable/index provenance and tuning status remain
unchanged.  It receives a distinct quality run key and keeps the old key in
`construction_run_key`.  `recall_curve` uses the main-body keys `ef`, `recall`
and `d_s`, augmented with exact total/query count.  Construction and query
instrumentation remain in separate provenance objects.  The original canonical
JSONL is never modified.  `dataset` retains construction identity (`sift100k` or
`sift1m`), while the explicit `analysis_dataset` alias (`bigann100k`/`bigann1m`)
and `algo=FastHNSW` let the existing recall-vs-d_s figure overlay the curve with
its corresponding main-body rows without relabeling scientific inputs.
CX-NND-003 establishes comparability for Recall@10 versus exact search-distance
cost and `d_s@0.95`, not for query time or QPS.  Its records intentionally omit
`query_seconds` and are excluded from the existing recall-vs-QPS figure.

Illustrative compact shape (numbers abbreviated; this is not a completed large
sweep):

```json
{
  "namespace": "fasthnsw-quality",
  "builder": "fastkcna-canonical",
  "algorithm": "fasthnsw",
  "algo": "FastHNSW",
  "analysis_dataset": "bigann100k",
  "pg_type": 2,
  "run_key": "<quality-key>",
  "construction_run_key": "3981b337d2e4",
  "build_calc": 897000240,
  "merge_calc": 0,
  "total_calc": 897000240,
  "distance_counts_by_phase": {"knng_candidate": 261957732, "construction_search": 254968481, "neighbor_prune": 324815440, "reverse_repair": 55258587, "other_construction": 0},
  "distance_counts_by_layer": {"0": 854149978, "1": 42420430, "2": 427890, "3": 1930, "4": 12},
  "k": 10,
  "nq": 10000,
  "recall_curve": [{"ef": 10, "recall": 0.90, "d_s": 275.0, "query_distance_total": 2750000, "query_count": 10000}],
  "d_s@0.95": null,
  "quality_evaluator": {"instrumentation": "stock-hnswlib-v0.8.0-counted-l2-v1", "query_threads": 1}
}
```

## Bounded validation evidence

### Exact counter and tiny end to end

```text
$ make -C cpp test
fast_hnsw_query_counter_test: PASS k=10 low_ef_count=20 count_a=20 count_b=21
```

The test independently evaluates a known squared-L2 value, checks once-per-call,
additivity and reset, saves/loads a deterministic stock index, compares
Recall@10 search labels to brute-force GT, proves two-query totals are additive,
and proves stored vector/label inspection does not alter the search counter.
Brute-force work uses an independent scalar function, not the wrapper.

A separate deterministic 128-vector/16-dimensional **FastKCNA pg2** index was
then written by the external pinned checkout and loaded directly by the new
reader.  With 16 queries and k=10, ef=10 produced exact total 1,050, mean 65.625,
Recall@10 0.9625; ef=16 produced total 1,321, mean 82.5625, recall 0.9875.
All labels were compared with independently computed brute-force GT.  Both
points passed layout, full label-permutation and 8/8 stored-vector checks; no
serialization conversion occurred and GT work was outside the evaluator.

### SIFT10K real-data smoke

A temporary FastKCNA pg2 index was built solely for the permitted smoke using
the unchanged canonical construction parameters (M/R/iterations/seeds were not
tuned).  The stock evaluator loaded it without conversion and evaluated the
common sweep on the actual SIFT query/GT convention:

| ef | Recall@10 | exact total query distances | d_s |
|---:|---:|---:|---:|
| 10 | 0.89943 | 2,748,743 | 274.8743 |
| 50 | 0.99763 | 7,664,498 | 766.4498 |
| 100 | 0.99983 | 12,282,293 | 1228.2293 |
| 200 | 0.99996 | 19,364,307 | 1936.4307 |
| 400 | 0.99996 | 29,770,785 | 2977.0785 |

Recall is nondecreasing and exact search cost increases at every point.
`d_s@0.95 = 528.0206649185335` by the shared interpolation helper.  GT recall was
computed only after parsing the emitted labels, outside the search process.

### Shared-helper regressions

Existing measured curves remain unchanged at Recall@10=0.95:

- BIGANN100K NGM/search_ef=10: `792.5877314606739`;
- BIGANN1M HNSWMerger/lambda=4: `1512.7745510460245`;
- BIGANN1M NGM/search_ef=10: `1379.4200343511447`.

### Bounded canonical large-index load

Canonical SIFT100K run `3981b337d2e4` was loaded directly with the stock reader
for one query only (`ef=10`, `k=10`).  Count=100,000, complete label permutation,
layout and 8/8 vector samples passed; the query used exactly 467 metric calls.
No SIFT100K or SIFT1M quality sweep was run.

## Prepared final runs (do not execute in Prime)

Build once:

```bash
make -C cpp fast_hnsw_quality_eval
```

Evaluate the already-built canonical SIFT100K pg2 record:

```bash
python -m ngmbench.cli_fasthnsw \
  --config config/fasthnsw_quality_sift100k.json \
  --construction-results results/fastkcna_canonical_sift100k.jsonl \
  --construction-run-key 3981b337d2e4
```

Evaluate the already-built canonical SIFT1M pg2 record:

```bash
python -m ngmbench.cli_fasthnsw \
  --config config/fasthnsw_quality_sift1m.json \
  --construction-results results/fastkcna_canonical_sift1m.jsonl \
  --construction-run-key ee2470d409d0
```

The observed keys are conveniences in these prepared commands/configs, not
product-code constants.  Both commands re-hash the selected existing index and
never invoke FastKCNA construction.

## Comparability conclusion

No semantic mismatch was found that requires `BLOCKED_COMPARABILITY`.  The old
curve stores a four-decimal mean after three timing repeats whereas the new
curve retains an exact integer total and exact-derived mean; both count the same
completed dataset-metric operation per query.  If FastHNSW does not bracket
0.95 on `[10,50,100,200,400]`, its `d_s@0.95` remains unavailable and the common
sweep remains the comparison evidence.


## CX-NND-004 shared stock-index profile

The evaluator and scientific definitions above are also used unchanged for
`LayerwiseNNDescentHNSW`.  Python construction selection now has two strict
profiles rather than a permissive generic record:

- `fasthnsw-quality` requires the accepted canonical FastKCNA pg2 record;
- `layerwise-nnd-hnsw-quality` requires a canonical
  `LayerwiseNNDescentHNSW` record, including zero `construction_search`,
  additive phase/layer totals, the pinned backend revision, and a verified
  output-index SHA.

Both profiles execute this same binary and share query schema
`coursepaper.fasthnsw.query`, exact single-thread L2 counting, Recall@10, and
bracketed `d_s@0.95`.  The schema name is retained to preserve accepted
CX-NND-003 evidence; it does not imply that the Layerwise constructor invokes
FastHNSW.  Layerwise output uses a separate results namespace and, like
FastHNSW, is excluded from QPS figures because the evaluator establishes no
comparable query timing.
