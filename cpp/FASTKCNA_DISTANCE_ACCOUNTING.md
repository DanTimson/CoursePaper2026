# FastKCNA canonical construction-distance accounting

## Fixed source and checkout state

The audited backend is `https://github.com/xdyangsh/FastKCNA.git` at
`e2f2d79d3de92419e7feea2f1a79d9efc5746f1d`.  Before editing, `HEAD` and
`origin` matched those values.  The tracked checkout was clean; `git status
--porcelain` listed only pre-existing untracked CMake/build products under
`code/` (`CMakeCache.txt`, `CMakeFiles/`, `Makefile`, executables and libraries).
No other revision is supported by this patch.

The shipped builder instantiates
`MatrixOracle<float, kgraph::metric::l2sqr>` in `code/build_index.cpp`.
Consequently one canonical call is one completed full squared-L2 evaluation.
The float specialization dispatches through `metric::l2sqr::apply<float>` to
AVX, SSE2, or the scalar loop according to the build ISA.  The counter is in
the oracle wrapper, after `apply` returns, so all dispatches count once.

## Distance-path audit

| Source | Function / actual metric expression | Classification | Canonical phase | Layer available | Counted? |
|---|---|---|---|---|---|
| `code/src/kgraph.cpp:209-253` | `LinearSearch` via `GenerateControl`; `oracle(i, nn.id)` | diagnostic control/ground-truth generation | none | actual build layer, but deliberately unused | **No.** The oracle is explicitly disabled before every `GenerateControl`. |
| `code/src/kgraph.cpp:370-411` | `KGraphConstructor::init`; `oracle(nn.id, n)` | construction initialization | `knng_candidate` | yes | Yes |
| `code/src/kgraph.cpp:413-433` | `KGraphConstructor::join`; `oracle(i, j)` | NNDescent candidate/refinement construction | `knng_candidate` | yes | Yes |
| `code/src/kgraph.cpp:664-746` | `pg_prune_normal`; `oracle(t_id, p_id)` | forward neighbor pruning | `neighbor_prune` | yes | Yes |
| `code/src/kgraph.cpp:748-993` | all four `pg_prune_fast` branches; `oracle(t_id, p_id)` | incremental forward neighbor pruning | `neighbor_prune` | yes | Yes |
| `code/src/kgraph.cpp:995-1027` | `add_reverse_edges`; `oracle(n, des)` | reverse-edge repair | `reverse_repair` | yes | Yes |
| `code/src/kgraph.cpp:1029-1091` | `reverse_prune`; `oracle(t_id, p_id)` | pruning oversized reverse-augmented lists | `reverse_repair` | yes | Yes |
| `code/src/kgraph.cpp:542-556,1333-1410` | `ComputeEp` / `get_neighbors`; `oracle(query, id)` | entry-point candidate-graph search using center-to-data squared L2 | `construction_search` | yes | Yes |
| `code/src/kgraph.cpp:558-662` | `tree_grow` / `findroot`; `oracle(root, id)` | connectivity construction not fitting the four narrower phases | `other_construction` | yes | Yes (not reached by pg0 or pg2, but covered for the source path) |
| `code/src/kgraph.cpp:1171-1319` | `BridgeView` / `BridgeView_join`; `oracle(id, nsg_id)` | bridge-view construction refinement | `construction_search` | yes | Yes |
| `code/src/kgraph.cpp:1505-1522` | `evaluate` | diagnostic recall over cached `Neighbor.dist`; no metric call | none | n/a | No call exists to count |
| `code/src/kgraph.cpp:1645-1681` | iteration recall/accuracy/delta diagnostics | comparisons over cached distances; no metric call | none | n/a | No call exists to count |
| `code/include/kgraph-data.h` | dataset partial-distance overload `DIST_TYPE::apply(proxy[i]+offset,...)` | live API but no caller in the pinned source | current construction phase if a future construction caller uses it | yes | Wrapper covered; currently zero |
| `code/include/kgraph-data.h` | nested `MatrixOracle::SearchOracle::operator()` | online query API; shipped `KGraphImpl::search` is a stub | post-construction/evaluation | no construction layer | No; it is not instrumented and has no live builder caller |
| `code/include/kgraph-data.h` | `computeDot`, `computeSelfDotPtr` | dot/norm helpers, unused and not full metric distance evaluations | none | n/a | No |
| `code/include/kgraph-data.h` | `Calcenter` | arithmetic center computation, not a distance | none | yes | No; ensuing `get_neighbors` distances are counted |
| `code/build_index.cpp`, `graph_utils.h` | loading, mapping/level generation, `save_pg`, `save_hnsw` | data handling/serialization | none | n/a | No metric calls |

This table covers every `oracle(...)` expression and every `DIST_TYPE::apply`
wrapper reachable from the shipped `build_index`.  Grep was used to enumerate
candidates, then each call context and caller was inspected.

## Exact meaning of upstream diagnostics

* `KGraphConstructor::n_comps` starts at zero for each `index->build` (therefore
  once per actual HNSW layer).  Only `join()` increments it: its OpenMP
  reduction `cc` counts one for each NNDescent join oracle call.  It excludes
  control generation, initialization, pruning, reverse repair, entry-point
  search, tree connectivity, and bridge search.  The patch exposes its exact
  integer sum as `diagnostic_upstream_n_comps` without making it canonical.
* Iteration `cost` is `n_comps / (N * (N - 1) / 2)` in the source.  Its
  denominator is the number of possible **unordered pairs**, not `N^2` and not
  `N(N-1)`.  It counts repeated join evaluations, not unique pairs, and can
  exceed one.  The printed float cannot reconstruct an exact integer reliably.
* `prune scan_rate` is the sum across nodes of each pruning routine's
  `cc / (N * (N - 1) / 2)`.  Each `cc` corresponds to a forward-prune oracle
  call.  It excludes reverse-edge insertion, reverse pruning, entry search, and
  other construction calls; float reductions/printing are inexact.
* `search scan_rate` is the sum across BridgeView join rounds of
  `cc / (N * (N - 1) / 2)`.  It covers bridge-view construction calls only,
  despite the label `search`; it is not online query work and excludes
  `ComputeEp` and every other phase.

None of these ratios is used to infer or subtract a canonical total.

## Counter and phase design

`MatrixOracle` owns cache-line-aligned worker slots.  Every slot has five plain
`uint64_t` phase cells and one `uint64_t` cell for each actual layer.  One
relaxed atomic state encodes disabled or the current `(layer, phase)`.  A metric
call performs one relaxed load and, if enabled, increments only the calling
worker's plain cells after `DIST_TYPE::apply` completes.  There is no globally
contended atomic increment per distance.

FastKCNA uses OpenMP only.  `build_index` calls `omp_set_num_threads(nthreads)`
and configures exactly `nthreads` slots.  Inside an OpenMP region the slot is
`omp_get_thread_num()`; serial/main-thread calls use slot 0.  All metric-bearing
parallel regions are fork/join and no main-thread metric call overlaps OpenMP
worker 0.  Phase/layer changes happen outside active metric-bearing teams.  At
the end, after all layers have joined, `distance_accounting_snapshot()` reduces
worker slots.  `build_index` independently verifies both additive sums before
emitting a success record.

Phase boundaries are source-faithful: initialization and joins are
`knng_candidate`; `ComputeEp` and `BridgeView` are `construction_search`;
forward PG pruning is `neighbor_prune`; reverse insertion and reverse pruning
are `reverse_repair`; tree connectivity is `other_construction`.  The oracle is
disabled immediately before each control generation and is enabled only after
it.  Diagnostic recall uses cached distances and performs no metric calls.

For pg0 the one build is actual layer 0.  For pg2, `build_index` uses the exact
`level` loop variable (highest level down to 0) as the layer key, and counts are
accumulated rather than reset between levels.  `num_perlevel[level]` is the
cumulative number of points assigned at least that level.  Level 0 is explicit;
all positive keys are upper levels.  Upstream doubles `nsg_R` internally only
while building level 0 and restores it afterward; that remains implementation
behavior and is not reported as another input `M`.

## Accepted patch provenance and build

The accepted CX-NND-002 instrumentation was validated against clean FastKCNA
revision `e2f2d79d3de92419e7feea2f1a79d9efc5746f1d`. Its historical patch
SHA-256 is `4146a086d95aa2596f67910cc0e70897c0126298bc9be979e68a0f99ec4f27e6`.

The patch artifact is not retained in the canonical CoursePaper repository.
Reproduction from a pristine FastKCNA checkout requires the separately retained
accepted CX-NND-002 instrumentation matching that SHA-256. Once that
instrumented checkout is present:

```bash
cd /path/to/FastKCNA/code
cmake .
cmake --build . -j2
```

Validation used a detached clean worktree at the pinned commit.  `git apply
--check`, apply, `git diff --check`, and `git apply --check --reverse` all
passed.  Patch SHA-256: `4146a086d95aa2596f67910cc0e70897c0126298bc9be979e68a0f99ec4f27e6`.

The emitted line appears only after successful index serialization and log
finalization:

```text
COURSEPAPER_DISTANCE_COUNTS {"schema":"coursepaper.fastkcna.distance_counts","version":1,"instrumentation":"fastkcna-canonical-distance-v1",...}
```

## Bounded validation evidence

### Direct, exactly enumerable wrapper test

```bash
cd /path/to/CoursePaper2026
g++ -std=c++11 -O2 -fopenmp -I/path/to/FastKCNA/code/include \
  cpp/fastkcna_distance_counter_test.cpp /path/to/FastKCNA/code/libkgraph.a -lrt \
  -o .fastkcna_validation/distance_counter_test
.fastkcna_validation/distance_counter_test
```

Observed:

```text
DIRECT_DISTANCE_COUNTER total=1007 layer0=7 layer1=1000 excluded_disabled=5 status=PASS
```

Seven serial calls and exactly 1000 four-thread calls were retained; five calls
made while disabled were excluded.

### Tiny deterministic data

The committed bounded validator is:

```bash
.venv/bin/python scripts/validate_fastkcna_distance_accounting.py \
  --fastkcna-root /path/to/FastKCNA --multithread 4
```

A deterministic 64-vector, 8-dimensional LSHKIT input was built for pg0 and
pg2 with 1 and 4 threads.  All records parsed through the CoursePaper adapter;
all phase and layer sums equaled total:

| pg | threads | canonical total | layers |
|---:|---:|---:|---|
| 0 | 1 | 2,710 | `0: 2,710` |
| 0 | 4 | 2,758 | `0: 2,758` |
| 2 | 1 | 5,842 | `0: 4,983; 1: 814; 2: 45; 3: 0` |
| 2 | 4 | 5,849 | `0: 4,997; 1: 807; 2: 45; 3: 0` |

Thread-count changes alter upstream RNG streams/scheduling and hence algorithm
work; equality across thread counts is not expected.  The exactly enumerable
four-thread test establishes that worker increments are not lost.

For pg0/one thread, canonical total was 2,710 and exact upstream join-only
`n_comps` was 2,454: difference **256**, exactly `N*S = 64*4` initialization
calls.  Separately, control generation performed `C*(N-1) = 5*63 = 315`
distances and these were excluded by the disabled scope.  The printed final
cost `1.21726` is the rounded diagnostic form of `2454 / (64*63/2)`.

### SIFT10K smoke (no larger job run)

Using the unchanged production parameters and four threads:

* pg0: total 28,995,389; phases `knng_candidate=28,995,389`; layer 0 equals total;
  diagnostic join-only `n_comps=28,875,389` (difference 120,000 = `N*S`).
* pg2: total 74,501,837; phases `knng_candidate=31,497,601`,
  `construction_search=20,821,421`, `neighbor_prune=20,555,655`,
  `reverse_repair=1,627,160`, `other_construction=0`; layers
  `0=71,402,217`, `1=3,095,129`, `2=4,479`, `3=12`, `4=0`.

No SIFT100K or SIFT1M counted run was launched in this task.
