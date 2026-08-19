# LayerwiseNNDescentHNSW (`L-NND-HNSW`)

CX-NND-004 implements the literal experimental baseline: assign one HNSW
hierarchy, run the pinned NNDescent/KGraph candidate constructor independently
on every nontrivial layer, and perform one HNSW-compatible conversion.  It is
not presented as a standard industrial constructor and it is not FastHNSW.

## Pinned sources and hierarchy audit

The backend is FastKCNA revision
`e2f2d79d3de92419e7feea2f1a79d9efc5746f1d` after the accepted CX-NND-002
instrumentation patch `fastkcna_canonical_distance_accounting.patch`.  No
CX-NND-004 external change is required.

Stock hnswlib is the vendored v0.8.0 revision
`3f3429661187e4c24a490a0f148fc6bc89042b3d`.  Its
`HierarchicalNSW::getRandomLevel` in `vendor/hnswlib/hnswalg.h` samples

```
U ~ uniform_real_distribution<double>(0,1)
level = int(-log(U) * (1/log(M)))
```

from a `std::default_random_engine` seeded once by the HNSW constructor.
FastKCNA `code/include/graph_utils.h::{getRandomLevel,getLevel}` is the same
expression, distribution, engine type, sequential point order, and
`1/log(M)` multiplier.  The builder calls that existing `getLevel(n, levels,
M, seed)` implementation with `M=16`, `seed=2024` before constructing any
layer.  Therefore there is no material level-semantics disagreement.  The
validation-only `--levels-file` path injects a fixed nonnegative vector and is
rejected by the canonical Python adapter.

`getMapping` orders points by descending exact maximum level and ascending
original label within a level.  Consequently nodes with `maximum_level >= l`
are exactly the internal prefix of length `layer_occupancies[l]`.  The smallest
original label at the maximum level becomes internal entry point zero.  Labels
stored in the index remain original row numbers.

## Independent candidate acquisition

`cpp/layerwise_nnd_hnsw_builder.cpp` creates a fresh `kgraph::KGraph::create()`
object and calls `index->build(...)` exactly once for each layer whose occupancy
is at least two, with:

- `pg_type = INDEX_KNNG` (raw pg0 candidate acquisition);
- `K=500`, `L=500`, `S=12`, `R=100`, `iterations=6`;
- `seed=2024`, `delta=0.002`, `controls=100`, `recall=0.98`;
- one OpenMP thread.

The oracle exposes only the current prefix through `set_size(occupancy)`.  A
new KGraph object, candidate vector, and `IndexInfo` are used for every call.
The pinned implementation also uses legacy `std::rand` in addition to its
seeded engines, so the builder performs `std::srand(2024)` immediately before
each invocation.  Thus no layer inherits another layer's RNG state or graph.
No candidate list is copied between layers and there is no cross-layer
candidate acquisition.

The upstream constructor's own deterministic small-N rules clamp `K`, `L`,
`S`, controls, and search safety fields to `N-1`; when `N <= K`, it sets
`K=S=N-1` and iterations to zero.  The record exposes occupancies, invocation
counts, requested parameters, effective K/L/S/iterations/controls, and actual
iterations.  `massq_S` is absent: it is consumed by FastHNSW `BridgeView`, not
by `INDEX_KNNG`.  Likewise FastHNSW `search_L`, `loop_i`, `alpha`, `tau`, and
bridge/refinement settings are not candidate parameters; only safe values are
assigned to otherwise present upstream fields that participate in small-N
constructor bookkeeping.

## Single conversion pass

For each node, candidate IDs are filtered for range/self, sorted, deduplicated,
and ranked by an actual squared-L2 query-to-candidate evaluation.  The builder
then implements stock `getNeighborsByHeuristic2` semantics: nearest candidates
are considered first and a candidate is rejected when its distance to an
already selected neighbor is strictly smaller than its distance to the query.

The two degree roles follow the pinned stock source exactly.  In
`vendor/hnswlib/hnswalg.h`, `mutuallyConnectNewElement` sets `Mcurmax` to
`maxM0_` (32) at layer zero or `maxM_` (16) above, but its initial outgoing
selection calls `getNeighborsByHeuristic2(top_candidates, M_)` on every layer
(lines 506--515 in the vendored file).  Therefore this builder's initial
directed diversification limit is `M=16` at every layer.  Only after reciprocal
insertion is the final receiving/storage capacity 32 at layer zero and 16
above.  Overflow re-diversification uses that final capacity, matching the
stock reciprocal-overflow call with `Mcurmax` at lines 584--603.  As in stock,
fewer-than-limit candidate sets return without heuristic pair-distance work.

Pinned tie behavior is also reproduced rather than described as ascending ID.
`CompareByFirst` compares only distance, after which
`getNeighborsByHeuristic2` moves `(-distance, internal_id)` into a default
`priority_queue<pair<...>>` (lines 443--460).  Lexicographic pair priority
therefore visits the nearest distance first and the **greatest internal ID
first on an exact distance tie**.  The builder ranks by distance ascending,
internal ID descending.  Its machine record exposes the rule, the per-layer
maximum initial/final degree, and the selected node-0 sample; the equal-distance
tiny test distinguishes descending from ascending ID and initial 16 from final
base capacity 32.

Selected arcs are made reciprocal once.  Nodes overflowing after reciprocal
insertion are visited in ascending node order and diversified once from their
resulting neighbor set; every removed edge is removed at both endpoints.
Removal cannot create another overflow and preserves reciprocity, so no
fixed-point graph-improvement loop is needed.  This is the minimum
degree/reciprocity repair, not NNDescent refinement.

## Why it is not FastHNSW

The new builder requests `INDEX_KNNG`, so `KGraphImpl::build` takes its early
raw-candidate branch.  The other branch in `code/src/kgraph.cpp` contains
`init_nhoods`, repeated `buildPG`, `pg_link`, `ComputeEp`, `tree_grow`, and
`BridgeView`; none is reached.  The builder neither calls FastKCNA
`build_index` as a pg2 black box nor reproduces search/prune/refine loops.
Entry-point selection is the predetermined hierarchy rule, not `ComputeEp`.
Canonical `construction_search` is asserted to equal zero by both C++ and
Python.

## Exact construction accounting

The accepted CX-NND-002 `MatrixOracle` increments after each completed
`DIST_TYPE::apply` in its dataset metric overloads.  It uses one cache-line
counter slot per configured OpenMP worker.  The builder configures all layer
slots once, sets the active layer before each invocation, and reduces only
after construction.  Diagnostic control/GT generation calls
`distance_accounting_disable()` upstream and is excluded.

Phases are:

- `knng_candidate`: KGraph initialization and NNDescent joins;
- `neighbor_prune`: query-to-candidate ranking and HNSW diversification;
- `reverse_repair`: overflow re-ranking/diversification after reciprocal union;
- `construction_search`: invariant zero;
- `other_construction`: invariant zero in the current baseline.

Serialization, SHA calculation, identity checks, validator work, queries, GT,
and recall are outside the construction oracle.  C++ asserts that the phase
sum and layer sum each equal `construction_total`; Python repeats the checks
and maps it to `build_calc`, with `merge_calc=0` and `total_calc=build_calc`.
A new process/oracle starts at zero; the repeated tiny test proves no stale
accumulation.

## Stock serialization and validation

`kgraph::save_hnsw` in the pinned `graph_utils.h` writes the ordinary hnswlib
binary layout directly: header, level-zero link/vector/label blocks, then
upper link lists.  It uses `maxM0=32`, `maxM=16`, the preassigned element
levels, internal entry point zero, reordered source vectors, and original
labels.  It does not use HNSWMerger serialization.

Before writing, the builder checks layer membership, ID ranges, no self or
duplicate edges, degree caps, and reciprocity.  The validation-only
`layerwise_nnd_hnsw_validate` then loads the file with the pinned stock
`HierarchicalNSW`, checks every injected level, label, vector, layer edge,
degree, reciprocal edge, and deterministic entry label.  The unchanged
`fast_hnsw_quality_eval` independently loads the index, validates the label
permutation and source-vector samples, and searches it.

## Quality path

`ngmbench/cli_layerwise_nnd_quality.py` is only an entry point to the generalized
CX-NND-003 stock quality integration.  The C++ query evaluator, counted L2
space, query schema, single-thread search, Recall@10 implementation, and
`ngmbench.quality.ds_at_recall` are unchanged.  The common sweep is
`10,50,100,200,400`; interpolation requires a measured bracket and never
extrapolates.  Layerwise results use the separate
`layerwise-nnd-hnsw-quality` namespace and make no QPS claim.

## Rebuild and patch stack

```
git checkout e2f2d79d3de92419e7feea2f1a79d9efc5746f1d  # FastKCNA
# apply CoursePaper cpp/fastkcna_canonical_distance_accounting.patch (CX-NND-002)
cmake -S /path/to/FastKCNA/code -B /path/to/FastKCNA/code
cmake --build /path/to/FastKCNA/code -j
make -C cpp layerwise FASTKCNA_ROOT=/path/to/FastKCNA
make -C cpp fast_hnsw_quality_eval
```

There is no NND-004 external patch and pg0/pg2 backend behavior is not changed.
