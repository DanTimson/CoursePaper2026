# HNSWMerger external-backend provenance

Status: **canonical provenance note**
Captured: **2026-08-20**

## Upstream identity

Repository:

`https://github.com/Kimchuls/HNSWMerger`

Captured upstream/local base:

`88f6b7179f651cf5f284bd4e52249a4b1323b5af`

At capture time:

- local `HEAD` = `origin/main`
- branch = `main`
- only two tracked files were modified in the working tree:
  - `HNSW-Merger/experiment.cpp`
  - `HNSW-Merger/test_config.h`

The unmodified upstream `HNSW-Merger/hnswalg.h` is therefore the authoritative distance-counter implementation for this source state.

## Retained exact working-tree diff

Canonical captured diff:

`cpp/patches/hnswmerger_local_vs_origin_88f6b71.patch`

SHA-256:

`761e8e3059f570ae22c800662c85e6de3431b0a44011b78a2539471c6cd36717`

Capture metadata:

`cpp/patches/hnswmerger_local_vs_origin_88f6b71_metadata.txt`

SHA-256:

`c523226d037d4bf6977e3d7d369074efbefb44e5b5d3eea077310b43bca2bf36`

The metadata also records untracked binaries, dumps, CSVs, and historical loose patch files present in the external checkout. Those untracked files are **not** included in the canonical source diff.

## What the local source changes do

### `experiment.cpp`

The captured changes:

- expose merge-algorithm parameter knobs from config;
- load the SIGM/INSERT index with an enlarged `max_elements` capacity;
- optionally set SIGM insertion `ef_construction`;
- print distance-call totals for INSERT and HNSW-Merger merge paths;
- pass explicit NGM/IGTM/CGTM parameter values;
- measure query `d_s` as the distance-counter delta divided by `nq`.

These changes **read** `get_dist_call_counter()` but do not define its increment semantics.

### `test_config.h`

The captured changes:

- add `COHERE1M` and `GIST1M` workload types/defaults;
- add merge parameter knobs:
  - `jump_ef`
  - `local_ef`
  - `next_step_k`
  - `next_step_ef`
  - `search_M`
  - `search_ef`
  - `merge_ef_construction`
- parse those fields from experiment configs.

## Distance-counter semantics

The counter implementation is upstream and unmodified locally.

HNSWMerger's metric wrapper increments the backing atomic counter once and delegates once to the real dataset-distance function. Therefore one counter increment corresponds to one invocation of the underlying dataset metric.

The Layerwise/FastKCNA oracle increments after its corresponding dataset metric evaluation completes. For successful runs, the two events are in one-to-one correspondence.

Methodological wording:

> HNSWMerger and Layerwise instrument separate backend-specific metric choke-points but count the same physical unit: one invocation of the underlying squared-L2 dataset metric. HNSWMerger increments immediately before delegation; the Layerwise/FastKCNA oracle increments immediately after the metric call completes. For successful runs, these counts are directly comparable.

Do not state that both counters increment after completion or that they share one literal choke-point.

## License

The current upstream HNSWMerger repository exposes an Apache-2.0 license. Earlier internal notes calling it unlicensed are stale.

Vendoring is therefore legally possible, but is not required by this experiment provided the project retains the exact upstream commit, exact local diff, rebuild instructions, and relevant binary hashes.

## Reproduction discipline

For a fresh external checkout:

1. clone the upstream repository;
2. checkout `88f6b7179f651cf5f284bd4e52249a4b1323b5af`;
3. apply `cpp/patches/hnswmerger_local_vs_origin_88f6b71.patch` from the repository root containing `HNSW-Merger/`;
4. rebuild the `builds` / `exps` binaries according to the project instructions;
5. compare resulting binary hashes against canonical run provenance where hashes are recorded.

The loose external files `merge_params_and_ds.patch`, `sigm_insert_min.patch`, and `test_config_params.patch` are historical/intermediate artifacts and must not supersede the exact retained full diff.
