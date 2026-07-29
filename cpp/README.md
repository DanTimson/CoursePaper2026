# C++ backend (patched HNSWMerger)

The distance-computation experiments run against the SIGMOD'26 HNSW-Merger code
(Jin et al., *Efficient Vector Index Merging in Vector Databases*, PACMMOD 4(1)
art. 31), open-sourced at <https://github.com/Kimchuls/HNSWMerger>. That upstream
is unlicensed, so it is **not vendored here** — clone it yourself and apply the
sources in this directory.

## What the patches add

Upstream instruments only ES/NGM/IGTM/CGTM for distance counts, and times the
INSERT/REBUILD/TWO_MERGE branches without counting distances. Our build:

- **`experiment.cpp`** (drop-in replacement for the upstream file):
  - counts distance computations in the SIGM/INSERT branch and the TWO_MERGE
    (HNSW-Merger) branch, so every strategy reports a comparable merge cost;
  - parses the five IGTM/CGTM merge knobs (`jump_ef`, `local_ef`, `next_step_k`,
    `next_step_ef`, `search_M`), plus `search_ef`, `merge_ef_construction`, and
    the HNSW-Merger `lambda`, and threads them to the live call sites;
  - prints per-query search distance computations (`d_s`) in the eval loop.
- **`dump_graph_level0.cpp`** (standalone): loads a saved index and emits its
  level-0 adjacency as CSV (`node_id,degree,neighbours`) plus degree/connectivity
  stats, for `scripts/graph_structure.py`.

## Build

    # in your HNSWMerger/HNSW-Merger clone, after copying experiment.cpp in:
    make exp                       # produces ./exps (and ./builds)

    # the graph dumper (flat include layout in this fork):
    g++ -O2 -fopenmp -Wno-write-strings -I. dump_graph_level0.cpp -o dump_graph

Point `config/*.json` `binaries.exps` / `binaries.builds` at the built binaries.