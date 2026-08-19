# External FastKCNA backend (exploratory)

FastKCNA is kept outside this repository, like HNSWMerger.  The published
checkout used here is <https://github.com/xdyangsh/FastKCNA>.  Clone it to any
user-controlled location and configure that location; do **not** copy its source
into CoursePaper2026.

## Locate and build

```bash
git clone https://github.com/xdyangsh/FastKCNA.git ~/FastKCNA
cd ~/FastKCNA/code
cmake .
cmake --build . -j2
# The upstream CMake file does not build the converter:
g++ -O3 -std=c++11 fvec2lshkit.cpp -o fvec2lshkit
```

FastKCNA requires CMake, a C++ compiler, OpenMP, and Boost headers.  With recent
CMake, the unchanged upstream build may require
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.  Configure the external paths in `.env`
(copy `.env.example`) and load them before running:

```bash
export FASTKCNA_ROOT=/path/to/FastKCNA
# Optional explicit overrides:
# export FASTKCNA_BUILD_INDEX=/path/to/build_index
# export FASTKCNA_FVEC2LSHKIT=/path/to/fvec2lshkit
```

The adapter fails clearly if the checkout, either executable, git revision, or
input is unavailable.  It records the resolved checkout/executable paths,
executable SHA-256 values, and exact FastKCNA git commit in every result.

## Repository-owned runner and preparation

`python -m ngmbench.cli_fastkcna --config CONFIG` uses an argv list (no shell),
captures stdout/stderr and exit status, measures wall time, and requires nonempty
FastKCNA log/index outputs.  The prepared SIFT fvecs slices remain the source of
truth.  `fvec2lshkit` conversion goes to
`.fastkcna_work/<dataset>/converted/<dataset>.lshkit`; a source-stat/converter
fingerprint sidecar plus binary header/size validation makes conversion
idempotent.  It never duplicates the original fvecs inside the repository.

Thread count is explicit in each config and may be overridden with
`--threads N` or `NGMBENCH_THREADS=N`.  All upstream CLI parameters and the
non-CLI fixed defaults (`seed=2024`, `delta=0.002`, `massq_S=10`) are recorded.
The four prepared jobs are deliberately **untuned exploratory** runs with
`alpha=60`.  No FastKCNA parameter is asserted to equal hnswlib
`ef_construction=200`.

Prepared jobs (do not launch until reviewed):

```bash
python -m ngmbench.cli_fastkcna --config config/fastkcna_sift100k_pg0.json
python -m ngmbench.cli_fastkcna --config config/fastkcna_sift100k_pg2.json
python -m ngmbench.cli_fastkcna --config config/fastkcna_sift1m_pg0.json
python -m ngmbench.cli_fastkcna --config config/fastkcna_sift1m_pg2.json
```

They write only `results/fastkcna_exploratory_sift100k.jsonl` or
`results/fastkcna_exploratory_sift1m.jsonl`; the runner refuses a JSONL filename
without an explicit `fastkcna` namespace.

## Tiny smoke and FastHNSW compatibility

After loading both backend locations, run only the deterministic 128-vector
smoke:

```bash
python scripts/smoke_fastkcna.py \
  --workdir /tmp/coursepaper-fastkcna-smoke \
  --hnswmerger-exps "$HNSWMERGER_BIN/exps"
```

It builds `pg_type=0` and `pg_type=2`, then attempts to load the FastHNSW output
through the existing patched HNSWMerger `REBUILD, rerun=false` evaluator.  The
compatibility attempt is evidence-producing: a loader rejection is recorded in
`smoke_summary.json` rather than papered over with a conversion.

FastKCNA `pg_type=2` writes its FastHNSW output successfully, but the current
CoursePaper2026/HNSWMerger evaluator rejects that file as corrupted or
unsupported:

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  Index seems to be corrupted or unsupported
```

Therefore the current evaluator format is incompatible with the FastKCNA
output.  No ad-hoc conversion is performed.

CX-NND-003 therefore uses a separate pinned stock-hnswlib reader and leaves both
serializations untouched.  See
[`FASTHNSW_QUALITY_EVALUATION.md`](FASTHNSW_QUALITY_EVALUATION.md) for exact query
metric counting, Recall@10/`d_s` equivalence, bounded validation, and the
prepared evaluation-only commands.

## Accounting boundary

Unmodified FastKCNA remains available only through the explicit
`fastkcna-exploratory` namespace.  Its `cost`, `prune scan_rate`, `search
scan_rate`, and analogous upstream counters remain diagnostic and are never
mapped to `build_calc`.

CX-NND-002 adds a reproducible canonical counter patch for the pinned revision.
See [`FASTKCNA_DISTANCE_ACCOUNTING.md`](FASTKCNA_DISTANCE_ACCOUNTING.md) for the
complete source audit, exact semantics, patch apply/rebuild commands, and
bounded validation.  Counted jobs use only the separate `fastkcna-canonical`
namespace and `results/fastkcna_canonical_*.jsonl` paths.  The adapter requires
and validates the prefixed backend JSON record before mapping its exact total
to `build_calc` and `total_calc` (`merge_calc=0`).  An uninstrumented or malformed
backend is a hard error in that mode; exploratory mode is not promoted even if
the executable happens to be instrumented.

## Monolithic REBUILD thread-invariance preparation

The prepared SIFT100K config uses the existing HNSWMerger `REBUILD` / P=1
monolithic path.  Every matrix point constructs rows 0..100K from an empty HNSW
with M=16 and ef_construction=200; construction threads are exactly 1/2/4/8:

```bash
python -m ngmbench.cli_cpp --config config/insert_thread_invariance_sift100k.json
```

The dispatcher includes construction thread count in the leaf cache key, so the
four points cannot reuse one another's index.  It preserves the backend's
canonical `build_calc` and `build_seconds`, records total runner wall time, and
uses the existing P=1 query-only path for final recall.  HNSWMerger itself is
not modified.
