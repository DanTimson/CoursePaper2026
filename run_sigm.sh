#!/usr/bin/env bash
# Pure-C++ SIGM run: build first-half leaf, then insert the second half.
# Requires the experiment.cpp patch (distance-calls print). No Python.
set -euo pipefail
ROOT=/home/remotedt/HNSWMerger/HNSW-Merger
CFG="$(cd "$(dirname "$0")" && pwd)"

for DS in sift1m gist1m; do
  echo "=== $DS: build leaf A ==="
  "$ROOT/builds" "$CFG/${DS}_build_leafA.cfg"
  echo "=== $DS: SIGM (insert second half) ==="
  "$ROOT/exps"   "$CFG/${DS}_sigm.cfg" | tee "${DS}_sigm.out"
done

echo; echo "==== SIGM merge cost ===="
grep -HE "distance calls|Total time for insertion" sift1m_sigm.out gist1m_sigm.out
