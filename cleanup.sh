#!/usr/bin/env bash
# Repo cleanup: remove the dead pure-Python implementation (superseded by the C++
# backend + the reviewer's reference driven through scripts/xval_python_ref.py),
# drop root duplicates and stale scratch, and archive pre-BIGANN results.
# Run from the repo root. Review, then `git add -A && git commit`.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "== 7. move live result logs into results/ (relative, prefix dropped) =="
mkdir -p results
for f in bigann10k bigann100k bigann1m bigann10m          struct100k struct_insert100k struct_sigm100k structdens; do
    [ -e "results_${f}.jsonl" ] && git mv "results_${f}.jsonl" "results/${f}.jsonl"
done
 
echo
echo "Done. Now copy in the regenerated files (new __init__.py, .gitignore,"
echo "README.md, cpp/, and the de-narrated scripts), then:"
echo "    git add -A && git commit -m 'Clean up: remove dead Python path, archive legacy results, add cpp/'"
