#!/usr/bin/env bash
# Repo cleanup: remove the dead pure-Python implementation (superseded by the C++
# backend + the reviewer's reference driven through scripts/xval_python_ref.py),
# drop root duplicates and stale scratch, and archive pre-BIGANN results.
# Run from the repo root. Review, then `git add -A && git commit`.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"


echo "== 6. standardize prepare_bigann (keep package copy, drop script dup) =="
git rm -q scripts/prepare_bigann.py

echo
echo "Done. Now copy in the regenerated files (new __init__.py, .gitignore,"
echo "README.md, cpp/, and the de-narrated scripts), then:"
echo "    git add -A && git commit -m 'Clean up: remove dead Python path, archive legacy results, add cpp/'"