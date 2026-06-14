#!/usr/bin/env bash
# Clone the reference HNSW-merge implementation (Ponomarenko) into vendor_repo/.
# This code is imported by ngmbench/vendor_api.py. It is kept out of git because
# the upstream repo has no license file; treat it as all-rights-reserved and
# check with the author before redistributing.
set -euo pipefail
REPO="https://github.com/aponom84/merging-navigable-graphs.git"
DEST="$(dirname "$0")/../vendor_repo"
if [ -d "$DEST/.git" ] || [ -f "$DEST/hnsw.py" ]; then
  echo "vendor_repo already present at $DEST"
else
  git clone --depth 1 "$REPO" "$DEST"
  echo "cloned reference impl into $DEST"
fi
