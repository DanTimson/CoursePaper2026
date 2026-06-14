#!/usr/bin/env bash
# Download TEXMEX ANN benchmark corpora into ./data, matching the paths in
# config/sift1m.json. Idempotent: skips an already-extracted dataset and reuses
# an already-downloaded tarball.
#
#   scripts/get_data.sh            # SIFT1M only (~168 MB download, ~0.5 GB extracted)
#   scripts/get_data.sh --gist     # GIST1M only (~2.6 GB download, ~3.6 GB extracted)
#   scripts/get_data.sh --all      # both
#
# Source: http://corpus-texmex.irisa.fr/  (irisa FTP). If your network blocks
# FTP, download the tarballs manually from that page into ./data and re-run --
# the script will skip the download and just extract.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
BASE_URL="ftp://ftp.irisa.fr/local/texmex/corpus"

want_sift=true; want_gist=false
case "${1:-}" in
  --gist) want_sift=false; want_gist=true ;;
  --all)  want_sift=true;  want_gist=true ;;
  --sift|"") ;;
  *) echo "usage: $0 [--sift|--gist|--all]"; exit 2 ;;
esac

fetch() {  # fetch <url> <dest>
  if command -v wget >/dev/null 2>&1; then wget -O "$2" "$1"
  elif command -v curl >/dev/null 2>&1; then curl -fL -o "$2" "$1"
  else echo "need wget or curl"; exit 1; fi
}

get() {  # get <name> e.g. sift / gist
  local name="$1" tar="$DATA/$1.tar.gz" dir="$DATA/$1"
  if [ -f "$dir/${name}_base.fvecs" ]; then
    echo "[$name] already extracted at $dir — skipping"; return
  fi
  mkdir -p "$DATA"
  [ -f "$tar" ] || { echo "[$name] downloading..."; fetch "$BASE_URL/$1.tar.gz" "$tar"; }
  echo "[$name] extracting..."; tar -xzf "$tar" -C "$DATA"
  ls -lh "$dir"/*.fvecs "$dir"/*.ivecs
}

$want_sift && get sift
$want_gist && get gist
echo "done. point config/*.json at data/<name>/<name>_base.fvecs etc."
