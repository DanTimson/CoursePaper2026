#!/usr/bin/env python3
"""Prepare non-SIFT benchmarks (Deep, Turing, GIST, Cohere) into the fvecs/ivecs
layout the harness expects, mirroring prepare_bigann.py for SIFT.

Each dataset is sliced to 10k/100k/1m/10m prefixes of its base, the query set is
written once, and ground truth is computed per slice by exact L2 search (the
source GT covers only the full base, so prefixes need their own). All four are
L2; the harness reads fvecs, so every source format is widened to float32.

Source formats
    Deep, Turing : .fbin   (big-ann-benchmarks: int32 npts, int32 dim, then
                            float32 data) -- github.com/harsha-simhadri/big-ann-benchmarks
    GIST         : .fvecs  (corpus-texmex; already float32, just sliced)
    Cohere       : .parquet (HF Cohere/wikipedia-22-12; an 'emb' column of
                            768-d float lists) -> extracted to fvecs first

Usage
    # Deep / Turing (fbin base + query in one dir)
    python scripts/prepare_dataset.py --dataset deep \
        --base data/deep/base.1B.fbin --query data/deep/query.public.10K.fbin \
        --out data/deep_scales --scales 10000 100000 1000000 10000000

    # GIST (fvecs)
    python scripts/prepare_dataset.py --dataset gist \
        --base data/gist/gist_base.fvecs --query data/gist/gist_query.fvecs \
        --out data/gist_scales --scales 10000 100000 1000000

    # Cohere (parquet shards -> pass the directory; emb column extracted)
    python scripts/prepare_dataset.py --dataset cohere \
        --base data/cohere/parquet --query data/cohere/parquet \
        --out data/cohere_scales --scales 10000 100000 1000000 --nq 1000

Writes, per scale tag:
    <dataset><tag>_base.fvecs   first N base vectors, float32
    <dataset>_query.fvecs       queries (written once)
    <dataset><tag>_gt.ivecs     exact k-NN of each query within the N-prefix
"""
from __future__ import annotations

import argparse
import os
import numpy as np

# reuse the vecs writers from the SIFT prep so the on-disk format is identical
from ngmbench.prepare_bigann import fvecs_write, ivecs_write, ivecs_read


# ----------------------------------------------------------------------------- readers
def fbin_shape(path: str) -> tuple[int, int]:
    """(npts, dim) from a big-ann .fbin header (two little-endian int32)."""
    with open(path, "rb") as f:
        npts, dim = np.frombuffer(f.read(8), dtype=np.int32)
    return int(npts), int(dim)


def fbin_read(path: str, count: int | None = None, offset: int = 0) -> np.ndarray:
    """Read `count` float32 vectors from a .fbin, starting at `offset`."""
    _, dim = fbin_shape(path)
    start = 8 + offset * dim * 4              # 8-byte header + row offset
    n = count if count is not None else -1
    raw = np.fromfile(path, dtype=np.float32, count=(n * dim if n > 0 else -1),
                      offset=start)
    return raw.reshape(-1, dim)


def fvecs_shape(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    rec = (dim + 1) * 4
    return os.path.getsize(path) // rec, dim


def fvecs_read(path: str, count: int | None = None, offset: int = 0) -> np.ndarray:
    _, dim = fvecs_shape(path)
    rec = dim + 1
    n = count if count is not None else -1
    raw = np.fromfile(path, dtype=np.float32,
                      count=(n * rec if n > 0 else -1), offset=offset * rec * 4)
    return raw.reshape(-1, rec)[:, 1:]


def parquet_read(path: str, count: int | None = None, col: str = "emb") -> np.ndarray:
    """Read up to `count` embedding rows from a parquet file or a directory of
    parquet shards, concatenating shards in sorted name order. Requires pyarrow."""
    import pyarrow.parquet as pq
    files = ([os.path.join(path, f) for f in sorted(os.listdir(path))
              if f.endswith(".parquet")] if os.path.isdir(path) else [path])
    out, got = [], 0
    for fp in files:
        t = pq.read_table(fp, columns=[col])
        a = np.asarray(t.column(col).to_pylist(), dtype=np.float32)
        out.append(a); got += len(a)
        if count is not None and got >= count:
            break
    arr = np.concatenate(out, axis=0)
    return arr[:count] if count is not None else arr


READERS = {
    "deep":   (fbin_read,    fbin_shape),
    "turing": (fbin_read,    fbin_shape),
    "gist":   (fvecs_read,   fvecs_shape),
    "cohere": (parquet_read, None),          # shape is discovered on read
}


# ----------------------------------------------------------------------------- ground truth
def ibin_read(path: str) -> np.ndarray:
    """big-ann ground-truth .ibin/.bin: uint32 nq, uint32 k, then nq*k uint32
    neighbour indices (distances, if present, follow and are ignored here)."""
    with open(path, "rb") as f:
        nq, k = np.frombuffer(f.read(8), dtype=np.uint32)
        idx = np.frombuffer(f.read(int(nq) * int(k) * 4), dtype=np.uint32)
    return idx.reshape(int(nq), int(k)).astype(np.int32)


def exact_gt(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Exact k-NN (L2) of each query within `base`, blocked to bound memory.
    Returns (nq, k) int32 indices. O(nq * N) - fine to 1M, slow but OK at 10M."""
    nq = query.shape[0]
    gt = np.empty((nq, k), dtype=np.int32)
    qn = (query * query).sum(1)
    bn = (base * base).sum(1)
    QB = 512
    for i in range(0, nq, QB):
        q = query[i:i + QB]
        # ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q.b  (argpartition on this is argmin on true dist)
        d = qn[i:i + QB, None] + bn[None, :] - 2.0 * (q @ base.T)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        rows = np.arange(idx.shape[0])[:, None]
        order = np.argsort(d[rows, idx], axis=1)
        gt[i:i + QB] = idx[rows, order]
    return gt


# ----------------------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(READERS))
    ap.add_argument("--base", required=True, help="base file (or parquet dir for cohere)")
    ap.add_argument("--query", required=True, help="query file (or parquet dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[10_000, 100_000, 1_000_000])
    ap.add_argument("--k", type=int, default=100, help="ground-truth neighbours")
    ap.add_argument("--nq", type=int, default=10_000, help="queries to use")
    ap.add_argument("--gt", default=None,
                    help="shipped ground-truth .ibin/.bin, valid ONLY for the "
                         "scale equal to the base size it was computed against "
                         "(e.g. the native 10M subset). Prefixes still brute-force.")
    ap.add_argument("--gt-scale", type=int, default=None,
                    help="the base size --gt corresponds to; at that scale --gt is "
                         "copied instead of recomputed.")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    read, shape = READERS[a.dataset]

    def tag(n):
        return f"{n // 1_000_000}m" if n >= 1_000_000 else f"{n // 1_000}k"

    # queries once
    qpath = os.path.join(a.out, f"{a.dataset}_query.fvecs")
    if a.dataset == "cohere":
        query = parquet_read(a.query, count=a.nq)
    else:
        query = read(a.query, count=a.nq)
    query = np.ascontiguousarray(query, dtype=np.float32)
    if not os.path.exists(qpath):
        fvecs_write(qpath, query)
        print(f"  wrote {qpath}  ({query.shape[0]}x{query.shape[1]})")

    for n in sorted(a.scales):
        base_dst = os.path.join(a.out, f"{a.dataset}{tag(n)}_base.fvecs")
        gt_dst = os.path.join(a.out, f"{a.dataset}{tag(n)}_gt.ivecs")
        if os.path.exists(base_dst) and os.path.exists(gt_dst):
            print(f"  {tag(n)}: exists, skipping"); continue
        base = np.ascontiguousarray(read(a.base, count=n), dtype=np.float32)
        if base.shape[0] < n:
            print(f"  ! only {base.shape[0]} vectors available (< {n}); using all")
        fvecs_write(base_dst, base)
        print(f"  wrote {base_dst}  ({base.shape[0]}x{base.shape[1]})")
        if a.gt and a.gt_scale == n:
            gt = ibin_read(a.gt)[:, :a.k]
            if gt.shape[0] != query.shape[0]:
                print(f"  ! shipped GT has {gt.shape[0]} queries but query set has "
                      f"{query.shape[0]}; falling back to exact recompute")
                gt = exact_gt(base, query, a.k)
            else:
                print(f"  using shipped GT {a.gt} (native size {n}, no brute-force)")
            ivecs_write(gt_dst, gt)
        else:
            gt = exact_gt(base, query, a.k)
            ivecs_write(gt_dst, gt)
            print(f"  wrote {gt_dst}  (exact L2 {a.k}-NN of {query.shape[0]} queries)")


if __name__ == "__main__":
    main()