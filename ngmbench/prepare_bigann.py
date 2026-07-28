#!/usr/bin/env python3
"""Prepare SIFT scale-sweep datasets as prefixes of BIGANN (ANN_SIFT1B).

Why prefixes: ANN_SIFT1M is NOT the same data as the first 1M of SIFT1B, so
mixing sources would vary the distribution *and* N at once and confound the
scale trend. Every scale here is the first N vectors of bigann_base.bvecs.

    python prepare_bigann.py --src /data/bigann --out /data/sift_scales \
                             --scales 10000 100000 1000000 10000000

Inputs expected in --src (from http://corpus-texmex.irisa.fr/):
    bigann_base.bvecs          (uint8, 128-d)
    bigann_query.bvecs         (uint8, 128-d, 10000 queries)
    gnd/idx_1M.ivecs           (optional; official GT for the 1M prefix)
    gnd/idx_10M.ivecs          (optional; official GT for the 10M prefix)

Outputs per scale N in --out:
    sift{tag}_base.fvecs       first N base vectors, float32
    sift_query.fvecs           queries, float32 (written once)
    sift{tag}_gt.ivecs         ground truth, top-`--k` neighbour ids

Ground truth: official gnd/idx_{N}.ivecs is used when present (it is provided
for n = 1M, 2M, 5M, 10M, 20M, 50M, 100M, 200M, 500M, 1B) and truncated to --k.
For non-official prefixes (10k, 100k) it is computed by exact brute force.

Note on format: the harness reads fvecs, so bvecs are widened to float32 here.
That costs 4x on disk (10M = 5.1 GB as fvecs vs 1.3 GB as bvecs) but avoids a
second C++ reader patch. SIFT values are integers 0..255, so widening is exact.
"""
from __future__ import annotations

import argparse
import os
import numpy as np


def bvecs_count(path: str, dim_hint: int = 128) -> tuple[int, int]:
    """(n_vectors, dim) of a .bvecs file, from the first record and file size."""
    with open(path, "rb") as f:
        dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    rec = 4 + dim
    n = os.path.getsize(path) // rec
    if dim_hint and dim != dim_hint:
        print(f"  ! {os.path.basename(path)} reports dim={dim} (expected {dim_hint})")
    return n, dim


def bvecs_read(path: str, count: int, offset: int = 0) -> np.ndarray:
    """Read `count` vectors starting at `offset` from a .bvecs file -> uint8 (count, dim)."""
    _, dim = bvecs_count(path)
    rec = 4 + dim
    raw = np.fromfile(path, dtype=np.uint8, count=count * rec, offset=offset * rec)
    return raw.reshape(-1, rec)[:, 4:]


def fvecs_write(path: str, arr: np.ndarray) -> None:
    """Write float32 (n, dim) as .fvecs, streaming in chunks to bound memory."""
    n, dim = arr.shape
    hdr = np.array([dim], dtype=np.int32)
    with open(path, "wb") as f:
        for i in range(0, n, 100_000):
            blk = arr[i:i + 100_000].astype(np.float32, copy=False)
            out = np.empty((blk.shape[0], dim + 1), dtype=np.float32)
            out[:, 0] = hdr.view(np.float32)[0]
            out[:, 1:] = blk
            out.tofile(f)


def ivecs_write(path: str, arr: np.ndarray) -> None:
    n, dim = arr.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = arr.astype(np.int32, copy=False)
    out.tofile(path)


def ivecs_read(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    dim = int(a[0])
    return a.reshape(-1, dim + 1)[:, 1:]


def convert_base(src: str, out: str, n: int, tag: str) -> str:
    """Widen the first n bvecs to fvecs, streaming so 10M doesn't blow up RAM."""
    dst = os.path.join(out, f"sift{tag}_base.fvecs")
    if os.path.exists(dst):
        print(f"  base exists, skipping: {dst}"); return dst
    _, dim = bvecs_count(src)
    rec = 4 + dim
    hdr = np.array([dim], dtype=np.int32).view(np.float32)[0]
    written = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while written < n:
            take = min(200_000, n - written)
            raw = np.frombuffer(fin.read(take * rec), dtype=np.uint8)
            if raw.size == 0:
                break
            blk = raw.reshape(-1, rec)[:, 4:]
            buf = np.empty((blk.shape[0], dim + 1), dtype=np.float32)
            buf[:, 0] = hdr
            buf[:, 1:] = blk
            buf.tofile(fout)
            written += blk.shape[0]
    print(f"  wrote {dst}  ({written} x {dim}, {os.path.getsize(dst)/1e9:.2f} GB)")
    return dst


def brute_force_gt(base_path: str, queries: np.ndarray, k: int,
                   chunk: int = 200_000) -> np.ndarray:
    """Exact top-k by L2, streaming the base in chunks.

    Uses ||b||^2 - 2 q.b as the ranking key (||q||^2 is constant per query and
    does not affect the ordering), so only one matmul per chunk is needed.
    """
    dim = queries.shape[1]
    nq = queries.shape[0]
    rec = 4 + dim * 4  # fvecs record size (float32)
    total = os.path.getsize(base_path) // rec
    best_d = np.full((nq, 0), 0.0, dtype=np.float32)
    best_i = np.zeros((nq, 0), dtype=np.int64)
    q = queries.astype(np.float32, copy=False)
    seen = 0
    with open(base_path, "rb") as f:
        while seen < total:
            take = min(chunk, total - seen)
            raw = np.frombuffer(f.read(take * rec), dtype=np.float32)
            blk = raw.reshape(-1, dim + 1)[:, 1:]
            d = (blk * blk).sum(1)[None, :] - 2.0 * (q @ blk.T)
            kk = min(k, d.shape[1])
            idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
            dv = np.take_along_axis(d, idx, axis=1)
            best_d = np.hstack([best_d, dv])
            best_i = np.hstack([best_i, idx + seen])
            keep = np.argpartition(best_d, min(k, best_d.shape[1]) - 1, axis=1)[:, :k]
            best_d = np.take_along_axis(best_d, keep, axis=1)
            best_i = np.take_along_axis(best_i, keep, axis=1)
            seen += blk.shape[0]
            print(f"    brute force {seen}/{total}", end="\r")
    order = np.argsort(best_d, axis=1)
    print()
    return np.take_along_axis(best_i, order, axis=1)


def tag_for(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with bigann_base.bvecs, bigann_query.bvecs, gnd/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[10_000, 100_000, 1_000_000, 10_000_000])
    ap.add_argument("--k", type=int, default=100, help="neighbours per query to store (>= harness kk)")
    ap.add_argument("--nq", type=int, default=0, help="cap query count (0 = all)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    base_src = os.path.join(a.src, "bigann_base.bvecs")
    query_src = os.path.join(a.src, "bigann_query.bvecs")
    for p in (base_src, query_src):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")

    nb_total, dim = bvecs_count(base_src)
    print(f"base: {nb_total} vectors of dim {dim}")

    q = bvecs_read(query_src, count=10_000).astype(np.float32)
    if a.nq:
        q = q[:a.nq]
    qpath = os.path.join(a.out, "sift_query.fvecs")
    if not os.path.exists(qpath):
        fvecs_write(qpath, q)
    print(f"queries: {q.shape[0]} x {q.shape[1]} -> {qpath}")

    for n in a.scales:
        if n > nb_total:
            print(f"! skip {n}: exceeds base size {nb_total}"); continue
        tag = tag_for(n)
        print(f"=== scale {n} (tag {tag}) ===")
        bpath = convert_base(base_src, a.out, n, tag)

        gtpath = os.path.join(a.out, f"sift{tag}_gt.ivecs")
        if os.path.exists(gtpath):
            print(f"  gt exists, skipping: {gtpath}"); continue

        official = os.path.join(a.src, "gnd", f"idx_{tag.upper()}.ivecs")
        if os.path.exists(official):
            gt = ivecs_read(official)[:q.shape[0], :a.k]
            print(f"  using official GT {official} -> top-{gt.shape[1]}")
        else:
            print(f"  no official GT for {tag}; computing exact brute force (k={a.k})")
            gt = brute_force_gt(bpath, q, a.k)
        ivecs_write(gtpath, gt)
        print(f"  wrote {gtpath}  ({gt.shape[0]} x {gt.shape[1]})")

    print("\ndone. Point each config's base/query/groundtruth at these files.")


if __name__ == "__main__":
    main()
