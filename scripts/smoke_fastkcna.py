#!/usr/bin/env python3
"""Run only a deterministic 128-vector FastKCNA/compatibility smoke test."""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

from ngmbench.index.fastkcna import (
    FastKCNAParams,
    FastKCNAPaths,
    FastKCNARunner,
    check_fasthnsw_compatibility,
    prepare_lshkit,
)


def write_vecs(path: Path, data: np.ndarray) -> None:
    blob = b"".join(
        struct.pack("<i", data.shape[1]) + np.asarray(row, dtype="<f4").tobytes()
        for row in data
    )
    if not path.exists() or path.read_bytes() != blob:
        path.write_bytes(blob)


def write_ivecs(path: Path, data: np.ndarray) -> None:
    blob = b"".join(
        struct.pack("<i", data.shape[1]) + np.asarray(row, dtype="<i4").tobytes()
        for row in data
    )
    if not path.exists() or path.read_bytes() != blob:
        path.write_bytes(blob)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/coursepaper-fastkcna-smoke")
    ap.add_argument("--fastkcna-root")
    ap.add_argument("--hnswmerger-exps", default="$HNSWMERGER_BIN/exps")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    base = rng.normal(size=(128, 16)).astype("<f4")
    query = base[:16].copy()
    distances = ((query[:, None, :] - base[None, :, :]) ** 2).sum(axis=2)
    groundtruth = np.argsort(distances, axis=1)[:, :10].astype("<i4")
    base_path, query_path, gt_path = workdir / "base.fvecs", workdir / "query.fvecs", workdir / "gt.ivecs"
    write_vecs(base_path, base)
    write_vecs(query_path, query)
    write_ivecs(gt_path, groundtruth)

    binary_config = {"checkout": args.fastkcna_root} if args.fastkcna_root else {}
    paths = FastKCNAPaths.resolve(binary_config)
    conversion = prepare_lshkit(base_path, workdir / "base.lshkit", paths)
    runner = FastKCNARunner(paths, workdir / "runs")
    common = dict(
        K=16, L=16, S=4, R=8, iter=2, search_L=16, search_K=16,
        nsg_R=4, step=2, loop_i=1, alpha=60, tau=0,
        nthreads=args.threads, controls=16, recall=0.99,
    )
    pg0 = runner.run(workdir / "base.lshkit", FastKCNAParams(pg_type=0, **common), "smoke-pg0", conversion)
    pg2 = runner.run(workdir / "base.lshkit", FastKCNAParams(pg_type=2, **common), "smoke-pg2", conversion)

    exps = Path(os.path.expandvars(os.path.expanduser(args.hnswmerger_exps)))
    compatibility = check_fasthnsw_compatibility(
        exps_bin=exps,
        index_path=Path(pg2["output_index_path"]),
        base_path=base_path,
        query_path=query_path,
        groundtruth_path=gt_path,
        workdir=workdir / "compatibility",
        dim=16, nb=128, nq=16,
    )
    summary = {
        "scope": "tiny smoke only; not an experimental result",
        "pg0": pg0,
        "pg2": pg2,
        "fasthnsw_existing_evaluator_compatibility": compatibility,
    }
    summary_path = workdir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "summary": str(summary_path),
        "pg0_exit": pg0["exit_status"], "pg0_index": pg0["output_index_path"],
        "pg2_exit": pg2["exit_status"], "pg2_index": pg2["output_index_path"],
        "compatibility_exit": compatibility["exit_status"],
        "compatible_with_existing_evaluator": compatibility["compatible"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
