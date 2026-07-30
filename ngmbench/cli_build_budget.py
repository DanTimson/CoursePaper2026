"""Run build-only partition sweeps against the instrumented HNSWMerger backend.

This deliberately does not run a merge.  For each dataset size N and partition
count P it builds every actual disjoint leaf and records

    L(N, P) = sum_j B(S_j)

rather than estimating the leaf-build total as P * B(N/P).

Example:
    python -m ngmbench.cli_build_budget \
        --config config/build_budget_bigann.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from .cache import ResultsLog
from .index.hnswmerger import (
    CppParams,
    HNSWMergerRunner,
    Paths,
    contiguous_partitions,
)


def _run_key(record: dict) -> str:
    identity = {
        "builder": record["builder"],
        "dataset": record["dataset"],
        "n": record["n"],
        "dim": record["dim"],
        "n_parts": record["n_parts"],
        "partition_method": record["partition_method"],
        "m": record["m"],
        "ef_construction": record["ef_construction"],
        "threads": record["threads"],
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure monolithic and P-leaf HNSW build costs without merging."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        conf = json.load(handle)

    hnsw = conf.get("hnsw", {})
    eval_cfg = conf.get("eval", {})
    partitions = [int(p) for p in conf.get("partitions", [1, 2, 4, 8, 16])]
    if not partitions or any(p <= 0 for p in partitions):
        raise ValueError("partitions must contain positive integers")

    results_path = conf.get("results_path", "results/build_budget.jsonl")
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results = ResultsLog(results_path)
    done = {row.get("run_key") for row in results.load_all()}

    for ds in conf["datasets"]:
        dataset_workdir = ds.get(
            "workdir",
            os.path.join(conf.get("workdir_root", ".build_budget_work"), ds["name"]),
        )
        paths = Paths(
            builds_bin=conf["binaries"]["builds"],
            exps_bin=conf["binaries"]["exps"],  # required by Paths; unused here
            base=ds["base"],
            query=ds["query"],
            groundtruth=ds["groundtruth"],
            workdir=dataset_workdir,
        )
        params = CppParams(
            dim=int(ds["dim"]),
            nb=int(ds["nb"]),
            M=int(hnsw.get("M", 16)),
            ef_construction=int(hnsw.get("ef_construction", 200)),
            k=int(eval_cfg.get("k", 10)),
            kk=int(eval_cfg.get("kk", 100)),
            nq=int(eval_cfg.get("nq", 10000)),
            efs_array=list(eval_cfg.get("efs_array", [10])),
            thread=int(conf.get("threads", 1)),
        )
        runner = HNSWMergerRunner(paths, params)

        for n_parts in partitions:
            if n_parts > params.nb:
                print(f"skip {ds['name']} P={n_parts}: P > N")
                continue

            ranges = contiguous_partitions(params.nb, n_parts)
            placeholder = {
                "builder": "hnswmerger-build-only",
                "algo": "BUILD_ONLY",
                "dataset": ds["name"],
                "n": params.nb,
                "dim": params.dim,
                "n_parts": n_parts,
                "partition_method": "range",
                "order": None,
                "m": params.M,
                "ef_construction": params.ef_construction,
                "threads": params.thread,
            }
            key = _run_key(placeholder)
            if key in done:
                print(f"skip cached log row: {ds['name']} P={n_parts}")
                continue

            leaves = []
            total_calc = 0
            total_seconds = 0.0
            for shard_id, (lo, hi) in enumerate(ranges):
                index_path, build = runner.build_leaf(lo, hi)
                calc = int(build.get("build_calc") or 0)
                seconds = float(build.get("build_seconds") or 0.0)
                total_calc += calc
                total_seconds += seconds
                leaves.append(
                    {
                        "shard_id": shard_id,
                        "lrange": lo,
                        "rrange": hi,
                        "size": hi - lo,
                        "build_calc": calc,
                        "build_seconds": seconds,
                        "cached": bool(build.get("cached", False)),
                        "index_path": index_path,
                    }
                )

            record = {
                **placeholder,
                "run_key": key,
                "build_calc": total_calc,
                "merge_calc": 0,
                "total_calc": total_calc,
                "build_seconds": total_seconds,
                "merge_seconds": 0.0,
                "leaf_builds": leaves,
                "config": {
                    "M": params.M,
                    "ef_construction": params.ef_construction,
                    "threads": params.thread,
                },
            }
            results.append(record)
            done.add(key)
            print(
                f"{ds['name']:12} P={n_parts:2d} "
                f"leaf_build_calc={total_calc:,} "
                f"per_point={total_calc / params.nb:.3f}"
            )


if __name__ == "__main__":
    main()
