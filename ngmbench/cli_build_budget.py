"""Run build-only partition sweeps against the instrumented HNSWMerger backend.

This deliberately does not run a merge. For each dataset size N and partition
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
    """Stable identity for one direct build-budget point.

    Only construction-defining fields participate. In particular, changing the
    output file or workdir does not manufacture a new experimental identity,
    while M / ef_construction / thread changes do.
    """
    identity = {
        k: record.get(k)
        for k in (
            "builder",
            "dataset",
            "n_parts",
            "partition_method",
            "m",
            "ef_construction",
            "threads",
        )
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure monolithic and P-leaf HNSW build costs without merging."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        conf = json.load(handle)

    hp = conf.get("hnsw", {})
    eval_cfg = conf.get("eval", {})
    partitions = [int(p) for p in conf.get("partitions", [1, 2, 4, 8, 16])]
    if not partitions or any(p <= 0 for p in partitions):
        raise ValueError("partitions must contain positive integers")

    results = ResultsLog(conf.get("results_path", "results/build_budget.jsonl"))
    Path(results.path).parent.mkdir(parents=True, exist_ok=True)
    done = {r.get("run_key") for r in results.load_all() if r.get("run_key")}

    datasets = conf.get("datasets") or []
    if not datasets:
        raise ValueError("config must contain a non-empty datasets list")

    for ds in datasets:
        dataset_workdir = ds.get("workdir") or os.path.join(
            conf.get("workdir_root", ".build_budget_work"), ds["name"]
        )
        paths = Paths(
            builds_bin=conf["binaries"]["builds"],
            exps_bin=conf["binaries"]["exps"],
            base=ds["base"],
            query=ds["query"],
            groundtruth=ds["groundtruth"],
            workdir=dataset_workdir,
        )
        params = CppParams(
            dim=int(ds["dim"]),
            nb=int(ds["nb"]),
            M=int(hp.get("M", 16)),
            ef_construction=int(hp.get("ef_construction", 200)),
            k=int(eval_cfg.get("k", 10)),
            kk=int(eval_cfg.get("kk", 100)),
            nq=int(eval_cfg.get("nq", 10000)),
            efs_array=list(eval_cfg.get("efs_array", [10])),
            thread=int(conf.get("threads", 1)),
        )
        runner = HNSWMergerRunner(paths, params)

        for p in partitions:
            if p > params.nb:
                print(f"skip {ds['name']} P={p}: P > N")
                continue

            placeholder = {
                "builder": "hnswmerger-build-only",
                "algo": "BUILD_ONLY",
                "dataset": ds["name"],
                "dim": params.dim,
                "n": params.nb,
                "n_parts": p,
                "partition_method": "range",
                "order": None,
                "m": params.M,
                "ef_construction": params.ef_construction,
                "threads": params.thread,
            }
            key = _run_key(placeholder)
            if key in done:
                print(f"skip cached log row: {ds['name']} P={p}")
                continue

            leaves = []
            total_calc = 0
            total_seconds = 0.0
            ranges = contiguous_partitions(params.nb, p)
            for shard_id, (lo, hi) in enumerate(ranges):
                idx, build = runner.build_leaf(lo, hi)
                calc = int(build.get("build_calc") or 0)
                seconds = float(build.get("build_seconds") or 0.0)
                total_calc += calc
                total_seconds += seconds
                leaves.append({
                    "shard_id": shard_id,
                    "lrange": lo,
                    "rrange": hi,
                    "size": hi - lo,
                    "build_calc": calc,
                    "build_seconds": seconds,
                    "cached": bool(build.get("cached", False)),
                    "index_path": idx,
                })

            record = {
                **placeholder,
                "run_key": key,
                "build_calc": total_calc,
                "merge_calc": 0,
                "total_calc": total_calc,
                "build_seconds": total_seconds,
                "merge_seconds": 0.0,
                "leaf_builds": leaves,
                "config": asdict(params),
            }
            if "experiment_metadata" in conf:
                record["experiment_metadata"] = conf["experiment_metadata"]
                record["config_path"] = os.path.abspath(args.config)

            results.append(record)
            done.add(key)
            print(
                f"{ds['name']} P={p} leaf_build_calc={total_calc} "
                f"per_point={total_calc / params.nb:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
