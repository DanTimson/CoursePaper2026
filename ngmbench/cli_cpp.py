"""CLI: run an HNSWMerger (C++) sweep and append records to the shared results log.

    python -m ngmbench.cli_cpp --config config/sift1m_cpp.json

Records share the schema the Python harness writes, so the same playground shows
both. C++ runs carry build_calc + merge_calc (distance counts) and a recall_curve
over efs. Leaf indexes are reused if already on disk; runs already in the results
log (matched by run_key) are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os

from .cache import ResultsLog
from .index.hnswmerger import CppParams, Paths, run_hnswmerger


def _run_key(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def expand(spec: dict):
    grid = [k for k, v in spec.items() if isinstance(v, list)]
    if not grid:
        yield dict(spec); return
    for combo in itertools.product(*[spec[k] for k in grid]):
        out = dict(spec); out.update(dict(zip(grid, combo)))
        yield out


def main(argv=None):
    ap = argparse.ArgumentParser(description="HNSWMerger C++ sweep runner.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    conf = json.load(open(args.config))

    ds, hp, ev = conf["dataset"], conf.get("hnsw", {}), conf.get("eval", {})
    paths = Paths(
        builds_bin=conf["binaries"]["builds"], exps_bin=conf["binaries"]["exps"],
        base=ds["base"], query=ds["query"], groundtruth=ds["groundtruth"],
        workdir=conf.get("workdir", ".hnswmerger_work"),
    )
    params = CppParams(
        dim=ds["dim"], nb=ds["nb"],
        M=hp.get("M", 16), ef_construction=hp.get("ef_construction", 200),
        k=ev.get("k", 10), kk=ev.get("kk", 100), nq=ev.get("nq", 10000),
        efs_array=ev.get("efs_array", [10, 50, 100, 200]),
    )

    results = ResultsLog(conf.get("results_path", "results.jsonl"))
    done = {r.get("run_key") for r in results.load_all()}

    runs = []
    for spec in conf.get("sweep", []):
        for s in expand(spec):
            runs.append((s["algo"], int(s["n_parts"]), s.get("order", "balanced")))

    print(f"{len(runs)} C++ run(s) -> {results.path}")
    for i, (algo, n_parts, order) in enumerate(runs, 1):
        key = _run_key({"b": "hnswmerger", "ds": ds["name"], "algo": algo,
                        "np": n_parts, "order": order, "M": params.M,
                        "efc": params.ef_construction, "nb": params.nb})
        if key in done:
            print(f"[{i}/{len(runs)}] skip (cached) {algo} parts={n_parts} {order}")
            continue
        rec = run_hnswmerger(algo, n_parts, order, paths, params)
        rec["run_key"] = key
        rec["dataset"] = ds["name"]
        results.append(rec)
        r = rec.get(f"recall@{params.k}")
        print(f"[{i}/{len(runs)}] {algo:10} parts={n_parts} {order:10} "
              f"build_calc={rec['build_calc']} merge_calc={rec['merge_calc']} "
              f"recall@{params.k}={r}")


if __name__ == "__main__":
    main()
