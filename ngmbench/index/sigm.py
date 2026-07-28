"""
ngmbench/index/sigm.py

SIGM (Simple Insertion Graph Merge) baseline.

SIGM = take the first sub-graph and sequentially insert the
REMAINING POINTS into it. It is the rebuild/insertion baseline the traversal
merges (NGM / IGTM / CGTM) are designed to beat on *merge cost* -- i.e. the
reviewer's «перестроение». It is NOT the monolithic INSERT baseline (a full
from-scratch build of all N points) that the existing harness reports under
the "INSERT" label. The two are different operations; conflating them is the
bug behind the "merge vs rebuild" confusion on SIFT1M.

Head-to-head quantity: SIGM's merge_calc (the insertion cost) goes in the
same column as the traversal merges' merge_calc.

Requires the experiment.cpp patch (sigm_experiment_cpp.patch). The stock
INSERT branch neither resizes the loaded leaf (so inserting past its
partition capacity throws "The number of elements exceeds the specified
limit") nor prints a distance count. The patch fixes both.
"""
from __future__ import annotations

import os
import json
import uuid

from ngmbench.index.hnswmerger import (
    HNSWMergerRunner,
    Paths,
    CppParams,
    parse_exps,
    _write_kv,
)


def sigm_insert(runner: HNSWMergerRunner, leaf0: str, lrange: int, rrange: int,
                total_n: int, save: bool = False) -> dict:
    """Load leaf0 and insert base points [lrange, rrange) into it (SIGM step).

    max_elements=total_n is what triggers the patched loadIndex to resize the
    leaf (built at partition capacity) up to the full N so the insertion fits.
    Returns parse_exps output: merge_calc (the printed 'distance calls') and
    merge_seconds ('Total time for insertion').
    """
    cfg = os.path.join(runner.p.workdir, f"sigm_{uuid.uuid4().hex[:8]}.cfg")
    _write_kv(cfg, {
        "workload_type": runner.workload, "merge_method": "INSERT",
        "dim": runner.cp.dim, "max_elements": total_n, "nb": total_n,
        "M": runner.cp.M, "ef_construction": runner.cp.ef_construction,
        "k": runner.cp.k, "kk": runner.cp.kk, "nq": runner.cp.nq,
        "iterations": 1, "rerun": "true",
        "save_index": "true" if save else "false",
        "base_filepath": runner.p.base, "query_filepath": runner.p.query,
        "groundtruth_filepath": runner.p.groundtruth,
        "index_path": leaf0,                       # single index, NOT "a,b"
        "lrange": lrange, "rrange": rrange,
        "save_path": runner.p.workdir,
        "efs_array": ", ".join(str(e) for e in runner.cp.efs_array),
    })
    return parse_exps(runner._run(runner.p.exps_bin, cfg), expect_method="INSERT")


def run_sigm(n_parts: int, order: str, paths: Paths, params: CppParams) -> dict:
    """SIGM baseline over n_parts contiguous ranges.

    Builds (or reuses the cached) first leaf, then inserts every remaining
    point [nb/n_parts, nb) into it -- exactly SIGM for the P-way case.

    Accounting: build_calc/build_seconds cover ONLY the first leaf, because
    SIGM re-inserts the other points rather than consuming their graphs.
    merge_calc/merge_seconds are the SIGM insertion cost -- the number to put
    next to the traversal merges' merge_calc. (If you prefer the merge rows'
    convention of summing all P leaf builds for build_calc, add them here; it
    does not affect the merge-column comparison the reviewer is asking about.)
    """
    runner = HNSWMergerRunner(paths, params)
    nb = params.nb
    first_hi = nb // n_parts

    leaf0, b = runner.build_leaf(0, first_hi)
    build_calc = b["build_calc"] or 0
    build_seconds = b["build_seconds"] or 0.0

    out = sigm_insert(runner, leaf0, lrange=first_hi, rrange=nb, total_n=nb)
    merge_calc = out["merge_calc"] or 0
    merge_seconds = out["merge_seconds"] or 0.0

    return {
        "builder": "hnswmerger", "algo": "SIGM", "n_parts": n_parts,
        "partition_method": "range", "order": order,
        "dim": params.dim, "n": nb,
        "m": params.M, "ef_construction": params.ef_construction,
        "build_calc": build_calc, "merge_calc": merge_calc,
        "total_calc": build_calc + merge_calc,
        "build_seconds": build_seconds, "merge_seconds": merge_seconds,
        f"recall@{params.k}": None,        # INSERT returns before eval; see sigm_recall
        "recall_curve": None,
    }


def sigm_recall(runner: HNSWMergerRunner, sigm_index_path: str, total_n: int) -> list | None:
    """OPTIONAL, UNVALIDATED helper for the recall column.

    Run run_sigm's underlying insertion with save=True first (that persists
    sigm_<WORKLOAD>.hnsw via the patch's optional save hunk), then evaluate it
    here. REBUILD with rerun=false simply loads index_path[0] and falls through
    to the query/recall loop, so it doubles as a "just evaluate this index"
    path. Confirm the recall lines parse on your build before trusting it.
    """
    cfg = os.path.join(runner.p.workdir, f"sigm_eval_{uuid.uuid4().hex[:8]}.cfg")
    _write_kv(cfg, {
        "workload_type": runner.workload, "merge_method": "REBUILD",
        "dim": runner.cp.dim, "max_elements": total_n, "nb": total_n,
        "M": runner.cp.M, "ef_construction": runner.cp.ef_construction,
        "k": runner.cp.k, "kk": runner.cp.kk, "nq": runner.cp.nq,
        "iterations": 1, "rerun": "false", "save_index": "false",
        "base_filepath": runner.p.base, "query_filepath": runner.p.query,
        "groundtruth_filepath": runner.p.groundtruth,
        "index_path": sigm_index_path, "save_path": runner.p.workdir,
        "efs_array": ", ".join(str(e) for e in runner.cp.efs_array),
    })
    return parse_exps(runner._run(runner.p.exps_bin, cfg)).get("recall_curve")


def append_row(row: dict, jsonl_path: str, dataset: str) -> None:
    """Append a SIGM row to results_cpp.jsonl with the dataset key up front,
    matching the other hnswmerger rows so make_figures.py picks it up."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps({"dataset": dataset, **row}) + "\n")


if __name__ == "__main__":
    # Wire these exactly as your cli_cpp does. Example (SIFT1M, 2-way):
    #
    #   from ngmbench.index.sigm import run_sigm, append_row
    #   paths = Paths(builds_bin=".../HNSW-Merger/builds",
    #                 exps_bin=".../HNSW-Merger/exps",
    #                 base=".../sift_base.fvecs",
    #                 query=".../sift_query.fvecs",
    #                 groundtruth=".../sift_groundtruth.ivecs",
    #                 workdir=".../scratch")
    #   params = CppParams(dim=128, nb=1_000_000, M=16, ef_construction=200)
    #   for p in (2, 4, 8):
    #       row = run_sigm(p, order="sequential", paths=paths, params=params)
    #       append_row(row, "results_cpp.jsonl", dataset="sift1m")
    #       print(row["algo"], row["n_parts"], "merge_calc=", row["merge_calc"])
    raise SystemExit("import run_sigm/append_row from this module; see the example above")
