"""Evaluate an explicitly selected canonical stock-HNSW index.

The accepted FastHNSW and LayerwiseNNDescentHNSW peers share the unchanged
stock evaluator, exact query counter, Recall@k, and bracketed d_s helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .cache import ResultsLog
from .index.fasthnsw import (
    QUALITY_NAMESPACE,
    FastHNSWError,
    FastHNSWRunner,
    _expand_path,
    enrich_quality_record,
    inspect_vecs,
    read_groundtruth_topk,
    select_stock_hnsw_construction,
    sha256_file,
    validate_construction_dataset,
)
from .index.layerwise_nnd import QUALITY_NAMESPACE as LAYERWISE_QUALITY_NAMESPACE


def _run_key(value: dict) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _safe_results_path(value: str, *, quality_namespace: str = QUALITY_NAMESPACE) -> Path:
    path = _expand_path(value, "results_path")
    required = (
        "fasthnsw_quality" if quality_namespace == QUALITY_NAMESPACE
        else "layerwise_nnd_hnsw_quality"
    )
    if path.suffix != ".jsonl" or required not in path.name.lower():
        raise FastHNSWError(
            f"{quality_namespace} results must use a distinct *{required}*.jsonl path; "
            f"refusing {value!r}"
        )
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing canonical stock-HNSW index without rebuilding it"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--construction-results", help="override canonical construction JSONL")
    parser.add_argument("--construction-run-key", help="override explicit canonical construction run key")
    parser.add_argument("--evaluator", help="override stock evaluator executable")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    try:
        conf = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FastHNSWError(f"cannot read FastHNSW quality config {config_path}: {exc}") from exc
    quality_namespace = conf.get("namespace")
    if quality_namespace == QUALITY_NAMESPACE:
        construction_kind = "fasthnsw"
        quality_algorithm = "stock-hnswlib-fasthnsw"
        quality_algo = "FastHNSW"
    elif quality_namespace == LAYERWISE_QUALITY_NAMESPACE:
        construction_kind = "layerwise_nnd_hnsw"
        quality_algorithm = "stock-hnswlib-layerwise-nnd-hnsw"
        quality_algo = "L-NND-HNSW"
    else:
        raise FastHNSWError(
            f"config namespace must be {QUALITY_NAMESPACE!r} or "
            f"{LAYERWISE_QUALITY_NAMESPACE!r}"
        )

    selection = dict(conf.get("construction", {}))
    construction_results = args.construction_results or selection.get("results_path")
    construction_run_key = args.construction_run_key or selection.get("run_key")
    if not construction_results or not construction_run_key:
        raise FastHNSWError(
            "canonical construction selection requires results_path and explicit run_key "
            "(in config construction or CLI overrides)"
        )
    construction = select_stock_hnsw_construction(
        _expand_path(construction_results, "construction.results_path"),
        construction_run_key, construction_kind=construction_kind,
    )

    dataset = dict(conf.get("dataset", {}))
    for field in ("name", "dim", "nb", "base", "query", "groundtruth"):
        if field not in dataset:
            raise FastHNSWError(f"dataset config is missing {field!r}")
    n, dim = int(dataset["nb"]), int(dataset["dim"])
    base = _expand_path(dataset["base"], "dataset.base")
    query = _expand_path(dataset["query"], "dataset.query")
    groundtruth = _expand_path(dataset["groundtruth"], "dataset.groundtruth")
    base_info = inspect_vecs(base, kind="fvecs")
    validate_construction_dataset(construction, dataset, base_info)

    eval_conf = dict(conf.get("eval", {}))
    k, kk, nq = int(eval_conf.get("k", 10)), int(eval_conf.get("kk", 100)), int(eval_conf.get("nq", 10000))
    efs = eval_conf.get("efs_array")
    if not isinstance(efs, list) or not efs or any(type(value) is not int for value in efs):
        raise FastHNSWError("eval.efs_array must be a nonempty list of integers")
    if len(set(efs)) != len(efs) or efs != sorted(efs) or any(value < k for value in efs):
        raise FastHNSWError("eval.efs_array must be unique, increasing, and every ef must be >= k")
    if not (0 < k <= kk and 0 < nq <= n):
        raise FastHNSWError(f"invalid eval dimensions: k={k}, kk={kk}, nq={nq}, n={n}")
    query_info = inspect_vecs(query, kind="fvecs")
    if (query_info["n"], query_info["dim"]) != (nq, dim):
        raise FastHNSWError(
            f"query shape mismatch: expected {(nq, dim)}, got {(query_info['n'], query_info['dim'])}"
        )
    gt_info = inspect_vecs(groundtruth, kind="ivecs")
    if (gt_info["n"], gt_info["dim"]) != (nq, kk):
        raise FastHNSWError(
            f"ground-truth shape mismatch: expected {(nq, kk)}, got {(gt_info['n'], gt_info['dim'])}"
        )

    binaries = dict(conf.get("binaries", {}))
    evaluator_value = (
        args.evaluator or os.environ.get("FASTHNSW_EVALUATOR") or
        binaries.get("evaluator") or "cpp/fast_hnsw_quality_eval"
    )
    evaluator = _expand_path(evaluator_value, "binaries.evaluator")
    workdir = _expand_path(conf.get("workdir", ".fasthnsw_quality_work"), "workdir")
    identity_samples = int(eval_conf.get("identity_samples", 8))
    if not (0 < identity_samples <= n):
        raise FastHNSWError("eval.identity_samples must be in [1,n]")
    evaluator_probe = FastHNSWRunner(evaluator, workdir)

    # Hash every live artifact used by the evaluator/recall computation. Index
    # hash was already checked against the immutable construction record.
    artifacts = {
        "base": {**base_info, "sha256": sha256_file(base)},
        "query": {**query_info, "sha256": sha256_file(query)},
        "groundtruth": {**gt_info, "sha256": sha256_file(groundtruth)},
        "index": {
            "path": construction["output_index_path"],
            "sha256": construction["verified_output_index_sha256"],
            "construction_record_sha256": construction["output_index_sha256"],
        },
    }
    identity = {
        "namespace": quality_namespace,
        "construction_results_path": construction["construction_results_path"],
        "construction_run_key": construction["run_key"],
        "index_sha256": construction["verified_output_index_sha256"],
        "dataset": {"name": dataset["name"], "n": n, "dim": dim},
        "artifact_sha256": {name: value["sha256"] for name, value in artifacts.items()},
        "eval": {"k": k, "kk": kk, "nq": nq, "efs_array": efs, "identity_samples": identity_samples},
        "evaluator_sha256": evaluator_probe.evaluator_sha256,
    }
    quality_run_key = _run_key(identity)
    results_path = _safe_results_path(
        conf["results_path"], quality_namespace=quality_namespace,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = ResultsLog(str(results_path))
    if quality_run_key in {row.get("quality_run_key") for row in results.load_all()}:
        print(f"skip (cached quality result) quality_run_key={quality_run_key}")
        return 0

    runner = FastHNSWRunner(evaluator, workdir / quality_run_key)
    gt_topk = read_groundtruth_topk(groundtruth, nq=nq, kk=kk, k=k, n=n)
    point_results = []
    for ef in efs:
        point_results.append(runner.run_point(
            index=Path(construction["output_index_path"]), query=query, base=base,
            ef=ef, k=k, nq=nq, dim=dim, n=n,
            index_sha256=construction["verified_output_index_sha256"],
            identity_samples=identity_samples,
        ))
    record = enrich_quality_record(
        construction, point_results, gt_topk,
        k=k, kk=kk, nq=nq, expected_efs=efs, quality_run_key=quality_run_key,
        evaluator_metadata={
            "path": str(evaluator), "sha256": evaluator_probe.evaluator_sha256,
            "identity_samples_requested": identity_samples,
        },
        artifact_metadata=artifacts, config_path=config_path,
        analysis_dataset=dataset.get("analysis_name"),
        quality_namespace=quality_namespace,
        quality_algorithm=quality_algorithm,
        quality_algo=quality_algo,
    )
    results.append(record)
    print(
        f"recorded construction_run_key={construction['run_key']} "
        f"quality_run_key={quality_run_key} points={len(efs)} -> {results_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
