"""Run the frozen canonical LayerwiseNNDescentHNSW constructor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .cache import ResultsLog
from .index.fastkcna import inspect_fvecs, prepare_lshkit, sha256_file
from .index.layerwise_nnd import (
    BUILD_NAMESPACE, LayerwiseNNDError, LayerwiseNNDParams,
    LayerwiseNNDPaths, LayerwiseNNDRunner, _expand,
)


def _run_key(value: dict) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _safe_results_path(value: str) -> Path:
    path = _expand(value, "results_path")
    if path.suffix != ".jsonl" or "layerwise_nnd_hnsw_canonical" not in path.name.lower():
        raise LayerwiseNNDError(
            "canonical LayerwiseNNDescentHNSW results require a distinct "
            f"*layerwise_nnd_hnsw_canonical*.jsonl path; refusing {value!r}"
        )
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Canonical LayerwiseNNDescentHNSW builder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--threads", type=int, help="must remain 1 for the canonical baseline")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    try:
        conf = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LayerwiseNNDError(f"cannot read layerwise config {config_path}: {exc}") from exc
    if conf.get("namespace") != BUILD_NAMESPACE:
        raise LayerwiseNNDError(f"config namespace must be exactly {BUILD_NAMESPACE!r}")
    if conf.get("tuning_status") != "untuned canonical":
        raise LayerwiseNNDError("tuning_status must be exactly 'untuned canonical'")
    dataset = dict(conf.get("dataset", {}))
    for field in ("name", "dim", "nb", "base"):
        if field not in dataset:
            raise LayerwiseNNDError(f"dataset config is missing {field!r}")
    source = inspect_fvecs(_expand(dataset["base"], "dataset.base"))
    if (source["n"], source["dim"]) != (int(dataset["nb"]), int(dataset["dim"])):
        raise LayerwiseNNDError("dataset config shape does not match source fvecs")
    values = dict(conf.get("candidate_parameters", {}))
    if args.threads is not None:
        values["threads"] = args.threads
    elif os.environ.get("NGMBENCH_THREADS"):
        values["threads"] = int(os.environ["NGMBENCH_THREADS"])
    try:
        params = LayerwiseNNDParams(**values)
    except TypeError as exc:
        raise LayerwiseNNDError(f"invalid candidate_parameters fields: {exc}") from exc
    params.validate_canonical()
    paths = LayerwiseNNDPaths.resolve(conf.get("binaries", {}))
    provenance = paths.metadata()
    workdir = _expand(conf.get("workdir", ".layerwise_nnd_hnsw_work"), "workdir")
    results_path = _safe_results_path(conf["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = ResultsLog(str(results_path))
    identity = {
        "namespace": BUILD_NAMESPACE,
        "dataset": {"name": dataset["name"], **source},
        "parameters": params.metadata(),
        "fastkcna_revision": provenance["revision"],
        "libkgraph_sha256": provenance["libkgraph_sha256"],
        "builder_sha256": provenance["layerwise_builder_sha256"],
        "tuning_status": conf.get("tuning_status"),
    }
    key = _run_key(identity)
    cached = [record for record in results.load_all() if record.get("run_key") == key]
    if cached:
        if len(cached) != 1:
            raise LayerwiseNNDError(f"duplicate cached construction run_key={key}")
        index = _expand(cached[0].get("output_index_path", ""), "cached.output_index_path")
        expected_sha = cached[0].get("output_index_sha256")
        if not index.is_file() or expected_sha != sha256_file(index):
            raise LayerwiseNNDError(
                f"cached construction run_key={key} has a missing/tampered output index"
            )
        print(f"skip (verified cached result) run_key={key} dataset={dataset['name']}")
        return 0
    converted = workdir / "converted" / f"{dataset['name']}.lshkit"
    conversion = prepare_lshkit(Path(source["path"]), converted, paths.fastkcna)
    runner = LayerwiseNNDRunner(paths, workdir / "runs")
    record = runner.run(
        converted, params, key, n=source["n"], dim=source["dim"], conversion=conversion,
    )
    record.update({
        "run_key": key, "dataset": dataset["name"], "dataset_source": source,
        "config_path": str(config_path), "tuning_status": "untuned canonical",
    })
    results.append(record)
    print(
        f"recorded run_key={key} dataset={dataset['name']} build_calc={record['build_calc']} "
        f"-> {results_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
