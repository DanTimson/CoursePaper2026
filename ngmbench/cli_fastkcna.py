"""CLI for bounded/exploratory runs against an external FastKCNA checkout.

Example:
    python -m ngmbench.cli_fastkcna --config config/fastkcna_sift100k_pg0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .cache import ResultsLog
from .index.fastkcna import (
    FastKCNAError,
    FastKCNAParams,
    FastKCNAPaths,
    FastKCNARunner,
    inspect_fvecs,
    prepare_lshkit,
)


def _run_key(value: dict) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _safe_results_path(value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if "$" in str(path):
        raise FastKCNAError(f"results_path contains an unresolved environment variable: {value!r}")
    if "fastkcna" not in path.name.lower() or path.suffix != ".jsonl":
        raise FastKCNAError(
            "FastKCNA results must use a new, explicit *fastkcna*.jsonl namespace; "
            f"refusing results_path={value!r}"
        )
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="External FastKCNA exploratory runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--threads", type=int, help="override config threads/nthreads")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    conf = json.loads(config_path.read_text())
    if conf.get("namespace") != "fastkcna-exploratory":
        raise FastKCNAError("config namespace must be exactly 'fastkcna-exploratory'")
    paths = FastKCNAPaths.resolve(conf.get("binaries", {}))
    provenance = paths.metadata()

    dataset = dict(conf["dataset"])
    source = inspect_fvecs(Path(os.path.expandvars(os.path.expanduser(dataset["base"]))))
    if int(dataset["nb"]) != source["n"] or int(dataset["dim"]) != source["dim"]:
        raise FastKCNAError(
            f"dataset config shape {(dataset['nb'], dataset['dim'])} does not match "
            f"fvecs shape {(source['n'], source['dim'])}: {source['path']}"
        )
    params_dict = dict(conf["fastkcna_params"])
    requested_threads = args.threads
    if requested_threads is None and os.environ.get("NGMBENCH_THREADS"):
        requested_threads = int(os.environ["NGMBENCH_THREADS"])
    if requested_threads is None:
        requested_threads = int(conf.get("threads", params_dict.get("nthreads", 1)))
    params_dict["nthreads"] = requested_threads
    params = FastKCNAParams(**params_dict)

    workdir = Path(os.path.expandvars(os.path.expanduser(conf.get("workdir", ".fastkcna_work")))).resolve()
    results_path = _safe_results_path(conf["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = ResultsLog(str(results_path))
    identity = {
        "namespace": conf["namespace"],
        "dataset": {"name": dataset["name"], **source},
        "params": params.complete_metadata(),
        "fastkcna_revision": provenance["revision"],
        "build_index_sha256": provenance["build_index_sha256"],
        "tuning_status": conf.get("tuning_status"),
    }
    key = _run_key(identity)
    done = {record.get("run_key") for record in results.load_all()}
    if key in done:
        print(f"skip (cached result) run_key={key} pg_type={params.pg_type}")
        return 0

    converted = workdir / "converted" / f"{dataset['name']}.lshkit"
    conversion = prepare_lshkit(Path(source["path"]), converted, paths)
    runner = FastKCNARunner(paths, workdir / "runs")
    record = runner.run(converted, params, key, conversion=conversion)
    record.update({
        "run_key": key,
        "namespace": conf["namespace"],
        "dataset": dataset["name"],
        "dataset_source": source,
        "config_path": str(config_path),
        "tuning_status": "untuned exploratory",
    })
    results.append(record)
    print(
        f"recorded run_key={key} pg_type={params.pg_type} threads={params.nthreads} "
        f"wall_seconds={record['wall_seconds']:.6f} -> {results_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
