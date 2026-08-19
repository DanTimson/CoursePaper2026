"""Stock-hnswlib FastHNSW quality evaluation integration.

The graph is selected from an immutable canonical FastKCNA construction record.
This module never converts or rebuilds it: it validates the recorded index hash,
drives the repository-owned stock-hnswlib evaluator once per ``efSearch`` point,
and computes the existing main-body top-k set-intersection recall in Python.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ngmbench.quality import ds_at_recall


QUERY_RECORD_PREFIX = "COURSEPAPER_FASTHNSW_QUERY "
QUERY_SCHEMA = "coursepaper.fasthnsw.query"
QUERY_VERSION = 1
QUERY_INSTRUMENTATION = "stock-hnswlib-v0.8.0-counted-l2-v1"
QUALITY_NAMESPACE = "fasthnsw-quality"
PINNED_FASTKCNA_REVISION = "e2f2d79d3de92419e7feea2f1a79d9efc5746f1d"
PINNED_HNSWLIB_REVISION = "3f3429661187e4c24a490a0f148fc6bc89042b3d"


class FastHNSWError(RuntimeError):
    """A canonical-selection, data-validation, evaluator, or result error."""


def sha256_file(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _expand_path(value: str | os.PathLike, field: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
        raise FastHNSWError(f"{field} contains an unresolved environment variable: {value!r}")
    return Path(expanded).resolve()


def inspect_vecs(path: Path, *, kind: str) -> dict:
    """Strictly inspect fvecs/ivecs shape and every per-row dimension header."""
    if kind not in ("fvecs", "ivecs"):
        raise ValueError("kind must be 'fvecs' or 'ivecs'")
    path = Path(path).resolve()
    if not path.is_file():
        raise FastHNSWError(f"{kind} file is missing: {path}")
    size = path.stat().st_size
    if size < 4:
        raise FastHNSWError(f"{kind} file is truncated: {path}")
    with path.open("rb") as stream:
        raw = stream.read(4)
    dim = struct.unpack("<i", raw)[0]
    if dim <= 0:
        raise FastHNSWError(f"{kind} has invalid row dimension {dim}: {path}")
    record_size = 4 * (dim + 1)
    if size % record_size:
        raise FastHNSWError(
            f"{kind} size {size} is not divisible by record size {record_size} "
            f"(dim={dim}): {path}"
        )
    n = size // record_size
    if n <= 0:
        raise FastHNSWError(f"{kind} contains no rows: {path}")
    # Validate all row headers without materialising vector payloads.
    with path.open("rb") as stream:
        for row in range(n):
            header = stream.read(4)
            if len(header) != 4 or struct.unpack("<i", header)[0] != dim:
                observed = None if len(header) != 4 else struct.unpack("<i", header)[0]
                raise FastHNSWError(
                    f"{kind} row {row} header mismatch: expected {dim}, got {observed}: {path}"
                )
            stream.seek(dim * 4, os.SEEK_CUR)
    stat = path.stat()
    return {
        "path": str(path), "kind": kind, "n": int(n), "dim": int(dim),
        "size": size, "mtime_ns": stat.st_mtime_ns,
    }


def read_groundtruth_topk(path: Path, *, nq: int, kk: int, k: int, n: int) -> list[tuple[int, ...]]:
    """Read/validate an ivecs GT artifact and retain its first k labels per row."""
    if not (0 < k <= kk and nq > 0 and n >= k):
        raise FastHNSWError(f"invalid GT request: nq={nq}, kk={kk}, k={k}, n={n}")
    info = inspect_vecs(path, kind="ivecs")
    if (info["n"], info["dim"]) != (nq, kk):
        raise FastHNSWError(
            f"ground-truth shape mismatch: expected {(nq, kk)}, got "
            f"{(info['n'], info['dim'])}: {info['path']}"
        )
    rows: list[tuple[int, ...]] = []
    with Path(info["path"]).open("rb") as stream:
        for row in range(nq):
            width = struct.unpack("<i", stream.read(4))[0]
            values = struct.unpack("<" + "i" * width, stream.read(width * 4))
            bad = next((label for label in values if label < 0 or label >= n), None)
            if bad is not None:
                raise FastHNSWError(
                    f"ground-truth label out of range at query {row}: {bad} not in [0,{n})"
                )
            if len(set(values[:k])) != k:
                raise FastHNSWError(f"ground-truth top-{k} contains duplicates at query {row}")
            rows.append(tuple(values[:k]))
    return rows


def recall_at_k(
    result_labels: Sequence[Sequence[int]],
    groundtruth_topk: Sequence[Sequence[int]],
    *,
    k: int,
    n: int | None = None,
) -> float:
    """Existing main-body recall: global top-k set-intersection count/(nq*k).

    Returned results are deliberately not converted to a set: matching duplicate
    occurrences count exactly as the existing C++ evaluator would. A missing/short
    row is a hard evaluator error, matching its assumption of exactly k results.
    """
    if type(k) is not int or k <= 0:
        raise FastHNSWError("recall k must be a positive integer")
    if not groundtruth_topk or len(result_labels) != len(groundtruth_topk):
        raise FastHNSWError(
            f"result/GT query-count mismatch: {len(result_labels)} vs {len(groundtruth_topk)}"
        )
    correct = 0
    for qi, (returned, truth) in enumerate(zip(result_labels, groundtruth_topk)):
        if len(returned) != k or len(truth) != k:
            raise FastHNSWError(
                f"query {qi} must have exactly k={k} returned and GT labels; "
                f"got {len(returned)} and {len(truth)}"
            )
        truth_set = set(truth)
        for label in returned:
            if type(label) is not int:
                raise FastHNSWError(f"returned label at query {qi} is not an integer: {label!r}")
            if n is not None and (label < 0 or label >= n):
                raise FastHNSWError(f"returned label at query {qi} is outside [0,{n}): {label}")
            if label in truth_set:
                correct += 1
    return correct / float(len(groundtruth_topk) * k)


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FastHNSWError(f"malformed JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise FastHNSWError(f"construction record at {path}:{line_no} is not an object")
            records.append(record)
    return records


def select_stock_hnsw_construction(
    results_path: Path, run_key: str, *, construction_kind: str,
) -> dict:
    """Select one canonical stock-HNSW record and verify its immutable index.

    ``construction_kind`` changes only construction-record validation.  Query
    execution, counting, recall, and interpolation are shared unchanged.
    """
    profiles = {
        "fasthnsw": {
            "required": {
                "namespace": "fastkcna-canonical", "builder": "fastkcna-canonical",
                "algorithm": "fasthnsw", "pg_type": 2,
                "canonical_distance_counts_available": True, "exit_status": 0,
            },
            "nested": "canonical_fastkcna_distance_counts",
        },
        "layerwise_nnd_hnsw": {
            "required": {
                "namespace": "layerwise-nnd-hnsw-canonical",
                "builder": "LayerwiseNNDescentHNSW",
                "algorithm": "LayerwiseNNDescentHNSW", "algo": "L-NND-HNSW",
                "canonical": True, "canonical_distance_counts_available": True,
                "exit_status": 0, "M": 16, "initial_diversification_limit": 16,
                "base_degree_cap": 32, "upper_degree_cap": 16,
                "diversification_tie_rule": "stock-nearest-distance-then-descending-internal-id",
                "level_seed": 2024,
            },
            "nested": "canonical_layerwise_distance_counts",
        },
    }
    if construction_kind not in profiles:
        raise FastHNSWError(f"unsupported stock-HNSW construction kind: {construction_kind!r}")
    profile = profiles[construction_kind]
    path = Path(results_path).resolve()
    if not path.is_file():
        raise FastHNSWError(f"canonical construction results file is missing: {path}")
    if not isinstance(run_key, str) or not run_key:
        raise FastHNSWError("an explicit nonempty canonical construction run_key is required")
    matches = [record for record in _load_jsonl(path) if record.get("run_key") == run_key]
    if len(matches) != 1:
        raise FastHNSWError(
            f"expected exactly one construction record with run_key={run_key!r} in {path}; "
            f"found {len(matches)}"
        )
    record = matches[0]
    for field, expected in profile["required"].items():
        if type(record.get(field)) is not type(expected) or record.get(field) != expected:
            raise FastHNSWError(
                f"construction record {field} must be exactly {expected!r}; got {record.get(field)!r}"
            )
    for field in ("build_calc", "merge_calc", "total_calc"):
        if type(record.get(field)) is not int or record[field] < 0:
            raise FastHNSWError(f"construction record {field} must be a nonnegative integer")
    if record["build_calc"] <= 0 or record["merge_calc"] != 0 or record["total_calc"] != record["build_calc"]:
        raise FastHNSWError("canonical construction build/merge/total accounting is inconsistent")
    phases = record.get("distance_counts_by_phase")
    layers = record.get("distance_counts_by_layer")
    if not isinstance(phases, dict) or not phases or not isinstance(layers, dict) or not layers:
        raise FastHNSWError("canonical construction phase/layer distance counts are missing")
    for family, counters in (("phase", phases), ("layer", layers)):
        if any(type(value) is not int or value < 0 for value in counters.values()):
            raise FastHNSWError(f"canonical construction {family} counts must be nonnegative integers")
    if sum(phases.values()) != record["build_calc"] or sum(layers.values()) != record["build_calc"]:
        raise FastHNSWError("canonical construction phase/layer counts do not sum to build_calc")
    if construction_kind == "layerwise_nnd_hnsw" and phases.get("construction_search") != 0:
        raise FastHNSWError("LayerwiseNNDescentHNSW construction_search must be exactly zero")
    canonical = record.get(profile["nested"])
    if canonical is not None:
        if canonical.get("construction_total") != record["build_calc"]:
            raise FastHNSWError("nested canonical construction total disagrees with build_calc")
        if canonical.get("phase_totals") != phases or canonical.get("layer_totals") != layers:
            raise FastHNSWError("nested canonical construction phase/layer counts disagree")
    revision = (record.get("fastkcna") or {}).get("revision")
    if revision != PINNED_FASTKCNA_REVISION:
        raise FastHNSWError(
            f"construction FastKCNA revision mismatch: expected {PINNED_FASTKCNA_REVISION}, got {revision!r}"
        )
    index = _expand_path(record.get("output_index_path", ""), "output_index_path")
    if not index.is_file() or index.stat().st_size <= 0:
        raise FastHNSWError(f"canonical construction index is missing/empty: {index}")
    expected_hash = record.get("output_index_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise FastHNSWError("construction output_index_sha256 must be a lowercase SHA-256")
    observed_hash = sha256_file(index)
    if observed_hash != expected_hash:
        raise FastHNSWError(
            f"canonical construction index SHA-256 mismatch: recorded={expected_hash}, observed={observed_hash}"
        )
    selected = copy.deepcopy(record)
    selected["output_index_path"] = str(index)
    selected["construction_results_path"] = str(path)
    selected["verified_output_index_sha256"] = observed_hash
    return selected


def select_canonical_construction(results_path: Path, run_key: str) -> dict:
    """Backward-compatible strict selector for accepted canonical FastHNSW pg2."""
    return select_stock_hnsw_construction(
        results_path, run_key, construction_kind="fasthnsw",
    )


def validate_construction_dataset(record: Mapping, dataset: Mapping, base_info: Mapping) -> None:
    """Bind the selected canonical construction to the requested live base prefix."""
    expected_name = dataset.get("name")
    n, dim = int(dataset.get("nb", -1)), int(dataset.get("dim", -1))
    if record.get("dataset") != expected_name:
        raise FastHNSWError(
            f"construction dataset mismatch: record={record.get('dataset')!r}, requested={expected_name!r}"
        )
    source = record.get("dataset_source")
    if not isinstance(source, dict):
        raise FastHNSWError("construction dataset_source is missing")
    if (source.get("n"), source.get("dim")) != (n, dim):
        raise FastHNSWError(
            f"construction dataset shape mismatch: record={(source.get('n'), source.get('dim'))}, "
            f"requested={(n, dim)}"
        )
    if (base_info.get("n"), base_info.get("dim")) != (n, dim):
        raise FastHNSWError(
            f"live base shape mismatch: got {(base_info.get('n'), base_info.get('dim'))}, "
            f"expected {(n, dim)}"
        )
    if Path(source.get("path", "")).resolve() != Path(base_info["path"]).resolve():
        raise FastHNSWError(
            f"construction/live base path mismatch: {source.get('path')!r} vs {base_info['path']!r}"
        )
    # The canonical record stores source stat identity (and its conversion sidecar
    # repeats it), but not a source SHA. Refuse a changed live file, then record a
    # fresh SHA in quality provenance.
    for field in ("size", "mtime_ns"):
        if source.get(field) != base_info.get(field):
            raise FastHNSWError(
                f"construction/live base {field} mismatch: {source.get(field)!r} vs {base_info.get(field)!r}"
            )


def parse_query_record(
    stdout: str,
    *,
    expected_index_sha256: str,
    expected_ef: int,
    expected_k: int,
    expected_nq: int,
    expected_n: int,
    expected_dim: int,
    expected_identity_samples: int,
) -> dict:
    """Parse and strictly validate one stock evaluator machine record."""
    payloads = [
        line[len(QUERY_RECORD_PREFIX):]
        for line in stdout.splitlines()
        if line.startswith(QUERY_RECORD_PREFIX)
    ]
    if len(payloads) != 1:
        raise FastHNSWError(
            f"stock FastHNSW evaluator requires exactly one {QUERY_RECORD_PREFIX.strip()!r} "
            f"record; found {len(payloads)}"
        )
    try:
        record = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise FastHNSWError(f"malformed stock FastHNSW query JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise FastHNSWError("stock FastHNSW query payload must be an object")
    required = {
        "schema", "version", "instrumentation", "hnswlib_revision", "metric",
        "query_threading", "index_sha256", "k", "ef_search", "query_count",
        "distance_evaluations_total", "distance_evaluations_mean",
        "index_element_count", "dim", "index_label_permutation_validated",
        "identity_samples_checked", "result_labels",
    }
    missing = sorted(required - set(record))
    if missing:
        raise FastHNSWError(f"stock FastHNSW query fields missing: {missing}")
    if record["schema"] != QUERY_SCHEMA or type(record["version"]) is not int or record["version"] != QUERY_VERSION:
        raise FastHNSWError(
            f"unsupported stock FastHNSW query schema/version: "
            f"{record.get('schema')!r}/{record.get('version')!r}"
        )
    if record["instrumentation"] != QUERY_INSTRUMENTATION:
        raise FastHNSWError(f"unsupported query instrumentation: {record['instrumentation']!r}")
    semantic_identity = {
        "hnswlib_revision": PINNED_HNSWLIB_REVISION,
        "metric": "squared_l2_float32",
        "query_threading": "single",
        "index_label_permutation_validated": True,
    }
    for field, expected in semantic_identity.items():
        if type(record.get(field)) is not type(expected) or record.get(field) != expected:
            raise FastHNSWError(
                f"query record {field} must be exactly {expected!r}; got {record.get(field)!r}"
            )
    exact = {
        "index_sha256": expected_index_sha256, "k": expected_k, "ef_search": expected_ef,
        "query_count": expected_nq, "index_element_count": expected_n, "dim": expected_dim,
    }
    for field, expected in exact.items():
        value = record.get(field)
        if type(expected) is int and type(value) is not int:
            raise FastHNSWError(f"query record {field} must be integer {expected}; got {value!r}")
        if value != expected:
            raise FastHNSWError(f"query record {field} mismatch: expected {expected!r}, got {value!r}")
    total = record["distance_evaluations_total"]
    if type(total) is not int or total < 0:
        raise FastHNSWError("distance_evaluations_total must be a nonnegative integer")
    mean = record["distance_evaluations_mean"]
    if type(mean) not in (int, float) or isinstance(mean, bool) or not math.isfinite(float(mean)) or mean < 0:
        raise FastHNSWError("distance_evaluations_mean must be a finite nonnegative number")
    expected_mean = total / float(expected_nq)
    if not math.isclose(float(mean), expected_mean, rel_tol=1e-12, abs_tol=1e-12):
        raise FastHNSWError(
            f"query distance mean mismatch: emitted={mean}, exact total/nq={expected_mean}"
        )
    samples = record["identity_samples_checked"]
    if type(samples) is not int or samples != expected_identity_samples:
        raise FastHNSWError(
            f"identity_samples_checked mismatch: expected {expected_identity_samples}, got {samples!r}"
        )
    labels = record["result_labels"]
    if not isinstance(labels, list) or len(labels) != expected_nq:
        raise FastHNSWError(
            f"result_labels must have exactly nq={expected_nq} rows; "
            f"got {len(labels) if isinstance(labels, list) else type(labels).__name__}"
        )
    for qi, row in enumerate(labels):
        if not isinstance(row, list) or len(row) != expected_k:
            raise FastHNSWError(f"result_labels query {qi} must contain exactly k={expected_k} labels")
        for label in row:
            if type(label) is not int or label < 0 or label >= expected_n:
                raise FastHNSWError(f"invalid result label at query {qi}: {label!r}")
    checked = dict(record)
    checked["distance_evaluations_mean"] = expected_mean
    return checked


class FastHNSWRunner:
    """Run the repository-owned single-thread stock evaluator once per ef point."""

    def __init__(self, evaluator: Path, workdir: Path):
        self.evaluator = Path(evaluator).resolve()
        if not self.evaluator.is_file() or not os.access(self.evaluator, os.X_OK):
            raise FastHNSWError(f"stock FastHNSW evaluator is missing/not executable: {self.evaluator}")
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.evaluator_sha256 = sha256_file(self.evaluator)

    def command(
        self, *, index: Path, query: Path, base: Path, ef: int, k: int,
        nq: int, dim: int, n: int, index_sha256: str, identity_samples: int,
    ) -> list[str]:
        return [
            str(self.evaluator),
            "--index", str(Path(index).resolve()),
            "--query", str(Path(query).resolve()),
            "--base", str(Path(base).resolve()),
            "--ef", str(ef), "--k", str(k), "--nq", str(nq),
            "--dim", str(dim), "--n", str(n),
            "--identity-samples", str(identity_samples),
            "--index-sha256", index_sha256,
        ]

    def run_point(self, **kwargs) -> dict:
        ef = int(kwargs["ef"])
        command = self.command(**kwargs)
        started = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(self.workdir))
        wall = time.perf_counter() - started
        stdout_path = self.workdir / f"ef{ef}.stdout.log"
        stderr_path = self.workdir / f"ef{ef}.stderr.log"
        stdout_path.write_text(proc.stdout)
        stderr_path.write_text(proc.stderr)
        if proc.returncode != 0:
            raise FastHNSWError(
                f"stock FastHNSW evaluator failed at ef={ef} (exit={proc.returncode}): "
                f"{shlex.join(command)}\nstdout: {stdout_path}\nstderr: {stderr_path}"
            )
        record = parse_query_record(
            proc.stdout,
            expected_index_sha256=kwargs["index_sha256"], expected_ef=ef,
            expected_k=int(kwargs["k"]), expected_nq=int(kwargs["nq"]),
            expected_n=int(kwargs["n"]), expected_dim=int(kwargs["dim"]),
            expected_identity_samples=int(kwargs["identity_samples"]),
        )
        return {
            "machine_record": record, "command": command, "command_shell": shlex.join(command),
            "exit_status": proc.returncode, "wall_seconds": wall,
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        }


def enrich_quality_record(
    construction: Mapping,
    point_results: Sequence[Mapping],
    groundtruth_topk: Sequence[Sequence[int]],
    *,
    k: int,
    kk: int,
    nq: int,
    expected_efs: Sequence[int],
    quality_run_key: str,
    evaluator_metadata: Mapping,
    artifact_metadata: Mapping,
    config_path: Path,
    analysis_dataset: str | None = None,
    quality_namespace: str = QUALITY_NAMESPACE,
    quality_algorithm: str = "stock-hnswlib-fasthnsw",
    quality_algo: str = "FastHNSW",
) -> dict:
    """Copy canonical construction evidence and add independently proven quality."""
    if [point["machine_record"]["ef_search"] for point in point_results] != list(expected_efs):
        raise FastHNSWError("query result ef order/coverage does not match requested sweep")
    n = int(construction["dataset_source"]["n"])
    curve = []
    point_provenance = []
    for point in point_results:
        machine = point["machine_record"]
        recall = recall_at_k(machine["result_labels"], groundtruth_topk, k=k, n=n)
        curve.append({
            "ef": machine["ef_search"], "recall": recall,
            "d_s": machine["distance_evaluations_mean"],
            "query_distance_total": machine["distance_evaluations_total"],
            "query_count": machine["query_count"],
        })
        point_provenance.append({
            "ef": machine["ef_search"],
            "query_distance_total": machine["distance_evaluations_total"],
            "d_s": machine["distance_evaluations_mean"],
            "recall": recall,
            "identity_samples_checked": machine["identity_samples_checked"],
            "hnswlib_revision": machine["hnswlib_revision"],
            "metric": machine["metric"],
            "query_threading": machine["query_threading"],
            "command": point["command"], "command_shell": point["command_shell"],
            "exit_status": point["exit_status"], "wall_seconds": point["wall_seconds"],
            "stdout_path": point["stdout_path"], "stderr_path": point["stderr_path"],
        })
    result = copy.deepcopy(dict(construction))
    construction_namespace = result.get("namespace")
    result.update({
        "namespace": quality_namespace,
        "run_key": quality_run_key,
        "construction_namespace": construction_namespace,
        "construction_run_key": construction.get("run_key"),
        "construction_results_path": construction.get("construction_results_path"),
        "quality_run_key": quality_run_key,
        "quality_evaluation_success": True,
        "quality_algorithm": quality_algorithm,
        "algo": quality_algo,
        "analysis_dataset": analysis_dataset or construction.get("dataset"),
        "k": k, "kk": kk, "nq": nq,
        f"recall@{k}": max((point["recall"] for point in curve), default=None),
        "recall_curve": curve,
        "d_s@0.95": ds_at_recall(curve, 0.95),
        "quality_evaluator": {
            **dict(evaluator_metadata),
            "schema": QUERY_SCHEMA, "version": QUERY_VERSION,
            "instrumentation": QUERY_INSTRUMENTATION,
            "hnswlib_revision": PINNED_HNSWLIB_REVISION,
            "query_threads": 1,
            "recall_definition": "sum of returned-label occurrences belonging to the GT top-k set / (nq*k)",
            "points": point_provenance,
        },
        "quality_artifacts": dict(artifact_metadata),
        "quality_config_path": str(Path(config_path).resolve()),
    })
    return result
