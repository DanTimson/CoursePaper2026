"""Canonical LayerwiseNNDescentHNSW construction adapter.

The C++ builder uses the accepted instrumented FastKCNA KGraph library once for
 each preassigned HNSW layer, then serializes one stock-hnswlib index.  This
module validates its single machine record and keeps construction evidence in a
namespace distinct from raw FastKCNA and FastHNSW results.
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .fastkcna import FastKCNAPaths, inspect_lshkit, sha256_file

BUILD_RECORD_PREFIX = "COURSEPAPER_LAYERWISE_NND "
BUILD_SCHEMA = "coursepaper.layerwise_nnd_hnsw.build"
BUILD_VERSION = 2
BUILD_INSTRUMENTATION = "fastkcna-metric-boundary-layerwise-v2"
BUILD_NAMESPACE = "layerwise-nnd-hnsw-canonical"
QUALITY_NAMESPACE = "layerwise-nnd-hnsw-quality"
PINNED_FASTKCNA_REVISION = "e2f2d79d3de92419e7feea2f1a79d9efc5746f1d"
CANONICAL_PHASES = (
    "knng_candidate", "construction_search", "neighbor_prune",
    "reverse_repair", "other_construction",
)


class LayerwiseNNDError(RuntimeError):
    """Invalid configuration, backend provenance, build, or machine evidence."""


def _expand(value: str | os.PathLike, field: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
        raise LayerwiseNNDError(f"{field} contains an unresolved environment variable: {value!r}")
    return Path(expanded).resolve()


@dataclass(frozen=True)
class LayerwiseNNDParams:
    K: int = 500
    L: int = 500
    S: int = 12
    R: int = 100
    iter: int = 6
    seed: int = 2024
    delta: float = 0.002
    controls: int = 100
    recall: float = 0.98
    M: int = 16
    threads: int = 1

    def validate_canonical(self) -> None:
        expected = type(self)()
        invalid_types = {
            name: value for name, value in asdict(self).items()
            if type(value) is not type(getattr(expected, name))
        }
        if invalid_types:
            raise LayerwiseNNDError(
                f"canonical parameter types are strict (bool/coerced numerics refused): {invalid_types}"
            )
        if any(value <= 0 for name, value in asdict(self).items() if name not in {"delta", "recall"}):
            raise LayerwiseNNDError("integer canonical parameters must be positive")
        if not math.isfinite(self.delta) or not math.isfinite(self.recall):
            raise LayerwiseNNDError("delta/recall must be finite")
        if self != expected:
            changed = {
                name: getattr(self, name) for name in asdict(self)
                if getattr(self, name) != getattr(expected, name)
            }
            raise LayerwiseNNDError(
                "canonical LayerwiseNNDescentHNSW parameters are frozen/untuned; "
                f"refusing overrides: {changed}"
            )

    def command_args(self) -> list[str]:
        return [
            "--K", str(self.K), "--L", str(self.L), "--S", str(self.S),
            "--R", str(self.R), "--iter", str(self.iter), "--seed", str(self.seed),
            "--delta", str(self.delta), "--controls", str(self.controls),
            "--recall", str(self.recall), "--M", str(self.M),
            "--threads", str(self.threads),
        ]

    def metadata(self) -> dict:
        return {
            **asdict(self),
            "base_degree_cap": 2 * self.M,
            "upper_degree_cap": self.M,
            "massq_S": None,
            "massq_S_reason": "FastKCNA BridgeView-only; not reached by pg_type=0 candidate acquisition",
            "tuning_status": "untuned canonical",
        }


@dataclass(frozen=True)
class LayerwiseNNDPaths:
    builder: Path
    fastkcna: FastKCNAPaths
    libkgraph: Path

    @classmethod
    def resolve(cls, config: Mapping | None = None) -> "LayerwiseNNDPaths":
        config = dict(config or {})
        fastkcna = FastKCNAPaths.resolve(config)
        builder = _expand(
            os.environ.get("LAYERWISE_NND_HNSW_BUILDER")
            or config.get("layerwise_builder")
            or "cpp/layerwise_nnd_hnsw_builder",
            "layerwise_builder",
        )
        libkgraph = _expand(
            config.get("libkgraph") or fastkcna.checkout / "code" / "libkgraph.a",
            "libkgraph",
        )
        for field, path, executable in (
            ("layerwise_builder", builder, True), ("libkgraph", libkgraph, False),
        ):
            if not path.is_file() or (executable and not os.access(path, os.X_OK)):
                raise LayerwiseNNDError(f"{field} is missing/not usable: {path}")
        obj = cls(builder=builder, fastkcna=fastkcna, libkgraph=libkgraph)
        if obj.fastkcna.revision() != PINNED_FASTKCNA_REVISION:
            raise LayerwiseNNDError(
                f"FastKCNA revision must be {PINNED_FASTKCNA_REVISION}; "
                f"got {obj.fastkcna.revision()}"
            )
        return obj

    def metadata(self) -> dict:
        return {
            **self.fastkcna.metadata(),
            "libkgraph": str(self.libkgraph),
            "libkgraph_sha256": sha256_file(self.libkgraph),
            "layerwise_builder": str(self.builder),
            "layerwise_builder_sha256": sha256_file(self.builder),
            "accepted_nnd002_patch": None,
            "accepted_nnd002_patch_sha256": "4146a086d95aa2596f67910cc0e70897c0126298bc9be979e68a0f99ec4f27e6",
            "accepted_nnd002_patch_retained_in_coursepaper": False,
            "nnd004_external_patch_required": False,
        }


def _counter(value, field: str) -> int:
    if type(value) is not int or value < 0:
        raise LayerwiseNNDError(f"{field} must be a nonnegative integer")
    return value


def parse_build_record(
    stdout: str, *, expected_n: int, expected_dim: int,
    expected_params: LayerwiseNNDParams, allow_injected_levels: bool = False,
) -> dict:
    payloads = [
        line[len(BUILD_RECORD_PREFIX):] for line in stdout.splitlines()
        if line.startswith(BUILD_RECORD_PREFIX)
    ]
    if len(payloads) != 1:
        raise LayerwiseNNDError(
            f"requires exactly one {BUILD_RECORD_PREFIX.strip()!r} record; found {len(payloads)}"
        )
    try:
        record = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise LayerwiseNNDError(f"malformed layerwise builder JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise LayerwiseNNDError("layerwise builder payload must be an object")
    if (record.get("schema"), record.get("version"), record.get("instrumentation")) != (
        BUILD_SCHEMA, BUILD_VERSION, BUILD_INSTRUMENTATION,
    ):
        raise LayerwiseNNDError("unsupported layerwise builder schema/version/instrumentation")
    exact = {
        "metric": "squared_l2_float32", "construction_threads": expected_params.threads,
        "n": expected_n, "dim": expected_dim, "M": 16,
        "initial_diversification_limit": 16,
        "base_degree_cap": 32, "upper_degree_cap": 16,
        "diversification_tie_rule": "stock-nearest-distance-then-descending-internal-id",
        "level_seed": 2024,
    }
    for field, expected in exact.items():
        if type(record.get(field)) is not type(expected) or record.get(field) != expected:
            raise LayerwiseNNDError(f"builder {field} mismatch: expected {expected!r}, got {record.get(field)!r}")
    level_rule = record.get("level_rule")
    if allow_injected_levels:
        if level_rule not in ("validation-injected-level-vector", "fastkcna-getLevel-stock-hnswlib-equivalent"):
            raise LayerwiseNNDError(f"unsupported validation level_rule: {level_rule!r}")
    elif level_rule != "fastkcna-getLevel-stock-hnswlib-equivalent":
        raise LayerwiseNNDError("canonical builds must use the fixed source-faithful generated hierarchy")
    if record.get("candidate_rng_rule") != "params-seed-mt19937-plus-per-invocation-srand-seed":
        raise LayerwiseNNDError("unsupported/non-independent candidate RNG reset rule")
    max_level = _counter(record.get("max_level"), "max_level")
    arrays = {}
    for field in (
        "layer_occupancies", "candidate_build_invocations", "effective_K",
        "effective_L", "effective_S", "effective_iterations", "actual_iterations",
        "effective_controls", "diagnostic_upstream_n_comps", "initial_selected_max_degree",
        "final_max_degree", "layer_totals",
    ):
        values = record.get(field)
        if not isinstance(values, list) or len(values) != max_level + 1:
            raise LayerwiseNNDError(f"{field} must have one value per actual layer")
        arrays[field] = [_counter(value, f"{field}[{i}]") for i, value in enumerate(values)]
    occupancy = arrays["layer_occupancies"]
    if occupancy[0] != expected_n or any(a < b for a, b in zip(occupancy, occupancy[1:])) or occupancy[-1] < 1:
        raise LayerwiseNNDError("invalid nested layer occupancies")
    if any(value > expected_params.M for value in arrays["initial_selected_max_degree"]):
        raise LayerwiseNNDError("initial diversification exceeded M=16")
    final_caps = [2 * expected_params.M] + [expected_params.M] * max_level
    if any(value > cap for value, cap in zip(arrays["final_max_degree"], final_caps)):
        raise LayerwiseNNDError("final degree exceeded its layer storage capacity")
    node0 = record.get("initial_node0_selected_neighbors")
    if not isinstance(node0, list) or len(node0) != max_level + 1:
        raise LayerwiseNNDError("initial_node0_selected_neighbors must have one list per layer")
    for layer, values in enumerate(node0):
        if not isinstance(values, list) or len(values) > expected_params.M:
            raise LayerwiseNNDError(f"initial_node0_selected_neighbors[{layer}] exceeds M=16")
        checked_values = [_counter(value, f"initial_node0_selected_neighbors[{layer}]") for value in values]
        if len(set(checked_values)) != len(checked_values) or any(
            value == 0 or value >= occupancy[layer] for value in checked_values
        ):
            raise LayerwiseNNDError(f"invalid initial node-0 selection at layer {layer}")
    expected_invocations = [1 if value >= 2 else 0 for value in occupancy]
    if arrays["candidate_build_invocations"] != expected_invocations:
        raise LayerwiseNNDError("each nontrivial layer must have exactly one independent candidate invocation")
    expected_k = [min(expected_params.K, value - 1) if value >= 2 else 0 for value in occupancy]
    expected_l = [min(expected_params.L, value - 1) if value >= 2 else 0 for value in occupancy]
    expected_s = [
        (value - 1 if value <= expected_params.K else min(expected_params.S, value - 1))
        if value >= 2 else 0 for value in occupancy
    ]
    expected_iterations = [
        (0 if value <= expected_params.K else expected_params.iter) if value >= 2 else 0
        for value in occupancy
    ]
    for field, expected in (
        ("effective_K", expected_k), ("effective_L", expected_l),
        ("effective_S", expected_s), ("effective_iterations", expected_iterations),
        ("effective_controls", [min(expected_params.controls, value - 1) if value >= 2 else 0 for value in occupancy]),
    ):
        if arrays[field] != expected:
            raise LayerwiseNNDError(f"{field} disagrees with pinned upstream size clamping")
    if any(actual > effective for actual, effective in zip(
        arrays["actual_iterations"], arrays["effective_iterations"]
    )):
        raise LayerwiseNNDError("actual_iterations exceeds its effective per-layer limit")
    phases = record.get("phase_totals")
    if not isinstance(phases, dict) or tuple(phases) != CANONICAL_PHASES:
        raise LayerwiseNNDError(f"phase_totals must contain ordered phases {CANONICAL_PHASES}")
    checked_phases = {name: _counter(phases[name], f"phase_totals.{name}") for name in CANONICAL_PHASES}
    total = _counter(record.get("construction_total"), "construction_total")
    if checked_phases["construction_search"] != 0:
        raise LayerwiseNNDError("canonical construction_search must be exactly zero")
    if sum(checked_phases.values()) != total or sum(arrays["layer_totals"]) != total:
        raise LayerwiseNNDError("phase/layer additive construction invariants failed")
    candidate = record.get("candidate_parameters")
    expected_candidate = {
        "K": expected_params.K, "L": expected_params.L, "S": expected_params.S,
        "R": expected_params.R, "iter": expected_params.iter, "seed": expected_params.seed,
        "delta": expected_params.delta, "controls": expected_params.controls,
        "recall_stop": expected_params.recall,
    }
    if not isinstance(candidate, dict) or set(candidate) != set(expected_candidate):
        raise LayerwiseNNDError("candidate_parameters fields disagree with the canonical API mapping")
    for field, expected in expected_candidate.items():
        value = candidate[field]
        if isinstance(expected, float):
            if type(value) not in (int, float) or abs(float(value) - expected) > 1e-7:
                raise LayerwiseNNDError(f"candidate parameter {field} mismatch")
        elif type(value) is not int or value != expected:
            raise LayerwiseNNDError(f"candidate parameter {field} mismatch")
    structural = record.get("structural_validation")
    required_structural = {"membership", "no_self_or_duplicate", "degree_caps", "reciprocal"}
    if not isinstance(structural, dict) or set(structural) != required_structural or any(
        structural[field] is not True for field in required_structural
    ):
        raise LayerwiseNNDError("builder structural validation evidence is incomplete")
    entry_internal = _counter(record.get("entry_point_internal"), "entry_point_internal")
    entry_label = _counter(record.get("entry_point_label"), "entry_point_label")
    if entry_internal != 0 or entry_label >= expected_n:
        raise LayerwiseNNDError("invalid deterministic top entry point")
    checked = dict(record)
    checked["phase_totals"] = checked_phases
    checked["layer_totals"] = {str(i): value for i, value in enumerate(arrays["layer_totals"])}
    return checked


class LayerwiseNNDRunner:
    def __init__(self, paths: LayerwiseNNDPaths, workdir: Path):
        self.paths = paths
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

    def command(self, data: Path, output: Path, params: LayerwiseNNDParams, levels_file: Path | None = None) -> list[str]:
        command = [str(self.paths.builder), "--data", str(Path(data).resolve()), "--output", str(Path(output).resolve()), *params.command_args()]
        if levels_file is not None:
            command.extend(["--levels-file", str(Path(levels_file).resolve())])
        return command

    def run(
        self, data: Path, params: LayerwiseNNDParams, run_id: str, *,
        n: int, dim: int, conversion: dict | None = None,
        levels_file: Path | None = None,
    ) -> dict:
        converted = inspect_lshkit(Path(data).resolve())
        if (converted["n"], converted["dim"]) != (n, dim):
            raise LayerwiseNNDError("converted dataset shape mismatch")
        output = self.workdir / f"{run_id}.layerwise-nnd-hnsw"
        stdout_path = self.workdir / f"{run_id}.stdout.log"
        stderr_path = self.workdir / f"{run_id}.stderr.log"
        for stale in (output, stdout_path, stderr_path):
            stale.unlink(missing_ok=True)
        command = self.command(Path(data), output, params, levels_file)
        env = dict(os.environ); env["OMP_NUM_THREADS"] = str(params.threads)
        started = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(self.workdir), env=env)
        wall = time.perf_counter() - started
        stdout_path.write_text(proc.stdout); stderr_path.write_text(proc.stderr)
        if proc.returncode != 0:
            raise LayerwiseNNDError(
                f"LayerwiseNNDescentHNSW build failed (exit={proc.returncode}): {shlex.join(command)}\n"
                f"stdout: {stdout_path}\nstderr: {stderr_path}"
            )
        if not output.is_file() or output.stat().st_size <= 0:
            raise LayerwiseNNDError("builder succeeded without a nonempty stock-HNSW index")
        machine = parse_build_record(
            proc.stdout, expected_n=n, expected_dim=dim, expected_params=params,
            allow_injected_levels=levels_file is not None,
        )
        total = machine["construction_total"]
        provenance = self.paths.metadata()
        return {
            "namespace": BUILD_NAMESPACE if levels_file is None else "layerwise-nnd-hnsw-validation",
            "builder": "LayerwiseNNDescentHNSW",
            "algorithm": "LayerwiseNNDescentHNSW",
            "algo": "L-NND-HNSW",
            "canonical": levels_file is None,
            "canonical_distance_counts_available": True,
            "build_calc": total, "merge_calc": 0, "total_calc": total,
            "distance_counts_by_phase": dict(machine["phase_totals"]),
            "distance_counts_by_layer": dict(machine["layer_totals"]),
            "layer_occupancies": machine["layer_occupancies"],
            "candidate_build_invocations": machine["candidate_build_invocations"],
            "effective_candidate_K": machine["effective_K"],
            "max_level": machine["max_level"],
            "entry_point_internal": machine["entry_point_internal"],
            "entry_point_label": machine["entry_point_label"],
            "M": 16, "initial_diversification_limit": 16,
            "base_degree_cap": 32, "upper_degree_cap": 16,
            "diversification_tie_rule": machine["diversification_tie_rule"],
            "initial_selected_max_degree": machine["initial_selected_max_degree"],
            "final_max_degree": machine["final_max_degree"],
            "level_seed": 2024, "level_rule": machine["level_rule"],
            "candidate_parameters": params.metadata(),
            "counter_schema": {
                "schema": machine["schema"], "version": machine["version"],
                "instrumentation": machine["instrumentation"],
                "metric_boundary": "MatrixOracle l2sqr operator(), after completed DIST_TYPE::apply",
            },
            "canonical_layerwise_distance_counts": machine,
            "command": command, "command_shell": shlex.join(command),
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
            "exit_status": proc.returncode, "wall_seconds": wall, "threads": params.threads,
            "fastkcna": provenance,
            "output_index_path": str(output), "output_index_sha256": sha256_file(output),
            "conversion": conversion,
        }
