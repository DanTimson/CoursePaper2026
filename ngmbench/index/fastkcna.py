"""Repository-owned adapter for the external FastKCNA implementation.

FastKCNA remains an external checkout.  This module resolves and fingerprints its
executables, converts fvecs to its lshkit container idempotently, and records an
auditable subprocess result.  FastKCNA's own ``cost``/``scan_rate`` values are
kept in a diagnostic namespace: they are not CoursePaper2026 distance counts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import struct
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional


DIAGNOSTIC_WARNING = (
    "FastKCNA cost, iteration cost, prune scan_rate, search scan_rate, and "
    "analogous upstream counters are diagnostic/noncanonical. Their accounting "
    "semantics have not been reconciled with CoursePaper2026 distance calls; "
    "they must not be used as build_calc, merge_calc, or total_calc."
)


class FastKCNAError(RuntimeError):
    """Explicit external-backend, conversion, or build failure."""


def _expand_path(value: str, field: str, environ: Mapping[str, str]) -> Path:
    # expandvars has no mapping argument, so replace against the supplied mapping
    # first (important for deterministic unit tests), then use normal user expansion.
    expanded = str(value)
    for name in set(re.findall(r"\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))", expanded)):
        key = name[0] or name[1]
        if key in environ:
            expanded = expanded.replace("${" + key + "}", environ[key]).replace("$" + key, environ[key])
    expanded = os.path.expanduser(expanded)
    unresolved = re.search(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)", expanded)
    if unresolved:
        raise FastKCNAError(
            f"FastKCNA {field} contains unresolved environment variable "
            f"{unresolved.group(0)!r}: {value!r}. Set FASTKCNA_ROOT or the "
            "corresponding explicit executable variable; see cpp/FASTKCNA.md."
        )
    return Path(expanded).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class FastKCNAPaths:
    checkout: Path
    build_index: Path
    fvec2lshkit: Path

    @classmethod
    def resolve(
        cls,
        config: Optional[Mapping[str, str]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "FastKCNAPaths":
        config = dict(config or {})
        environ = dict(os.environ if environ is None else environ)
        root_value = config.get("checkout") or environ.get("FASTKCNA_ROOT") or "~/FastKCNA"
        checkout = _expand_path(root_value, "checkout", environ)
        # Explicit executable environment overrides let prepared configs work
        # with nonstandard external build directories without editing JSON.
        build_value = (
            environ.get("FASTKCNA_BUILD_INDEX")
            or config.get("build_index")
            or str(checkout / "code" / "build_index")
        )
        convert_value = (
            environ.get("FASTKCNA_FVEC2LSHKIT")
            or config.get("fvec2lshkit")
            or str(checkout / "code" / "fvec2lshkit")
        )
        obj = cls(
            checkout=checkout,
            build_index=_expand_path(build_value, "build_index", environ),
            fvec2lshkit=_expand_path(convert_value, "fvec2lshkit", environ),
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        if not self.checkout.is_dir():
            raise FastKCNAError(
                f"FastKCNA checkout is missing: {self.checkout}. Clone it outside "
                "CoursePaper2026 and set FASTKCNA_ROOT; see cpp/FASTKCNA.md."
            )
        for field in ("build_index", "fvec2lshkit"):
            path = getattr(self, field)
            if not path.is_file():
                raise FastKCNAError(
                    f"FastKCNA executable {field} is missing: {path}. Build the "
                    "external checkout as documented in cpp/FASTKCNA.md."
                )
            if not os.access(path, os.X_OK):
                raise FastKCNAError(f"FastKCNA executable {field} is not executable: {path}")

    def revision(self) -> str:
        proc = subprocess.run(
            ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        revision = proc.stdout.strip()
        if proc.returncode != 0 or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise FastKCNAError(
                f"Cannot resolve FastKCNA git revision at {self.checkout}: "
                f"exit={proc.returncode}, stderr={proc.stderr.strip()!r}"
            )
        return revision.lower()

    def metadata(self) -> dict:
        return {
            "checkout": str(self.checkout),
            "revision": self.revision(),
            "build_index": str(self.build_index),
            "build_index_sha256": sha256_file(self.build_index),
            "fvec2lshkit": str(self.fvec2lshkit),
            "fvec2lshkit_sha256": sha256_file(self.fvec2lshkit),
        }


@dataclass(frozen=True)
class FastKCNAParams:
    pg_type: int
    K: int
    L: int
    S: int
    R: int
    iter: int
    search_L: int
    search_K: int
    nsg_R: int
    step: int
    loop_i: int
    alpha: float
    tau: float
    nthreads: int
    controls: int
    recall: float
    # Upstream fixed defaults are not accepted as CLI flags but affect the run.
    seed: int = 2024
    delta: float = 0.002
    massq_S: int = 10

    def __post_init__(self) -> None:
        if self.pg_type not in (0, 2):
            raise ValueError("FastKCNA exploratory adapter supports only pg_type=0 or pg_type=2")
        for field in ("K", "L", "S", "R", "iter", "search_L", "search_K", "nsg_R", "step", "loop_i", "nthreads", "controls"):
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"FastKCNA parameter {field} must be positive")
        if self.alpha != 60:
            raise ValueError("CX-NND-001 exploratory FastKCNA configs require alpha=60")
        fixed = {"seed": 2024, "delta": 0.002, "massq_S": 10}
        changed = {name: getattr(self, name) for name, value in fixed.items() if getattr(self, name) != value}
        if changed:
            raise ValueError(
                "FastKCNA does not expose these fixed defaults as CLI flags; "
                f"refusing misleading overrides: {changed}"
            )

    def command_args(self) -> list[str]:
        ordered = (
            "K", "L", "S", "R", "iter", "search_L", "nsg_R", "search_K",
            "step", "loop_i", "alpha", "tau", "nthreads", "controls", "recall", "pg_type",
        )
        values = asdict(self)
        args: list[str] = []
        for name in ordered:
            args.extend(["-" + name, str(values[name])])
        return args

    def complete_metadata(self) -> dict:
        return {
            **asdict(self),
            "fixed_upstream_defaults": ["seed", "delta", "massq_S"],
            "tuning_status": "untuned exploratory",
            "hnswlib_ef_construction_equivalence": None,
        }


def inspect_fvecs(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FastKCNAError(f"FastKCNA input fvecs is missing: {path}")
    size = path.stat().st_size
    if size < 4:
        raise FastKCNAError(f"FastKCNA input fvecs is truncated: {path} ({size} bytes)")
    with path.open("rb") as f:
        raw = f.read(4)
    dim = struct.unpack("<i", raw)[0]
    if dim <= 0:
        raise FastKCNAError(f"FastKCNA input fvecs has invalid dimension {dim}: {path}")
    record_size = 4 + dim * 4
    if size % record_size:
        raise FastKCNAError(
            f"FastKCNA input fvecs size {size} is not divisible by record size "
            f"{record_size} (dim={dim}): {path}"
        )
    n = size // record_size
    if n <= 0:
        raise FastKCNAError(f"FastKCNA input fvecs contains no vectors: {path}")
    st = path.stat()
    return {
        "path": str(path), "size": size, "mtime_ns": st.st_mtime_ns,
        "n": n, "dim": dim,
    }


def inspect_lshkit(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 12:
        raise FastKCNAError(f"FastKCNA lshkit artifact is missing or truncated: {path}")
    with path.open("rb") as f:
        marker, n, dim = struct.unpack("<iii", f.read(12))
    expected = 12 + n * dim * 4
    if marker != 4 or n <= 0 or dim <= 0 or path.stat().st_size != expected:
        raise FastKCNAError(
            f"Invalid FastKCNA lshkit artifact {path}: header=(marker={marker}, "
            f"n={n}, dim={dim}), size={path.stat().st_size}, expected={expected}"
        )
    return {"path": str(path.resolve()), "marker": marker, "n": n, "dim": dim, "size": expected}


def _source_identity(source: dict, converter_sha256: str) -> dict:
    return {
        "path": source["path"], "size": source["size"], "mtime_ns": source["mtime_ns"],
        "n": source["n"], "dim": source["dim"], "converter_sha256": converter_sha256,
    }


def prepare_lshkit(source_path: Path, output_path: Path, paths: FastKCNAPaths) -> dict:
    """Convert once, reuse only if the validated artifact and sidecar still match."""
    source = inspect_fvecs(Path(source_path))
    output = Path(output_path).resolve()
    sidecar = Path(str(output) + ".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = _source_identity(source, sha256_file(paths.fvec2lshkit))
    if output.exists() and sidecar.exists():
        try:
            previous = json.loads(sidecar.read_text())
            converted = inspect_lshkit(output)
            checksum_matches = previous.get("output_sha256") == sha256_file(output)
            if (
                previous.get("identity") == identity
                and checksum_matches
                and (converted["n"], converted["dim"]) == (source["n"], source["dim"])
            ):
                return {**previous, "cached": True, "output": str(output)}
        except (OSError, ValueError, json.JSONDecodeError, FastKCNAError):
            pass

    temp = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    command = [str(paths.fvec2lshkit), source["path"], str(temp)]
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=str(paths.checkout))
    wall = time.perf_counter() - started
    process = {
        "command": command, "command_shell": shlex.join(command),
        "exit_status": proc.returncode, "wall_seconds": wall,
        "stdout": proc.stdout, "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        temp.unlink(missing_ok=True)
        raise FastKCNAError(
            f"FastKCNA conversion failed (exit={proc.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    try:
        converted = inspect_lshkit(temp)
        if (converted["n"], converted["dim"]) != (source["n"], source["dim"]):
            raise FastKCNAError(
                f"FastKCNA conversion shape mismatch: source={(source['n'], source['dim'])}, "
                f"converted={(converted['n'], converted['dim'])}"
            )
        os.replace(temp, output)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    metadata = {
        "identity": identity, "source": source, "output": str(output),
        "output_sha256": sha256_file(output), "process": process, "cached": False,
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def _number(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_diagnostics(stdout: str, log_path: Path) -> dict:
    iteration_cost = [float(x) for x in re.findall(r"\bcost:\s*([0-9.eE+-]+)", stdout)]
    rows = []
    if log_path.is_file():
        with log_path.open(newline="") as f:
            for row in csv.reader(f):
                if row:
                    rows.append({"name": row[0], "values": [_number(x) for x in row[1:] if x != ""]})
    def values(name: str) -> list:
        return [r["values"] for r in rows if r["name"].strip().lower() == name]
    return {
        "noncanonical": True,
        "warning": DIAGNOSTIC_WARNING,
        "iteration_cost_ratios": iteration_cost,
        "prune_scan_rate": values("prune scan_rate"),
        "search_scan_rate": values("search scan_rate"),
        "upstream_log_rows": rows,
    }


class FastKCNARunner:
    def __init__(self, paths: FastKCNAPaths, workdir: Path):
        self.paths = paths
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

    def command(self, data_path: Path, output_path: Path, log_path: Path, params: FastKCNAParams) -> list[str]:
        return [
            str(self.paths.build_index),
            "-data_path", str(Path(data_path).resolve()),
            "-index_path", str(Path(output_path).resolve()),
            "-log_path", str(Path(log_path).resolve()),
            *params.command_args(),
        ]

    def run(self, data_path: Path, params: FastKCNAParams, run_id: str, conversion: Optional[dict] = None) -> dict:
        data = Path(data_path).resolve()
        if not data.is_file():
            raise FastKCNAError(f"FastKCNA converted input is missing: {data}")
        inspect_lshkit(data)
        output = self.workdir / f"{run_id}.fastkcna-index"
        log = self.workdir / f"{run_id}.fastkcna.csv"
        stdout_path = self.workdir / f"{run_id}.stdout.log"
        stderr_path = self.workdir / f"{run_id}.stderr.log"
        for stale in (output, log, stdout_path, stderr_path):
            stale.unlink(missing_ok=True)
        command = self.command(data, output, log, params)
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(params.nthreads)
        started = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(self.paths.checkout), env=env)
        wall = time.perf_counter() - started
        stdout_path.write_text(proc.stdout)
        stderr_path.write_text(proc.stderr)
        if proc.returncode != 0:
            raise FastKCNAError(
                f"FastKCNA build failed (exit={proc.returncode}): {shlex.join(command)}\n"
                f"stdout: {stdout_path}\nstderr: {stderr_path}"
            )
        missing = [str(p) for p in (output, log) if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise FastKCNAError(
                "FastKCNA exited successfully but expected output is missing/empty: " + ", ".join(missing)
            )
        provenance = self.paths.metadata()
        return {
            "builder": "fastkcna-exploratory",
            "algorithm": "raw-knng" if params.pg_type == 0 else "fasthnsw",
            "pg_type": params.pg_type,
            "canonical_distance_counts_available": False,
            "accounting_warning": DIAGNOSTIC_WARNING,
            "command": command,
            "command_shell": shlex.join(command),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "exit_status": proc.returncode,
            "wall_seconds": wall,
            "threads": params.nthreads,
            "fastkcna": provenance,
            "fastkcna_log_path": str(log),
            "output_index_path": str(output),
            "output_index_sha256": sha256_file(output),
            "fastkcna_params": params.complete_metadata(),
            "diagnostic_fastkcna_counters": parse_diagnostics(proc.stdout, log),
            "conversion": conversion,
        }


def check_fasthnsw_compatibility(
    *,
    exps_bin: Path,
    index_path: Path,
    base_path: Path,
    query_path: Path,
    groundtruth_path: Path,
    workdir: Path,
    dim: int,
    nb: int,
    nq: int,
    k: int = 10,
    kk: int = 10,
    efs: tuple[int, ...] = (10, 20),
) -> dict:
    """Try the existing HNSWMerger evaluator's unmodified load-only path.

    An incompatible serialization is evidence, not a FastKCNA build failure, so
    this function always returns the external status/output unless prerequisites
    are missing.  No ad-hoc index conversion is attempted.
    """
    from .hnswmerger import parse_exps

    exps = Path(exps_bin).resolve()
    if not exps.is_file() or not os.access(exps, os.X_OK):
        raise FastKCNAError(f"HNSWMerger evaluator executable is missing/not executable: {exps}")
    required = [Path(index_path), Path(base_path), Path(query_path), Path(groundtruth_path)]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FastKCNAError("Compatibility input is missing: " + ", ".join(missing))
    outdir = Path(workdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = outdir / "fasthnsw-load-only.cfg"
    values = {
        "workload_type": "SIFT1M", "merge_method": "REBUILD",
        "dim": dim, "max_elements": nb, "nb": nb,
        "M": 16, "ef_construction": 200,
        "k": k, "kk": kk, "nq": nq,
        "iterations": 1, "rerun": "false", "save_index": "false",
        "base_filepath": str(Path(base_path).resolve()),
        "query_filepath": str(Path(query_path).resolve()),
        "groundtruth_filepath": str(Path(groundtruth_path).resolve()),
        "index_path": str(Path(index_path).resolve()),
        "save_path": str(outdir),
        "efs_array": ", ".join(str(x) for x in efs),
        "thread": 1,
    }
    cfg.write_text("".join(f"{key} = {value}\n" for key, value in values.items()))
    command = [str(exps), str(cfg)]
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=str(exps.parent))
    wall = time.perf_counter() - started
    stdout_path = outdir / "fasthnsw-load-only.stdout.log"
    stderr_path = outdir / "fasthnsw-load-only.stderr.log"
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    curve = None
    parse_error = None
    if proc.returncode == 0:
        try:
            curve = parse_exps(proc.stdout, expect_method="REBUILD").get("recall_curve")
        except Exception as exc:  # preserve exact smoke evidence rather than masking it
            parse_error = repr(exc)
    compatible = proc.returncode == 0 and parse_error is None and bool(curve)
    return {
        "compatible": compatible,
        "command": command,
        "command_shell": shlex.join(command),
        "config_path": str(cfg),
        "exit_status": proc.returncode,
        "wall_seconds": wall,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "recall_curve": curve,
        "parse_error": parse_error,
        "evidence": (
            "FastHNSW output loaded and produced an existing-evaluator recall curve."
            if compatible else
            "FastHNSW output was not consumable by the existing HNSWMerger load-only evaluator; no conversion attempted."
        ),
    }
