"""Generate canonical constructor-comparison figures and tables.

This script is intentionally separate from the historical merge-strategy figure
pipeline.  It consumes only explicitly named canonical evidence files and fails
loudly when a selected construction/quality identity is ambiguous.

Default evidence:
  * Layerwise NN-Descent HNSW: canonical build + quality at 10K/100K/1M
  * FastHNSW pg2: canonical quality (and its embedded construction record) at
    100K/1M, cross-checked against the canonical construction JSONL
  * Monolithic HNSW: direct BUILD_ONLY budget at 10K/100K/1M plus the separate
    1M quality-only provenance record

Outputs are PNG+PDF figures, CSV tables, and a JSON manifest containing input
SHA-256 hashes and exact selected run identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# Make direct `python scripts/...` execution work from an uninstalled checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ngmbench.quality import ds_at_recall

VERSION = 1
SCALES = (("10k", 10_000), ("100k", 100_000), ("1m", 1_000_000))
FASTHNSW_SCALES = (("100k", 100_000), ("1m", 1_000_000))
TARGET_RECALL = 0.95

LAYERWISE_PARAMS = {
    "K": 500,
    "L": 500,
    "S": 12,
    "R": 100,
    "iter": 6,
    "seed": 2024,
    "delta": 0.002,
    "controls": 100,
    "recall": 0.98,
    "M": 16,
    "threads": 1,
}

FASTHNSW_PARAMS = {
    "pg_type": 2,
    "K": 500,
    "L": 500,
    "S": 12,
    "R": 100,
    "iter": 6,
    "search_L": 80,
    "search_K": 500,
    "nsg_R": 16,
    "step": 10,
    "loop_i": 2,
    "alpha": 60,
    "tau": 0,
    "nthreads": 1,
    "controls": 100,
    "recall": 0.98,
    "seed": 2024,
    "delta": 0.002,
    "massq_S": 10,
}


@dataclass(frozen=True)
class ConstructorPoint:
    method: str
    tag: str
    n: int
    build_calc: int
    construction_run_key: str
    quality_run_key: str | None
    quality_source_run_key: str | None
    d_s_at_095: float | None
    recall_curve: tuple[dict, ...]
    index_sha256: str | None
    build_evidence: str
    quality_evidence: str | None
    build_evidence_sha256: str
    quality_evidence_sha256: str | None
    row: dict

    @property
    def build_per_vector(self) -> float:
        return self.build_calc / self.n


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"required canonical evidence is missing: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_no}")
        rows.append(row)
    if not rows:
        raise ValueError(f"canonical evidence file has no rows: {path}")
    return rows


def _identity(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return json.dumps(row, sort_keys=True)


def _unique(rows: Iterable[dict], predicate: Callable[[dict], bool], label: str,
            identity_keys: tuple[str, ...] = ("run_key",)) -> dict:
    selected: dict[str, dict] = {}
    for row in rows:
        if predicate(row):
            selected[_identity(row, *identity_keys)] = row
    if not selected:
        raise ValueError(f"no canonical evidence matched {label}")
    if len(selected) != 1:
        ids = sorted(selected)
        raise ValueError(f"ambiguous canonical evidence for {label}: {ids}")
    return next(iter(selected.values()))


def _matches_numeric(value, expected, tol=1e-9) -> bool:
    if isinstance(expected, float):
        return isinstance(value, (int, float)) and math.isclose(float(value), expected, rel_tol=0, abs_tol=tol)
    return value == expected


def _matches_params(actual: dict, expected: dict) -> bool:
    return isinstance(actual, dict) and all(
        key in actual and _matches_numeric(actual[key], value)
        for key, value in expected.items()
    )


def _validate_curve(curve, label: str) -> tuple[dict, ...]:
    if not isinstance(curve, list) or not curve:
        raise ValueError(f"{label}: recall_curve must be a non-empty list")
    checked = []
    seen_ef = set()
    for point in curve:
        if not isinstance(point, dict):
            raise ValueError(f"{label}: recall_curve point must be an object")
        ef, recall, d_s = point.get("ef"), point.get("recall"), point.get("d_s")
        if type(ef) is not int or ef in seen_ef:
            raise ValueError(f"{label}: ef values must be unique integers")
        if not isinstance(recall, (int, float)) or not (0 <= recall <= 1):
            raise ValueError(f"{label}: invalid recall at ef={ef}")
        if not isinstance(d_s, (int, float)) or d_s <= 0:
            raise ValueError(f"{label}: invalid d_s at ef={ef}")
        seen_ef.add(ef)
        checked.append(dict(point))
    checked.sort(key=lambda p: p["ef"])
    return tuple(checked)


def _validated_ds(curve: tuple[dict, ...], stored, label: str) -> float:
    value = ds_at_recall(list(curve), TARGET_RECALL)
    if value is None:
        raise ValueError(f"{label}: curve does not bracket recall {TARGET_RECALL}")
    if stored is not None and not math.isclose(float(stored), value, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f"{label}: stored d_s@0.95={stored} disagrees with recomputed {value}")
    return float(value)


def _layerwise_point(results: Path, tag: str, n: int) -> ConstructorPoint:
    build_path = results / f"layerwise_nnd_hnsw_canonical_sift{tag}.jsonl"
    quality_path = results / f"layerwise_nnd_hnsw_quality_sift{tag}.jsonl"
    build_rows = _load_jsonl(build_path)
    quality_rows = _load_jsonl(quality_path)

    def build_ok(r):
        return (
            r.get("namespace") == "layerwise-nnd-hnsw-canonical"
            and r.get("builder") == "LayerwiseNNDescentHNSW"
            and r.get("algo") == "L-NND-HNSW"
            and r.get("canonical") is True
            and r.get("dataset") == f"sift{tag}"
            and r.get("tuning_status") == "untuned canonical"
            and r.get("canonical_distance_counts_available") is True
            and r.get("counter_schema", {}).get("instrumentation") == "fastkcna-metric-boundary-layerwise-v2"
            and r.get("initial_diversification_limit") == 16
            and r.get("base_degree_cap") == 32
            and r.get("upper_degree_cap") == 16
            and _matches_params(r.get("candidate_parameters"), LAYERWISE_PARAMS)
            and r.get("dataset_source", {}).get("n") == n
        )

    build = _unique(build_rows, build_ok, f"Layerwise build {tag}")
    run_key = build.get("run_key")
    if not run_key:
        raise ValueError(f"Layerwise build {tag}: missing run_key")
    build_calc = build.get("build_calc")
    phases = build.get("distance_counts_by_phase")
    if type(build_calc) is not int or build_calc <= 0:
        raise ValueError(f"Layerwise build {tag}: invalid build_calc")
    if not isinstance(phases, dict) or sum(phases.values()) != build_calc:
        raise ValueError(f"Layerwise build {tag}: phase totals do not equal build_calc")
    layer_totals = build.get("distance_counts_by_layer")
    if not isinstance(layer_totals, dict) or sum(layer_totals.values()) != build_calc:
        raise ValueError(f"Layerwise build {tag}: layer totals do not equal build_calc")

    quality = _unique(
        quality_rows,
        lambda r: (
            r.get("namespace") == "layerwise-nnd-hnsw-quality"
            and r.get("builder") == "LayerwiseNNDescentHNSW"
            and r.get("algo") == "L-NND-HNSW"
            and r.get("canonical") is True
            and r.get("analysis_dataset") == f"bigann{tag}"
            and r.get("construction_namespace") == "layerwise-nnd-hnsw-canonical"
            and r.get("construction_run_key") == run_key
            and r.get("quality_evaluation_success") is True
        ),
        f"Layerwise quality {tag} for construction {run_key}",
        ("quality_run_key", "run_key"),
    )
    if quality.get("build_calc") != build_calc:
        raise ValueError(f"Layerwise {tag}: quality/build construction totals disagree")
    if quality.get("output_index_sha256") != build.get("output_index_sha256"):
        raise ValueError(f"Layerwise {tag}: quality/build index SHA disagree")
    curve = _validate_curve(quality.get("recall_curve"), f"Layerwise quality {tag}")
    d_s = _validated_ds(curve, quality.get("d_s@0.95"), f"Layerwise quality {tag}")

    return ConstructorPoint(
        method="Layerwise NN-Descent",
        tag=tag,
        n=n,
        build_calc=build_calc,
        construction_run_key=run_key,
        quality_run_key=quality.get("quality_run_key") or quality.get("run_key"),
        quality_source_run_key=None,
        d_s_at_095=d_s,
        recall_curve=curve,
        index_sha256=build.get("output_index_sha256"),
        build_evidence=str(build_path),
        quality_evidence=str(quality_path),
        build_evidence_sha256=_sha256(build_path),
        quality_evidence_sha256=_sha256(quality_path),
        row=build,
    )


def _fasthnsw_point(results: Path, tag: str, n: int) -> ConstructorPoint:
    build_path = results / f"fastkcna_canonical_sift{tag}.jsonl"
    quality_path = results / f"fasthnsw_quality_sift{tag}.jsonl"
    build_rows = _load_jsonl(build_path)
    quality_rows = _load_jsonl(quality_path)

    quality_candidates = [r for r in quality_rows if (
        r.get("namespace") == "fasthnsw-quality"
        and r.get("builder") == "fastkcna-canonical"
        and r.get("algo") == "FastHNSW"
        and r.get("pg_type") == 2
        and r.get("analysis_dataset") == f"bigann{tag}"
        and r.get("construction_namespace") == "fastkcna-canonical"
        and r.get("quality_evaluation_success") is True
        and _matches_params(r.get("fastkcna_params"), FASTHNSW_PARAMS)
    )]
    quality = _unique(
        quality_candidates,
        lambda r: True,
        f"FastHNSW quality {tag}",
        ("quality_run_key", "run_key"),
    )
    construction_key = quality.get("construction_run_key")
    if not construction_key:
        raise ValueError(f"FastHNSW quality {tag}: missing construction_run_key")

    build = _unique(
        build_rows,
        lambda r: (
            r.get("builder") == "fastkcna-canonical"
            and r.get("pg_type") == 2
            and r.get("run_key") == construction_key
            and r.get("dataset") == f"sift{tag}"
            and r.get("canonical_distance_counts_available") is True
            and r.get("counter_schema", {}).get("instrumentation") == "fastkcna-canonical-distance-v1"
            and _matches_params(r.get("fastkcna_params"), FASTHNSW_PARAMS)
        ),
        f"FastHNSW construction {tag} run_key={construction_key}",
    )
    build_calc = build.get("build_calc")
    if type(build_calc) is not int or build_calc <= 0:
        raise ValueError(f"FastHNSW build {tag}: invalid build_calc")
    phases = build.get("distance_counts_by_phase")
    if not isinstance(phases, dict) or sum(phases.values()) != build_calc:
        raise ValueError(f"FastHNSW build {tag}: phase totals do not equal build_calc")
    if quality.get("build_calc") != build_calc:
        raise ValueError(f"FastHNSW {tag}: quality/build construction totals disagree")
    curve = _validate_curve(quality.get("recall_curve"), f"FastHNSW quality {tag}")
    d_s = _validated_ds(curve, quality.get("d_s@0.95"), f"FastHNSW quality {tag}")
    quality_index_sha = quality.get("quality_artifacts", {}).get("index", {}).get("sha256")
    if quality_index_sha and quality_index_sha != build.get("output_index_sha256"):
        raise ValueError(f"FastHNSW {tag}: quality/build index SHA disagree")

    return ConstructorPoint(
        method="FastHNSW pg2",
        tag=tag,
        n=n,
        build_calc=build_calc,
        construction_run_key=construction_key,
        quality_run_key=quality.get("quality_run_key") or quality.get("run_key"),
        quality_source_run_key=None,
        d_s_at_095=d_s,
        recall_curve=curve,
        index_sha256=build.get("output_index_sha256"),
        build_evidence=str(build_path),
        quality_evidence=str(quality_path),
        build_evidence_sha256=_sha256(build_path),
        quality_evidence_sha256=_sha256(quality_path),
        row=build,
    )


def _monolithic_build_points(results: Path) -> dict[str, ConstructorPoint]:
    budget_path = results / "build_budget_bigann.jsonl"
    rows = _load_jsonl(budget_path)
    out = {}
    for tag, n in SCALES:
        row = _unique(
            rows,
            lambda r, tag=tag, n=n: (
                r.get("builder") == "hnswmerger-build-only"
                and r.get("algo") == "BUILD_ONLY"
                and r.get("dataset") == f"bigann{tag}"
                and r.get("n") == n
                and r.get("n_parts") == 1
                and r.get("partition_method") == "range"
                and r.get("m") == 16
                and r.get("ef_construction") == 200
                and r.get("threads") == 1
            ),
            f"monolithic BUILD_ONLY {tag}",
        )
        build_calc = row.get("build_calc")
        if type(build_calc) is not int or build_calc <= 0 or row.get("total_calc") != build_calc:
            raise ValueError(f"monolithic BUILD_ONLY {tag}: invalid build total")
        out[tag] = ConstructorPoint(
            method="Monolithic HNSW",
            tag=tag,
            n=n,
            build_calc=build_calc,
            construction_run_key=str(row.get("run_key") or ""),
            quality_run_key=None,
            quality_source_run_key=None,
            d_s_at_095=None,
            recall_curve=tuple(),
            index_sha256=None,
            build_evidence=str(budget_path),
            quality_evidence=None,
            build_evidence_sha256=_sha256(budget_path),
            quality_evidence_sha256=None,
            row=row,
        )
    return out


def _attach_monolithic_quality(results: Path, point: ConstructorPoint) -> ConstructorPoint:
    quality_path = results / "hnsw_monolithic_quality_bigann1m.jsonl"
    rows = _load_jsonl(quality_path)
    q = _unique(
        rows,
        lambda r: (
            r.get("schema") == "coursepaper.hnsw_monolithic_quality"
            and r.get("version") == 1
            and r.get("dataset") == "bigann1m"
            and r.get("builder") == "hnswmerger"
            and r.get("logical_method") == "INSERT"
            and r.get("quality_dispatch_method") == "REBUILD"
            and r.get("quality_rerun") is False
            and r.get("quality_save_index") is False
            and r.get("n") == 1_000_000
            and r.get("m") == 16
            and r.get("ef_construction") == 200
            and r.get("threads") == 1
            and r.get("metric") in {
                "completed_squared_l2_calls_per_query",
                "squared_l2_metric_invocations_per_query",
            }
        ),
        "monolithic HNSW 1M quality",
        ("source_run_key",),
    )
    source_run_key = q.get("source_run_key")
    if not source_run_key:
        raise ValueError("monolithic HNSW 1M quality: missing historical source_run_key")
    if q.get("construction_build_calc") != point.build_calc:
        raise ValueError("monolithic 1M quality/build-budget construction totals disagree")
    curve = _validate_curve(q.get("recall_curve"), "monolithic HNSW 1M quality")
    stored = (q.get("d_s_at_recall") or {}).get("value")
    d_s = _validated_ds(curve, stored, "monolithic HNSW 1M quality")
    return ConstructorPoint(
        method=point.method,
        tag=point.tag,
        n=point.n,
        build_calc=point.build_calc,
        construction_run_key=point.construction_run_key,
        quality_run_key=None,
        quality_source_run_key=str(source_run_key),
        d_s_at_095=d_s,
        recall_curve=curve,
        index_sha256=q.get("index_sha256"),
        build_evidence=point.build_evidence,
        quality_evidence=str(quality_path),
        build_evidence_sha256=point.build_evidence_sha256,
        quality_evidence_sha256=_sha256(quality_path),
        row=point.row,
    )


def collect(results_dir: Path):
    layerwise = {tag: _layerwise_point(results_dir, tag, n) for tag, n in SCALES}
    fasthnsw = {tag: _fasthnsw_point(results_dir, tag, n) for tag, n in FASTHNSW_SCALES}
    mono = _monolithic_build_points(results_dir)
    mono["1m"] = _attach_monolithic_quality(results_dir, mono["1m"])
    return {"Monolithic HNSW": mono, "Layerwise NN-Descent": layerwise, "FastHNSW pg2": fasthnsw}


def _save(fig, out: Path, stem: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    for ext in ("png", "pdf"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        generated.append(str(path))
    plt.close(fig)
    return generated


def fig_build_scaling(data, out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, points in data.items():
        ordered = sorted(points.values(), key=lambda p: p.n)
        ax.plot([p.n for p in ordered], [p.build_calc for p in ordered], marker="o", label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dataset size N")
    ax.set_ylabel("construction dataset-distance evaluations")
    ax.set_title("Canonical constructor distance cost vs dataset size")
    ax.legend()
    return _save(fig, out, "constructor_build_scaling")


def fig_build_per_vector(data, out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, points in data.items():
        ordered = sorted(points.values(), key=lambda p: p.n)
        ax.plot([p.n for p in ordered], [p.build_per_vector for p in ordered], marker="o", label=method)
    ax.set_xscale("log")
    ax.set_xlabel("dataset size N")
    ax.set_ylabel("construction distance evaluations / vector")
    ax.set_title("Canonical constructor distance cost per input vector")
    ax.legend()
    return _save(fig, out, "constructor_build_per_vector")


def fig_layerwise_phases(layerwise: dict[str, ConstructorPoint], out: Path) -> list[str]:
    ordered = [layerwise[tag] for tag, _ in SCALES]
    labels = [p.tag.upper() for p in ordered]
    phase_defs = [
        ("knng_candidate", "NN-Descent candidate"),
        ("neighbor_prune", "prune/diversify"),
        ("reverse_repair", "reverse repair"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bottom = [0.0] * len(ordered)
    x = list(range(len(ordered)))
    for key, label in phase_defs:
        vals = [p.row["distance_counts_by_phase"].get(key, 0) / p.n for p in ordered]
        ax.bar(x, vals, bottom=bottom, label=label)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_xticks(x, labels)
    ax.set_ylabel("distance evaluations / vector")
    ax.set_title("Layerwise construction phases per input vector")
    ax.legend()
    return _save(fig, out, "layerwise_phase_per_vector")


def fig_tradeoff_1m(data, out: Path) -> list[str]:
    pts = [data[name]["1m"] for name in ("Monolithic HNSW", "Layerwise NN-Descent", "FastHNSW pg2")]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for p in pts:
        if p.d_s_at_095 is None:
            raise ValueError(f"{p.method}: missing 1M matched-recall d_s")
        ax.scatter([p.d_s_at_095], [p.build_calc / 1e9], s=80)
        ax.annotate(p.method, (p.d_s_at_095, p.build_calc / 1e9), xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("search distance evaluations / query at Recall@10 = 0.95")
    ax.set_ylabel("construction dataset-distance evaluations (billions)")
    ax.set_title("1M construction/search-distance trade-off")
    return _save(fig, out, "constructor_tradeoff_1m")


def fig_recall_vs_ds_1m(data, out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name in ("Monolithic HNSW", "Layerwise NN-Descent", "FastHNSW pg2"):
        p = data[name]["1m"]
        curve = sorted(p.recall_curve, key=lambda x: x["d_s"])
        ax.plot([x["d_s"] for x in curve], [x["recall"] for x in curve], marker="o", label=name)
    ax.axhline(TARGET_RECALL, linestyle="--", linewidth=1)
    ax.set_xlabel("search distance evaluations / query (d_s)")
    ax.set_ylabel("Recall@10")
    ax.set_title("1M search quality vs counted search distance")
    ax.legend()
    return _save(fig, out, "recall_vs_ds_1m")


def _write_constructor_summary(data, out: Path) -> str:
    path = out / "constructor_summary.csv"
    cols = [
        "method", "scale", "n", "build_calc", "build_per_vector", "d_s_at_0_95",
        "construction_run_key", "quality_run_key", "quality_source_run_key", "index_sha256",
        "build_evidence", "quality_evidence",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for method in ("Monolithic HNSW", "Layerwise NN-Descent", "FastHNSW pg2"):
            for p in sorted(data[method].values(), key=lambda x: x.n):
                w.writerow({
                    "method": method,
                    "scale": p.tag,
                    "n": p.n,
                    "build_calc": p.build_calc,
                    "build_per_vector": f"{p.build_per_vector:.12g}",
                    "d_s_at_0_95": "" if p.d_s_at_095 is None else f"{p.d_s_at_095:.12g}",
                    "construction_run_key": p.construction_run_key,
                    "quality_run_key": p.quality_run_key or "",
                    "quality_source_run_key": p.quality_source_run_key or "",
                    "index_sha256": p.index_sha256 or "",
                    "build_evidence": p.build_evidence,
                    "quality_evidence": p.quality_evidence or "",
                })
    return str(path)


def _write_layerwise_scaling(layerwise: dict[str, ConstructorPoint], out: Path) -> str:
    path = out / "layerwise_scaling.csv"
    phase_keys = ["knng_candidate", "construction_search", "neighbor_prune", "reverse_repair", "other_construction"]
    cols = [
        "scale", "n", "build_calc", "build_per_vector",
        "layer0_calc", "layer0_per_vector",
        "layer_occupancies", "effective_candidate_K", "actual_iterations",
    ]
    for key in phase_keys:
        cols.extend([key, f"{key}_per_vector", f"{key}_fraction"])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for tag, _ in SCALES:
            p = layerwise[tag]
            phases = p.row["distance_counts_by_phase"]
            layer0 = p.row["distance_counts_by_layer"].get("0", 0)
            actual_iterations = p.row.get("canonical_layerwise_distance_counts", {}).get("actual_iterations")
            row = {
                "scale": tag,
                "n": p.n,
                "build_calc": p.build_calc,
                "build_per_vector": f"{p.build_per_vector:.12g}",
                "layer0_calc": layer0,
                "layer0_per_vector": f"{layer0 / p.n:.12g}",
                "layer_occupancies": json.dumps(p.row.get("layer_occupancies"), separators=(",", ":")),
                "effective_candidate_K": json.dumps(p.row.get("effective_candidate_K"), separators=(",", ":")),
                "actual_iterations": json.dumps(actual_iterations, separators=(",", ":")),
            }
            for key in phase_keys:
                value = phases.get(key, 0)
                row[key] = value
                row[f"{key}_per_vector"] = f"{value / p.n:.12g}"
                row[f"{key}_fraction"] = f"{value / p.build_calc:.12g}"
            w.writerow(row)
    return str(path)


def _write_quality_curves(data, out: Path) -> str:
    path = out / "quality_curves_1m.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "ef", "recall", "d_s"], lineterminator="\n")
        w.writeheader()
        for name in ("Monolithic HNSW", "Layerwise NN-Descent", "FastHNSW pg2"):
            for point in data[name]["1m"].recall_curve:
                w.writerow({"method": name, "ef": point["ef"], "recall": point["recall"], "d_s": point["d_s"]})
    return str(path)


def _write_manifest(data, out: Path, generated: list[str]) -> str:
    files: dict[str, str] = {}
    selections = []
    for method, points in data.items():
        for p in points.values():
            files[p.build_evidence] = p.build_evidence_sha256
            if p.quality_evidence and p.quality_evidence_sha256:
                files[p.quality_evidence] = p.quality_evidence_sha256
            if p.quality_run_key:
                quality_linkage = "exact_construction_run_key"
            elif p.quality_source_run_key:
                quality_linkage = "historical_source_run_matched_by_config_and_build_calc"
            else:
                quality_linkage = None
            selections.append({
                "method": method,
                "scale": p.tag,
                "n": p.n,
                "build_calc": p.build_calc,
                "construction_run_key": p.construction_run_key,
                "quality_run_key": p.quality_run_key,
                "quality_source_run_key": p.quality_source_run_key,
                "quality_linkage": quality_linkage,
                "index_sha256": p.index_sha256,
                "d_s_at_0_95": p.d_s_at_095,
            })
    lw = data["Layerwise NN-Descent"]
    mono = data["Monolithic HNSW"]
    fast = data["FastHNSW pg2"]
    derived = {
        "layerwise_build_decade_ratio_10k_to_100k": lw["100k"].build_calc / lw["10k"].build_calc,
        "layerwise_build_decade_ratio_100k_to_1m": lw["1m"].build_calc / lw["100k"].build_calc,
        "layerwise_finite_range_loglog_slope_10k_to_1m": math.log(lw["1m"].build_calc / lw["10k"].build_calc) / math.log(100),
        "layerwise_vs_monolithic_1m_build_fraction": lw["1m"].build_calc / mono["1m"].build_calc,
        "layerwise_vs_monolithic_1m_build_saving_fraction": 1 - lw["1m"].build_calc / mono["1m"].build_calc,
        "layerwise_vs_monolithic_1m_ds_fraction": lw["1m"].d_s_at_095 / mono["1m"].d_s_at_095,
        "fasthnsw_vs_monolithic_1m_build_fraction": fast["1m"].build_calc / mono["1m"].build_calc,
        "fasthnsw_vs_monolithic_1m_ds_fraction": fast["1m"].d_s_at_095 / mono["1m"].d_s_at_095,
    }
    manifest = {
        "schema": "coursepaper.constructor_figures_manifest",
        "version": VERSION,
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "target_recall": TARGET_RECALL,
        "selection_policy": (
            "explicit canonical file + frozen semantics; Layerwise/FastHNSW quality uses exact "
            "construction-run linkage; monolithic 1M quality retains its historical source run "
            "and is matched to BUILD_ONLY by frozen config + identical construction count; ambiguity is an error"
        ),
        "input_sha256": dict(sorted(files.items())),
        "selections": sorted(selections, key=lambda r: (r["method"], r["n"])),
        "derived": derived,
        "generated": sorted(generated),
    }
    path = out / "constructor_figures_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return str(path)


def generate(results_dir: Path, out: Path) -> list[str]:
    data = collect(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    generated += fig_build_scaling(data, out)
    generated += fig_build_per_vector(data, out)
    generated += fig_layerwise_phases(data["Layerwise NN-Descent"], out)
    generated += fig_tradeoff_1m(data, out)
    generated += fig_recall_vs_ds_1m(data, out)
    generated += [
        _write_constructor_summary(data, out),
        _write_layerwise_scaling(data["Layerwise NN-Descent"], out),
        _write_quality_curves(data, out),
    ]
    manifest = _write_manifest(data, out, generated)
    generated.append(manifest)
    return generated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate canonical constructor comparison figures/tables")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="docs/figures/constructors")
    args = ap.parse_args(argv)
    generated = generate(Path(args.results_dir), Path(args.out))
    for path in generated:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
