"""Generate the project-wide 1M constructor + merge result packet.

Version 2 distinguishes two HNSW construction regimes and two cost questions:

* primary HNSW regime: ef_construction=200
* secondary HNSW regime: ef_construction=32

For recursive merge construction, canonical merge evidence comes from the
existing primary ``partition_bigann1m.jsonl`` matrix (ef_construction=200) and
the secondary ``total_cost_bigann1m_efc32.jsonl`` matrix (ef_construction=32),
both restricted to balanced P=2/4/8/16.  The leaf-build term is independently
validated against the corresponding BUILD_ONLY budget file before composing
end-to-end cost.  Therefore no fitted build model or same-build SIGM
normalization is used.

Layerwise NN-Descent and FastHNSW do not have an HNSW ef_construction parameter;
they are shown as fixed constructor anchors in both regime panels.

Outputs deliberately keep merge-only cost separate from end-to-end construction
cost.  Ambiguous canonical evidence is an error.  A final graph that does not
reach Recall@10=0.95 remains in construction-cost outputs, but has no d_s@0.95
point rather than being extrapolated beyond the measured recall curve.
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
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ngmbench.quality import ds_at_recall

VERSION = 2
TARGET_RECALL = 0.95
N = 1_000_000
PARTITIONS = (2, 4, 8, 16)
REGIMES = ("efc200", "efc32")
REGIME_EFC = {"efc200": 200, "efc32": 32}
MERGE_MATRIX_FILES = {
    "efc200": "partition_bigann1m.jsonl",
    "efc32": "total_cost_bigann1m_efc32.jsonl",
}
BUILD_BUDGET_FILES = {
    "efc200": "build_budget_bigann.jsonl",
    "efc32": "build_budget_bigann_efc32.jsonl",
}
MERGE_ALGOS = ("IGTM", "CGTM", "NGM", "TWO_MERGE")
DISPLAY = {
    "IGTM": "IGTM",
    "CGTM": "CGTM",
    "NGM": "NGM",
    "TWO_MERGE": "HNSWMerger",
}
CANONICAL_PARAMS = {
    "IGTM": {"jump_ef": 5, "local_ef": 7, "next_step_k": 3,
             "next_step_ef": 3, "search_M": 5},
    "CGTM": {"jump_ef": 15, "local_ef": 5, "next_step_k": 3,
             "search_M": 5},
    "NGM": {"search_ef": 10},
    "TWO_MERGE": {"merge_lambda": 4},
}


@dataclass(frozen=True)
class OverallPoint:
    family: str
    method: str
    regime: str
    n: int
    n_parts: int
    ef_construction: int | None
    total_calc: int
    build_calc: int
    merge_calc: int
    d_s_at_095: float | None
    recall_curve: tuple[dict, ...]
    construction_run_key: str | None
    quality_run_key: str | None
    source_file: str
    source_sha256: str
    quality_file: str | None
    quality_sha256: str | None
    index_sha256: str | None = None

    @property
    def max_recall(self) -> float | None:
        if not self.recall_curve:
            return None
        return max(float(p["recall"]) for p in self.recall_curve)

    @property
    def reaches_target(self) -> bool:
        return self.d_s_at_095 is not None


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
        checked.append({"ef": ef, "recall": float(recall), "d_s": float(d_s)})
    checked.sort(key=lambda p: p["ef"])
    return tuple(checked)


def _ds_if_reached(curve: tuple[dict, ...], target: float = TARGET_RECALL) -> float | None:
    if not curve:
        return None
    value = ds_at_recall(list(curve), target)
    return None if value is None else float(value)


def _matches_params(actual: dict, expected: dict) -> bool:
    return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())


def _efc_row(row: dict) -> int:
    value = row.get("ef_construction")
    if value is None:
        value = (row.get("params") or {}).get("ef_construction")
    return 200 if value is None else int(value)


def _candidate_identity(row: dict) -> str:
    return str(row.get("run_key") or json.dumps(row, sort_keys=True))


def _canonical_merge_row(row: dict, algo: str, n_parts: int, efc: int) -> bool:
    if row.get("builder") != "hnswmerger" or row.get("algo") != algo:
        return False
    if row.get("n_parts") != n_parts or _efc_row(row) != efc:
        return False
    if row.get("order") not in (None, "balanced"):
        return False
    if not _matches_params(row.get("params") or {}, CANONICAL_PARAMS[algo]):
        return False
    if algo == "TWO_MERGE" and (row.get("params") or {}).get("merge_lambda_mode", "fixed") != "fixed":
        return False
    return True


def _unique_row(rows: Iterable[dict], predicate, label: str) -> dict:
    selected = {}
    for row in rows:
        if predicate(row):
            selected[_candidate_identity(row)] = row
    if not selected:
        raise ValueError(f"no canonical evidence matched {label}")
    if len(selected) != 1:
        raise ValueError(f"ambiguous canonical evidence for {label}: {sorted(selected)}")
    return next(iter(selected.values()))


def _collect_fixed_constructors(results_dir: Path) -> list[OverallPoint]:
    """Collect efc200 monolithic + Layerwise + FastHNSW from the reviewed constructor packet."""
    from scripts.make_constructor_figures import collect as collect_constructors

    data = collect_constructors(results_dir)
    out = []
    for method in ("Monolithic HNSW", "Layerwise NN-Descent", "FastHNSW pg2"):
        p = data[method]["1m"]
        regime = "efc200" if method == "Monolithic HNSW" else "fixed"
        out.append(OverallPoint(
            family="constructor",
            method=("Monolithic HNSW efc=200" if method == "Monolithic HNSW" else method),
            regime=regime,
            n=p.n,
            n_parts=1,
            ef_construction=(200 if method == "Monolithic HNSW" else None),
            total_calc=p.build_calc,
            build_calc=p.build_calc,
            merge_calc=0,
            d_s_at_095=p.d_s_at_095,
            recall_curve=tuple(p.recall_curve),
            construction_run_key=p.construction_run_key,
            quality_run_key=p.quality_run_key or p.quality_source_run_key,
            source_file=p.build_evidence,
            source_sha256=p.build_evidence_sha256,
            quality_file=p.quality_evidence,
            quality_sha256=p.quality_evidence_sha256,
            index_sha256=p.index_sha256,
        ))
    return out


def _collect_monolithic_efc32(results_dir: Path) -> OverallPoint:
    build_path = results_dir / "build_budget_bigann_efc32.jsonl"
    quality_path = results_dir / "secondary_canonical_bigann1m_efc32.jsonl"
    build_rows = _load_jsonl(build_path)
    quality_rows = _load_jsonl(quality_path)

    build = _unique_row(
        build_rows,
        lambda r: (
            r.get("builder") == "hnswmerger-build-only"
            and r.get("algo") == "BUILD_ONLY"
            and r.get("dataset") == "bigann1m"
            and r.get("n") == N
            and r.get("n_parts") == 1
            and r.get("partition_method") == "range"
            and r.get("m") == 16
            and _efc_row(r) == 32
            and r.get("threads") == 1
        ),
        "monolithic BUILD_ONLY bigann1m efc32",
    )
    quality = _unique_row(
        quality_rows,
        lambda r: (
            r.get("builder") == "hnswmerger"
            and r.get("algo") == "INSERT"
            and r.get("dataset") == "bigann1m"
            and r.get("n") == N
            and r.get("n_parts") == 1
            and r.get("order") in (None, "balanced")
            and _efc_row(r) == 32
        ),
        "monolithic INSERT bigann1m efc32 quality",
    )
    build_calc = build.get("build_calc")
    if type(build_calc) is not int or build_calc <= 0 or build.get("total_calc") != build_calc:
        raise ValueError("monolithic efc32 BUILD_ONLY: invalid build total")
    if quality.get("build_calc") != build_calc:
        raise ValueError("monolithic efc32 BUILD_ONLY/INSERT construction totals disagree")
    curve = _validate_curve(quality.get("recall_curve"), "monolithic efc32 quality")
    return OverallPoint(
        family="constructor",
        method="Monolithic HNSW efc=32",
        regime="efc32",
        n=N,
        n_parts=1,
        ef_construction=32,
        total_calc=build_calc,
        build_calc=build_calc,
        merge_calc=0,
        d_s_at_095=_ds_if_reached(curve),
        recall_curve=curve,
        construction_run_key=str(build.get("run_key") or ""),
        quality_run_key=str(quality.get("run_key") or ""),
        source_file=str(build_path),
        source_sha256=_sha256(build_path),
        quality_file=str(quality_path),
        quality_sha256=_sha256(quality_path),
        index_sha256=quality.get("merged_index_sha256") or quality.get("index_sha256"),
    )


def _canonical_build_budget_row(row: dict, n_parts: int, efc: int) -> bool:
    return (
        row.get("builder") == "hnswmerger-build-only"
        and row.get("algo") == "BUILD_ONLY"
        and row.get("dataset") == "bigann1m"
        and row.get("n") == N
        and row.get("n_parts") == n_parts
        and row.get("partition_method") == "range"
        and row.get("m") == 16
        and _efc_row(row) == efc
        and row.get("threads") == 1
    )


def _collect_merge_matrix(results_dir: Path, regime: str) -> list[OverallPoint]:
    efc = REGIME_EFC[regime]
    merge_path = results_dir / MERGE_MATRIX_FILES[regime]
    budget_path = results_dir / BUILD_BUDGET_FILES[regime]
    rows = _load_jsonl(merge_path)
    budgets = _load_jsonl(budget_path)
    merge_sha = _sha256(merge_path)
    budget_sha = _sha256(budget_path)
    out = []
    for algo in MERGE_ALGOS:
        for p in PARTITIONS:
            row = _unique_row(
                rows,
                lambda r, algo=algo, p=p: _canonical_merge_row(r, algo, p, efc),
                f"{DISPLAY[algo]} P={p} {regime}",
            )
            budget = _unique_row(
                budgets,
                lambda r, p=p: _canonical_build_budget_row(r, p, efc),
                f"BUILD_ONLY P={p} {regime}",
            )
            recorded_build = row.get("build_calc")
            build_calc = budget.get("build_calc")
            merge_calc = row.get("merge_calc")
            recorded_total = row.get("total_calc")
            if type(build_calc) is not int or build_calc <= 0 or budget.get("total_calc") != build_calc:
                raise ValueError(f"BUILD_ONLY P={p} {regime}: invalid build total")
            if recorded_build != build_calc:
                raise ValueError(
                    f"{DISPLAY[algo]} P={p} {regime}: merge-row build_calc {recorded_build} "
                    f"disagrees with independent BUILD_ONLY {build_calc}"
                )
            if type(merge_calc) is not int or merge_calc <= 0:
                raise ValueError(f"{DISPLAY[algo]} P={p} {regime}: invalid merge_calc")
            total_calc = build_calc + merge_calc
            if type(recorded_total) is not int or recorded_total != total_calc:
                raise ValueError(f"{DISPLAY[algo]} P={p} {regime}: invalid measured total_calc")
            raw_curve = row.get("recall_curve")
            curve = _validate_curve(raw_curve, f"{DISPLAY[algo]} P={p} {regime} quality") if raw_curve else tuple()
            out.append(OverallPoint(
                family="merge",
                method=DISPLAY[algo],
                regime=regime,
                n=int(row.get("n") or N),
                n_parts=p,
                ef_construction=efc,
                total_calc=total_calc,
                build_calc=build_calc,
                merge_calc=merge_calc,
                d_s_at_095=_ds_if_reached(curve),
                recall_curve=curve,
                construction_run_key=str(budget.get("run_key") or ""),
                quality_run_key=(str(row.get("run_key") or "") if curve else None),
                source_file=str(budget_path),
                source_sha256=budget_sha,
                quality_file=str(merge_path),
                quality_sha256=merge_sha,
                index_sha256=row.get("merged_index_sha256") or row.get("index_sha256"),
            ))
    return out


def collect(results_dir: Path) -> list[OverallPoint]:
    points = _collect_fixed_constructors(results_dir)
    points.append(_collect_monolithic_efc32(results_dir))
    for regime in REGIMES:
        points.extend(_collect_merge_matrix(results_dir, regime))
    return points


def _anchors(points: list[OverallPoint]) -> dict[str, OverallPoint]:
    return {p.method: p for p in points if p.family == "constructor"}


def _merge_points(points: list[OverallPoint], regime: str, method: str) -> list[OverallPoint]:
    return sorted(
        [p for p in points if p.family == "merge" and p.regime == regime and p.method == method],
        key=lambda p: p.n_parts,
    )


def _save(fig, out: Path, stem: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    for ext in ("png", "pdf"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        generated.append(str(path))
    plt.close(fig)
    return generated


def fig_end_to_end_vs_partitions(points: list[OverallPoint], out: Path) -> list[str]:
    anchors = _anchors(points)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    for ax, regime in zip(axes, REGIMES):
        mono = anchors[f"Monolithic HNSW efc={REGIME_EFC[regime]}"]
        for method in ("HNSWMerger", "IGTM", "CGTM", "NGM"):
            pts = _merge_points(points, regime, method)
            ax.plot([p.n_parts for p in pts], [p.total_calc / mono.total_calc for p in pts], marker="o", label=method)
        ax.axhline(1.0, linestyle="--", linewidth=1.2, label="Monolithic HNSW")
        for name in ("Layerwise NN-Descent", "FastHNSW pg2"):
            p = anchors[name]
            ax.axhline(p.total_calc / mono.total_calc, linestyle=":", linewidth=1.1, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(PARTITIONS, [str(p) for p in PARTITIONS])
        ax.set_xlabel("number of source partitions P")
        ax.set_ylabel("total construction distance / monolithic")
        ax.set_title(f"ef_construction={REGIME_EFC[regime]}")
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("1M end-to-end recursive construction cost")
    return _save(fig, out, "overall_end_to_end_vs_partitions_1m")


TRADEOFF_LABEL_OFFSETS = {
    "HNSWMerger": {"start": (5, -12), "end": (5, -12)},
    "IGTM": {"start": (5, -12), "end": (5, -12)},
    "CGTM": {"start": (5, 6), "end": (5, 6)},
    "NGM": {"start": (5, 6), "end": (5, 6)},
}


def _tradeoff_y_limits(points: list[OverallPoint], regime: str) -> tuple[float, float]:
    """Zoom trade-off panels to monolithic + plottable merge trajectories.

    Fixed constructor anchors are intentionally excluded because FastHNSW would
    otherwise compress the merge comparison into a narrow strip.
    """
    anchors = _anchors(points)
    visible = [anchors[f"Monolithic HNSW efc={REGIME_EFC[regime]}"].total_calc / 1e9]
    visible.extend(
        p.total_calc / 1e9
        for method in ("HNSWMerger", "IGTM", "CGTM", "NGM")
        for p in _merge_points(points, regime, method)
        if p.d_s_at_095 is not None
    )
    lo, hi = min(visible), max(visible)
    span = max(hi - lo, hi * 0.05, 0.1)
    pad = span * 0.07
    return max(0.0, lo - pad), hi + pad


def _tradeoff_unreached_note(points: list[OverallPoint], regime: str) -> str | None:
    failed = []
    for method in ("HNSWMerger", "IGTM", "CGTM", "NGM"):
        for p in _merge_points(points, regime, method):
            if p.d_s_at_095 is None:
                failed.append(f"{method} P{p.n_parts}")
    if not failed:
        return None
    return "Not plotted (<0.95 recall): " + ", ".join(failed)


def fig_tradeoff_by_regime(points: list[OverallPoint], out: Path) -> list[str]:
    anchors = _anchors(points)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    for ax, regime in zip(axes, REGIMES):
        mono = anchors[f"Monolithic HNSW efc={REGIME_EFC[regime]}"]
        if mono.d_s_at_095 is not None:
            ax.scatter(
                [mono.d_s_at_095], [mono.total_calc / 1e9],
                s=75, marker="D", label="Monolithic HNSW",
            )
            ax.annotate(
                f"Monolithic HNSW efc={REGIME_EFC[regime]}",
                (mono.d_s_at_095, mono.total_calc / 1e9),
                xytext=(5, 5), textcoords="offset points", fontsize=8,
            )

        for method in ("HNSWMerger", "IGTM", "CGTM", "NGM"):
            pts = [p for p in _merge_points(points, regime, method) if p.d_s_at_095 is not None]
            if not pts:
                continue
            ax.plot([p.d_s_at_095 for p in pts], [p.total_calc / 1e9 for p in pts], marker="o", label=method)

            endpoints = [pts[0]] if len(pts) == 1 else [pts[0], pts[-1]]
            for i, p in enumerate(endpoints):
                which = "start" if i == 0 else "end"
                ax.annotate(
                    f"P{p.n_parts}",
                    (p.d_s_at_095, p.total_calc / 1e9),
                    xytext=TRADEOFF_LABEL_OFFSETS[method][which],
                    textcoords="offset points", fontsize=7,
                )

        note = _tradeoff_unreached_note(points, regime)
        if note:
            ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, va="bottom")

        ax.set_xlabel("search distance evaluations / query at Recall@10=0.95")
        ax.set_ylabel("total construction distance evaluations (billions)")
        ax.set_title(f"ef_construction={REGIME_EFC[regime]}")
        ax.set_ylim(*_tradeoff_y_limits(points, regime))

    fixed = [anchors["Layerwise NN-Descent"], anchors["FastHNSW pg2"]]
    fixed_text = "; ".join(
        f"{p.method}: {p.total_calc / 1e9:.3f}B construction, d_s@.95={p.d_s_at_095:.0f}"
        for p in fixed if p.d_s_at_095 is not None
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[1].legend(
            handles, labels, loc="upper left", ncol=2, fontsize=8,
            title="Series (both panels)", frameon=True,
        )
    fig.suptitle(
        "1M construction/search trade-off by HNSW construction regime\n"
        f"Fixed constructor anchors (not plotted): {fixed_text}",
        fontsize=11,
    )
    return _save(fig, out, "overall_tradeoff_by_regime_1m")


def fig_search_vs_partitions(points: list[OverallPoint], out: Path) -> list[str]:
    anchors = _anchors(points)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    for ax, regime in zip(axes, REGIMES):
        for method in ("HNSWMerger", "IGTM", "CGTM", "NGM"):
            pts = _merge_points(points, regime, method)
            xs = [p.n_parts for p in pts if p.d_s_at_095 is not None]
            ys = [p.d_s_at_095 for p in pts if p.d_s_at_095 is not None]
            if xs:
                ax.plot(xs, ys, marker="o", label=method)
        for name in (f"Monolithic HNSW efc={REGIME_EFC[regime]}", "Layerwise NN-Descent", "FastHNSW pg2"):
            p = anchors[name]
            if p.d_s_at_095 is not None:
                ax.axhline(p.d_s_at_095, linestyle="--" if "Monolithic" in name else ":", linewidth=1.1, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(PARTITIONS, [str(p) for p in PARTITIONS])
        ax.set_xlabel("number of source partitions P")
        ax.set_ylabel("d_s at Recall@10=0.95")
        ax.set_title(f"ef_construction={REGIME_EFC[regime]}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("1M final-index search cost across recursive merge depth")
    return _save(fig, out, "overall_search_vs_partitions_1m")


def fig_merge_only_vs_partitions(points: list[OverallPoint], out: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    for ax, regime in zip(axes, REGIMES):
        for method in ("HNSWMerger", "IGTM", "CGTM", "NGM"):
            pts = _merge_points(points, regime, method)
            ax.plot([p.n_parts for p in pts], [p.merge_calc / 1e9 for p in pts], marker="o", label=method)
        ax.set_xscale("log", base=2)
        ax.set_xticks(PARTITIONS, [str(p) for p in PARTITIONS])
        ax.set_xlabel("number of source partitions P")
        ax.set_ylabel("merge-phase distance evaluations (billions)")
        ax.set_title(f"ef_construction={REGIME_EFC[regime]}")
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("1M merge-only distance cost (supervisor-paper framing)")
    return _save(fig, out, "overall_merge_only_vs_partitions_1m")


def _monolithic_for_regime(points: list[OverallPoint], regime: str) -> OverallPoint:
    target = f"Monolithic HNSW efc={REGIME_EFC[regime]}"
    matches = [p for p in points if p.method == target]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {target} point")
    return matches[0]


def _write_summary(points: list[OverallPoint], out: Path) -> str:
    path = out / "overall_summary.csv"
    cols = [
        "family", "regime", "method", "n", "n_parts", "ef_construction",
        "build_calc", "merge_calc", "total_calc", "total_per_vector",
        "total_vs_monolithic_same_regime", "quality_available", "reaches_recall_0_95",
        "max_recall", "d_s_at_0_95", "construction_run_key", "quality_run_key",
        "source_file", "quality_file", "index_sha256",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for p in points:
            if p.regime in REGIMES:
                mono = _monolithic_for_regime(points, p.regime)
                fraction = p.total_calc / mono.total_calc
            else:
                fraction = None
            w.writerow({
                "family": p.family,
                "regime": p.regime,
                "method": p.method,
                "n": p.n,
                "n_parts": p.n_parts,
                "ef_construction": "" if p.ef_construction is None else p.ef_construction,
                "build_calc": p.build_calc,
                "merge_calc": p.merge_calc,
                "total_calc": p.total_calc,
                "total_per_vector": f"{p.total_calc / p.n:.12g}",
                "total_vs_monolithic_same_regime": "" if fraction is None else f"{fraction:.12g}",
                "quality_available": bool(p.recall_curve),
                "reaches_recall_0_95": p.reaches_target,
                "max_recall": "" if p.max_recall is None else f"{p.max_recall:.12g}",
                "d_s_at_0_95": "" if p.d_s_at_095 is None else f"{p.d_s_at_095:.12g}",
                "construction_run_key": p.construction_run_key or "",
                "quality_run_key": p.quality_run_key or "",
                "source_file": p.source_file,
                "quality_file": p.quality_file or "",
                "index_sha256": p.index_sha256 or "",
            })
    return str(path)


def _write_quality_curves(points: list[OverallPoint], out: Path) -> str:
    path = out / "overall_quality_curves_1m.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime", "method", "n_parts", "ef", "recall", "d_s"], lineterminator="\n")
        w.writeheader()
        for p in points:
            for q in p.recall_curve:
                w.writerow({
                    "regime": p.regime,
                    "method": p.method,
                    "n_parts": p.n_parts,
                    "ef": q["ef"],
                    "recall": q["recall"],
                    "d_s": q["d_s"],
                })
    return str(path)


def _write_efc32_p2_parity(results_dir: Path, out: Path) -> str:
    """Preserve the dedicated P=2 efc32 parity sweep without guessing supervisor d_s semantics."""
    src = results_dir / "secondary_canonical_bigann1m_efc32.jsonl"
    rows = _load_jsonl(src)
    path = out / "secondary_canonical_p2_efc32.csv"
    with path.open("w", newline="") as f:
        cols = ["algo", "config", "build_calc", "merge_calc", "total_calc", "max_recall", "d_s_at_0_95", "run_key"]
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            if r.get("dataset") != "bigann1m" or _efc_row(r) != 32:
                continue
            params = r.get("params") or {}
            if r.get("algo") == "TWO_MERGE":
                cfg = f"lambda={params.get('merge_lambda')}"
            elif r.get("algo") == "SIGM":
                cfg = f"j={params.get('merge_ef_construction')}"
            elif r.get("algo") == "NGM":
                cfg = f"j={params.get('search_ef')}"
            elif r.get("algo") in {"IGTM", "CGTM"}:
                cfg = f"j={params.get('jump_ef')},l={params.get('local_ef')}"
            else:
                cfg = "-"
            raw_curve = r.get("recall_curve")
            curve = _validate_curve(raw_curve, f"parity {r.get('algo')} {cfg}") if raw_curve else tuple()
            w.writerow({
                "algo": r.get("algo"),
                "config": cfg,
                "build_calc": r.get("build_calc") or 0,
                "merge_calc": r.get("merge_calc") or 0,
                "total_calc": r.get("total_calc") or ((r.get("build_calc") or 0) + (r.get("merge_calc") or 0)),
                "max_recall": "" if not curve else f"{max(x['recall'] for x in curve):.12g}",
                "d_s_at_0_95": "" if _ds_if_reached(curve) is None else f"{_ds_if_reached(curve):.12g}",
                "run_key": r.get("run_key") or "",
            })
    return str(path)


def _write_manifest(points: list[OverallPoint], out: Path, generated: list[str], results_dir: Path) -> str:
    inputs = {}
    selections = []
    for p in points:
        inputs[p.source_file] = p.source_sha256
        if p.quality_file and p.quality_sha256:
            inputs[p.quality_file] = p.quality_sha256
        selections.append({
            "family": p.family,
            "regime": p.regime,
            "method": p.method,
            "n_parts": p.n_parts,
            "ef_construction": p.ef_construction,
            "build_calc": p.build_calc,
            "merge_calc": p.merge_calc,
            "total_calc": p.total_calc,
            "quality_available": bool(p.recall_curve),
            "max_recall": p.max_recall,
            "reaches_recall_0_95": p.reaches_target,
            "d_s_at_0_95": p.d_s_at_095,
            "construction_run_key": p.construction_run_key,
            "quality_run_key": p.quality_run_key,
            "index_sha256": p.index_sha256,
        })
    parity = results_dir / "secondary_canonical_bigann1m_efc32.jsonl"
    inputs[str(parity)] = _sha256(parity)
    manifest = {
        "schema": "coursepaper.overall_figures_manifest",
        "version": VERSION,
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "target_recall": TARGET_RECALL,
        "selection_policy": (
            "end-to-end merge trajectories use measured balanced P=2/4/8/16 merge rows "
            "(primary partition_bigann1m.jsonl; secondary total_cost_bigann1m_efc32.jsonl) "
            "with explicit canonical parameters and efc matched to regime; each recorded leaf-build "
            "sum must exactly match the independent BUILD_ONLY budget for the same P before total cost is composed; "
            "SIGM is excluded from recursive total-cost trajectories because P-leaf charging is not its honest construction path; "
            "Layerwise/FastHNSW are fixed constructor anchors; points not reaching Recall@10=0.95 are retained in cost tables but not extrapolated"
        ),
        "input_sha256": dict(sorted(inputs.items())),
        "selections": sorted(selections, key=lambda r: (r["regime"], r["method"], r["n_parts"])),
        "generated": sorted(generated),
    }
    path = out / "overall_figures_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return str(path)


def generate_from_points(points: list[OverallPoint], out: Path) -> list[str]:
    """Figure/table helper used by tests; does not write provenance manifest."""
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    generated += fig_end_to_end_vs_partitions(points, out)
    generated += fig_tradeoff_by_regime(points, out)
    generated += fig_search_vs_partitions(points, out)
    generated += fig_merge_only_vs_partitions(points, out)
    generated += [_write_summary(points, out), _write_quality_curves(points, out)]
    return generated


def generate(results_dir: Path, out: Path) -> list[str]:
    points = collect(results_dir)
    generated = generate_from_points(points, out)
    generated.append(_write_efc32_p2_parity(results_dir, out))
    manifest = _write_manifest(points, out, generated, results_dir)
    generated.append(manifest)
    return generated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate project-wide 1M constructor+merge figures/tables")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="docs/figures/overall")
    args = ap.parse_args(argv)
    generated = generate(Path(args.results_dir), Path(args.out))
    for path in generated:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
