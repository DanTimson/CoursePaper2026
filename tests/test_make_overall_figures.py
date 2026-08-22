import csv
import json
from pathlib import Path

import pytest

from scripts.make_overall_figures import (
    OverallPoint,
    _collect_merge_matrix,
    _collect_monolithic_efc32,
    _tradeoff_unreached_note,
    _tradeoff_y_limits,
    generate_from_points,
)


def _curve(max_recall=0.99, base=100.0):
    return (
        {"ef": 50, "recall": min(0.90, max_recall), "d_s": base},
        {"ef": 100, "recall": min(0.96, max_recall), "d_s": base * 2},
        {"ef": 200, "recall": max_recall, "d_s": base * 3},
    )


def _p(family, method, regime, parts, total, build, merge, ds=200.0, curve=None, efc=None):
    if curve is None:
        curve = _curve()
    return OverallPoint(
        family=family,
        method=method,
        regime=regime,
        n=1_000_000,
        n_parts=parts,
        ef_construction=efc,
        total_calc=total,
        build_calc=build,
        merge_calc=merge,
        d_s_at_095=ds,
        recall_curve=curve,
        construction_run_key=f"{regime}-{method}-{parts}",
        quality_run_key=f"q-{regime}-{method}-{parts}",
        source_file=f"{regime}.jsonl",
        source_sha256=f"sha-{regime}",
        quality_file=f"{regime}.jsonl",
        quality_sha256=f"sha-{regime}",
    )


def _synthetic_points():
    pts = [
        _p("constructor", "Monolithic HNSW efc=200", "efc200", 1, 400, 400, 0, 120.0, efc=200),
        _p("constructor", "Monolithic HNSW efc=32", "efc32", 1, 100, 100, 0, 160.0, efc=32),
        _p("constructor", "Layerwise NN-Descent", "fixed", 1, 330, 330, 0, 140.0),
        _p("constructor", "FastHNSW pg2", "fixed", 1, 980, 980, 0, 119.0),
    ]
    for regime, efc, mono in (("efc200", 200, 400), ("efc32", 32, 100)):
        for mi, method in enumerate(("HNSWMerger", "IGTM", "CGTM", "NGM")):
            for p in (2, 4, 8, 16):
                build = int(mono * (0.95 - 0.02 * (p.bit_length() - 2)))
                merge = (mi + 1) * p * 3
                total = build + merge
                # one point deliberately fails to reach target
                if regime == "efc32" and method == "IGTM" and p == 16:
                    curve = _curve(max_recall=0.94, base=120.0)
                    ds = None
                else:
                    curve = _curve(base=100.0 + mi * 10 + p)
                    ds = 180.0 + mi * 10 + p
                pts.append(_p("merge", method, regime, p, total, build, merge, ds, curve, efc))
    return pts


def test_generate_from_points_writes_regime_packet(tmp_path: Path):
    generated = generate_from_points(_synthetic_points(), tmp_path)
    expected = {
        "overall_end_to_end_vs_partitions_1m.png",
        "overall_tradeoff_by_regime_1m.png",
        "overall_search_vs_partitions_1m.png",
        "overall_merge_only_vs_partitions_1m.png",
        "overall_summary.csv",
        "overall_quality_curves_1m.csv",
    }
    assert expected.issubset({Path(x).name for x in generated})

    rows = list(csv.DictReader((tmp_path / "overall_summary.csv").open()))
    assert len(rows) == 36
    failed = next(r for r in rows if r["regime"] == "efc32" and r["method"] == "IGTM" and r["n_parts"] == "16")
    assert failed["reaches_recall_0_95"] == "False"
    assert failed["d_s_at_0_95"] == ""
    assert failed["max_recall"] == "0.94"


def _merge_row(algo, p, *, efc=32, run_key=None, params=None, build=100, merge=20, max_recall=0.99):
    defaults = {
        "IGTM": {"jump_ef": 5, "local_ef": 7, "next_step_k": 3, "next_step_ef": 3, "search_M": 5},
        "CGTM": {"jump_ef": 15, "local_ef": 5, "next_step_k": 3, "search_M": 5},
        "NGM": {"search_ef": 10},
        "TWO_MERGE": {"merge_lambda": 4, "merge_lambda_mode": "fixed"},
    }
    return {
        "builder": "hnswmerger",
        "algo": algo,
        "dataset": "bigann1m",
        "n": 1_000_000,
        "n_parts": p,
        "order": "balanced",
        "ef_construction": efc,
        "params": params or defaults[algo],
        "build_calc": build,
        "merge_calc": merge,
        "total_calc": build + merge,
        "recall_curve": [
            {"ef": 50, "recall": min(0.90, max_recall), "d_s": 100.0},
            {"ef": 100, "recall": min(0.96, max_recall), "d_s": 200.0},
            {"ef": 200, "recall": max_recall, "d_s": 300.0},
        ],
        "run_key": run_key or f"{algo}-{p}",
    }


def test_collect_merge_matrix_requires_all_balanced_canonical_rows(tmp_path: Path):
    rows = []
    for algo in ("IGTM", "CGTM", "NGM", "TWO_MERGE"):
        for p in (2, 4, 8, 16):
            rows.append(_merge_row(algo, p, max_recall=(0.94 if algo == "IGTM" and p == 16 else 0.99)))
    # cheaper noncanonical NGM row must not be selected
    rows.append(_merge_row("NGM", 2, run_key="cheaper", params={"search_ef": 20}, merge=1))
    path = tmp_path / "total_cost_bigann1m_efc32.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    budgets = [
        {
            "builder": "hnswmerger-build-only", "algo": "BUILD_ONLY", "dataset": "bigann1m",
            "n": 1_000_000, "n_parts": p, "partition_method": "range", "m": 16,
            "ef_construction": 32, "threads": 1, "build_calc": 100, "total_calc": 100,
            "run_key": f"budget32-{p}",
        }
        for p in (2, 4, 8, 16)
    ]
    (tmp_path / "build_budget_bigann_efc32.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in budgets)
    )

    pts = _collect_merge_matrix(tmp_path, "efc32")
    assert len(pts) == 16
    ngm2 = next(p for p in pts if p.method == "NGM" and p.n_parts == 2)
    assert ngm2.merge_calc == 20
    igtm16 = next(p for p in pts if p.method == "IGTM" and p.n_parts == 16)
    assert igtm16.max_recall == pytest.approx(0.94)
    assert igtm16.d_s_at_095 is None


def test_collect_merge_matrix_rejects_ambiguous_canonical_row(tmp_path: Path):
    rows = []
    for algo in ("IGTM", "CGTM", "NGM", "TWO_MERGE"):
        for p in (2, 4, 8, 16):
            rows.append(_merge_row(algo, p))
    rows.append(_merge_row("NGM", 2, run_key="duplicate"))
    (tmp_path / "total_cost_bigann1m_efc32.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    budgets = [
        {
            "builder": "hnswmerger-build-only", "algo": "BUILD_ONLY", "dataset": "bigann1m",
            "n": 1_000_000, "n_parts": p, "partition_method": "range", "m": 16,
            "ef_construction": 32, "threads": 1, "build_calc": 100, "total_calc": 100,
            "run_key": f"budget32-{p}",
        }
        for p in (2, 4, 8, 16)
    ]
    (tmp_path / "build_budget_bigann_efc32.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in budgets)
    )
    with pytest.raises(ValueError, match="ambiguous canonical evidence"):
        _collect_merge_matrix(tmp_path, "efc32")



def test_collect_merge_matrix_primary_uses_partition_rows_and_build_budget(tmp_path: Path):
    rows = []
    for algo in ("IGTM", "CGTM", "NGM", "TWO_MERGE"):
        for p in (2, 4, 8, 16):
            rows.append(_merge_row(algo, p, efc=200, build=100 + p, merge=20 + p))
    (tmp_path / "partition_bigann1m.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    budgets = [
        {
            "builder": "hnswmerger-build-only", "algo": "BUILD_ONLY", "dataset": "bigann1m",
            "n": 1_000_000, "n_parts": p, "partition_method": "range", "m": 16,
            "ef_construction": 200, "threads": 1, "build_calc": 100 + p, "total_calc": 100 + p,
            "run_key": f"budget200-{p}",
        }
        for p in (2, 4, 8, 16)
    ]
    (tmp_path / "build_budget_bigann.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in budgets)
    )

    pts = _collect_merge_matrix(tmp_path, "efc200")
    assert len(pts) == 16
    h16 = next(p for p in pts if p.method == "HNSWMerger" and p.n_parts == 16)
    assert h16.build_calc == 116
    assert h16.merge_calc == 36
    assert h16.total_calc == 152
    assert h16.construction_run_key == "budget200-16"
    assert h16.quality_run_key == "TWO_MERGE-16"
    assert h16.source_file.endswith("build_budget_bigann.jsonl")
    assert h16.quality_file.endswith("partition_bigann1m.jsonl")


def test_collect_merge_matrix_rejects_build_budget_mismatch(tmp_path: Path):
    rows = []
    for algo in ("IGTM", "CGTM", "NGM", "TWO_MERGE"):
        for p in (2, 4, 8, 16):
            rows.append(_merge_row(algo, p, efc=200, build=100, merge=20))
    (tmp_path / "partition_bigann1m.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    budgets = [
        {
            "builder": "hnswmerger-build-only", "algo": "BUILD_ONLY", "dataset": "bigann1m",
            "n": 1_000_000, "n_parts": p, "partition_method": "range", "m": 16,
            "ef_construction": 200, "threads": 1, "build_calc": (101 if p == 2 else 100),
            "total_calc": (101 if p == 2 else 100), "run_key": f"budget200-{p}",
        }
        for p in (2, 4, 8, 16)
    ]
    (tmp_path / "build_budget_bigann.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in budgets)
    )
    with pytest.raises(ValueError, match="disagrees with independent BUILD_ONLY"):
        _collect_merge_matrix(tmp_path, "efc200")

def test_collect_monolithic_efc32_links_build_budget_to_insert_quality(tmp_path: Path):
    build = {
        "builder": "hnswmerger-build-only", "algo": "BUILD_ONLY", "dataset": "bigann1m", "n": 1_000_000,
        "n_parts": 1, "partition_method": "range", "m": 16, "ef_construction": 32, "threads": 1,
        "build_calc": 802, "total_calc": 802, "run_key": "build32",
    }
    qual = {
        "builder": "hnswmerger", "algo": "INSERT", "dataset": "bigann1m", "n": 1_000_000,
        "n_parts": 1, "order": "balanced", "ef_construction": 32, "build_calc": 802, "merge_calc": 0,
        "total_calc": 802, "run_key": "insert32",
        "recall_curve": [
            {"ef": 50, "recall": 0.90, "d_s": 100.0},
            {"ef": 100, "recall": 0.96, "d_s": 200.0},
        ],
    }
    (tmp_path / "build_budget_bigann_efc32.jsonl").write_text(json.dumps(build) + "\n")
    (tmp_path / "secondary_canonical_bigann1m_efc32.jsonl").write_text(json.dumps(qual) + "\n")
    p = _collect_monolithic_efc32(tmp_path)
    assert p.total_calc == 802
    assert p.construction_run_key == "build32"
    assert p.quality_run_key == "insert32"
    assert p.d_s_at_095 == pytest.approx(183.33333333333331)



def test_tradeoff_legend_is_visible_and_includes_monolithic(monkeypatch, tmp_path: Path):
    import scripts.make_overall_figures as mof

    captured = {}

    def capture(fig, out, stem):
        captured["fig"] = fig
        return []

    monkeypatch.setattr(mof, "_save", capture)
    mof.fig_tradeoff_by_regime(_synthetic_points(), tmp_path)

    fig = captured["fig"]
    left, right = fig.axes
    assert left.get_legend() is None
    legend = right.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "Series (both panels)"
    assert [t.get_text() for t in legend.get_texts()] == [
        "Monolithic HNSW", "HNSWMerger", "IGTM", "CGTM", "NGM"
    ]


def test_tradeoff_zoom_excludes_fixed_constructor_anchors():
    from dataclasses import replace

    scale = 10_000_000
    points = [
        replace(
            p,
            total_calc=p.total_calc * scale,
            build_calc=p.build_calc * scale,
            merge_calc=p.merge_calc * scale,
        )
        for p in _synthetic_points()
    ]
    lo200, hi200 = _tradeoff_y_limits(points, "efc200")
    lo32, hi32 = _tradeoff_y_limits(points, "efc32")

    fast = next(p for p in points if p.method == "FastHNSW pg2").total_calc / 1e9
    assert hi200 < fast
    assert hi32 < fast
    assert lo200 > 0
    assert lo32 > 0


def test_tradeoff_unreached_note_names_only_failed_points():
    note = _tradeoff_unreached_note(_synthetic_points(), "efc32")
    assert note == "Not plotted (<0.95 recall): IGTM P16"
    assert _tradeoff_unreached_note(_synthetic_points(), "efc200") is None
