import csv
import json
from pathlib import Path

import pytest

from scripts.make_constructor_figures import collect, generate


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _curve(mult=1.0):
    return [
        {"ef": 10, "recall": 0.80, "d_s": 100.0 * mult},
        {"ef": 50, "recall": 0.94, "d_s": 200.0 * mult},
        {"ef": 100, "recall": 0.98, "d_s": 300.0 * mult},
        {"ef": 200, "recall": 0.995, "d_s": 500.0 * mult},
        {"ef": 400, "recall": 0.999, "d_s": 800.0 * mult},
    ]


def _ds095(mult=1.0):
    # Linear interpolation between recall .94 at 200*m and .98 at 300*m.
    return 225.0 * mult


def _layerwise_build(tag, n, run_key, total):
    cand = int(total * 0.66)
    prune = total - cand - 10
    return {
        "namespace": "layerwise-nnd-hnsw-canonical",
        "builder": "LayerwiseNNDescentHNSW",
        "algo": "L-NND-HNSW",
        "canonical": True,
        "dataset": f"sift{tag}",
        "tuning_status": "untuned canonical",
        "canonical_distance_counts_available": True,
        "build_calc": total,
        "total_calc": total,
        "run_key": run_key,
        "output_index_sha256": f"sha-{run_key}",
        "counter_schema": {"instrumentation": "fastkcna-metric-boundary-layerwise-v2"},
        "initial_diversification_limit": 16,
        "base_degree_cap": 32,
        "upper_degree_cap": 16,
        "candidate_parameters": {
            "K": 500, "L": 500, "S": 12, "R": 100, "iter": 6,
            "seed": 2024, "delta": 0.002, "controls": 100,
            "recall": 0.98, "M": 16, "threads": 1,
        },
        "dataset_source": {"n": n},
        "distance_counts_by_phase": {
            "knng_candidate": cand,
            "construction_search": 0,
            "neighbor_prune": prune,
            "reverse_repair": 10,
            "other_construction": 0,
        },
        "distance_counts_by_layer": {"0": total - 1, "1": 1},
        "canonical_layerwise_distance_counts": {"actual_iterations": [6, 0]},
    }


def _layerwise_quality(tag, run_key, total, mult=1.0):
    return {
        "namespace": "layerwise-nnd-hnsw-quality",
        "builder": "LayerwiseNNDescentHNSW",
        "algo": "L-NND-HNSW",
        "canonical": True,
        "analysis_dataset": f"bigann{tag}",
        "construction_namespace": "layerwise-nnd-hnsw-canonical",
        "construction_run_key": run_key,
        "quality_run_key": f"q-{run_key}",
        "quality_evaluation_success": True,
        "build_calc": total,
        "output_index_sha256": f"sha-{run_key}",
        "recall_curve": _curve(mult),
        "d_s@0.95": _ds095(mult),
    }


def _fast_params():
    return {
        "pg_type": 2, "K": 500, "L": 500, "S": 12, "R": 100,
        "iter": 6, "search_L": 80, "search_K": 500, "nsg_R": 16,
        "step": 10, "loop_i": 2, "alpha": 60, "tau": 0,
        "nthreads": 1, "controls": 100, "recall": 0.98,
        "seed": 2024, "delta": 0.002, "massq_S": 10,
    }


def _fasthnsw_build(tag, run_key, total):
    return {
        "builder": "fastkcna-canonical",
        "pg_type": 2,
        "run_key": run_key,
        "dataset": f"sift{tag}",
        "canonical_distance_counts_available": True,
        "counter_schema": {"instrumentation": "fastkcna-canonical-distance-v1"},
        "fastkcna_params": _fast_params(),
        "build_calc": total,
        "output_index_sha256": f"fast-sha-{run_key}",
        "distance_counts_by_phase": {
            "knng_candidate": total // 4,
            "construction_search": total // 4,
            "neighbor_prune": total // 4,
            "reverse_repair": total - 3 * (total // 4),
            "other_construction": 0,
        },
    }


def _fasthnsw_quality(tag, run_key, total, mult=1.0):
    return {
        "namespace": "fasthnsw-quality",
        "builder": "fastkcna-canonical",
        "algo": "FastHNSW",
        "pg_type": 2,
        "analysis_dataset": f"bigann{tag}",
        "construction_namespace": "fastkcna-canonical",
        "construction_run_key": run_key,
        "quality_run_key": f"fq-{run_key}",
        "quality_evaluation_success": True,
        "fastkcna_params": _fast_params(),
        "build_calc": total,
        "recall_curve": _curve(mult),
        "d_s@0.95": _ds095(mult),
        "quality_artifacts": {"index": {"sha256": f"fast-sha-{run_key}"}},
    }


def _populate(tmp_path: Path):
    scales = [("10k", 10_000, 45_000_000), ("100k", 100_000, 380_000_000), ("1m", 1_000_000, 3_300_000_000)]
    for tag, n, total in scales:
        rk = f"lw-{tag}"
        _write_jsonl(tmp_path / f"layerwise_nnd_hnsw_canonical_sift{tag}.jsonl", [_layerwise_build(tag, n, rk, total)])
        _write_jsonl(tmp_path / f"layerwise_nnd_hnsw_quality_sift{tag}.jsonl", [_layerwise_quality(tag, rk, total, 1.1)])

    for tag, n, total in [("100k", 100_000, 700_000_000), ("1m", 1_000_000, 9_800_000_000)]:
        rk = f"fh-{tag}"
        _write_jsonl(tmp_path / f"fastkcna_canonical_sift{tag}.jsonl", [
            {**_fasthnsw_build(tag, f"pg0-{tag}", total // 3), "pg_type": 0, "fastkcna_params": {**_fast_params(), "pg_type": 0}},
            _fasthnsw_build(tag, rk, total),
        ])
        _write_jsonl(tmp_path / f"fasthnsw_quality_sift{tag}.jsonl", [_fasthnsw_quality(tag, rk, total, 0.9)])

    budget = []
    for tag, n, total in [("10k", 10_000, 20_000_000), ("100k", 100_000, 300_000_000), ("1m", 1_000_000, 3_700_000_000)]:
        budget.append({
            "builder": "hnswmerger-build-only", "algo": "BUILD_ONLY",
            "dataset": f"bigann{tag}", "n": n, "n_parts": 1,
            "partition_method": "range", "m": 16, "ef_construction": 200,
            "threads": 1, "run_key": f"mono-{tag}", "build_calc": total,
            "total_calc": total,
        })
    _write_jsonl(tmp_path / "build_budget_bigann.jsonl", budget)
    _write_jsonl(tmp_path / "hnsw_monolithic_quality_bigann1m.jsonl", [{
        "schema": "coursepaper.hnsw_monolithic_quality", "version": 1,
        "dataset": "bigann1m", "source_run_key": "historical-insert-1m",
        "builder": "hnswmerger", "logical_method": "INSERT",
        "quality_dispatch_method": "REBUILD", "quality_rerun": False,
        "quality_save_index": False, "n": 1_000_000, "m": 16,
        "ef_construction": 200, "threads": 1,
        "construction_build_calc": 3_700_000_000,
        "metric": "completed_squared_l2_calls_per_query",
        "index_sha256": "mono-index-sha",
        "recall_curve": _curve(1.0),
        "d_s_at_recall": {"value": _ds095(1.0)},
    }])


def test_collect_cross_checks_explicit_construction_quality_identity(tmp_path):
    _populate(tmp_path)
    data = collect(tmp_path)
    assert data["Layerwise NN-Descent"]["1m"].construction_run_key == "lw-1m"
    assert data["FastHNSW pg2"]["1m"].construction_run_key == "fh-1m"
    assert data["Monolithic HNSW"]["1m"].build_calc == 3_700_000_000
    assert data["Monolithic HNSW"]["1m"].d_s_at_095 == pytest.approx(225.0)


def test_ambiguous_layerwise_build_fails_loudly(tmp_path):
    _populate(tmp_path)
    path = tmp_path / "layerwise_nnd_hnsw_canonical_sift1m.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    duplicate = dict(rows[0])
    duplicate["run_key"] = "different-canonical-run"
    _write_jsonl(path, rows + [duplicate])
    with pytest.raises(ValueError, match="ambiguous canonical evidence"):
        collect(tmp_path)


def test_generate_writes_figures_tables_and_manifest(tmp_path):
    results = tmp_path / "results"
    out = tmp_path / "figures"
    _populate(results)
    generated = generate(results, out)
    expected = {
        "constructor_build_scaling.png", "constructor_build_scaling.pdf",
        "constructor_build_per_vector.png", "constructor_build_per_vector.pdf",
        "layerwise_phase_per_vector.png", "layerwise_phase_per_vector.pdf",
        "constructor_tradeoff_1m.png", "constructor_tradeoff_1m.pdf",
        "recall_vs_ds_1m.png", "recall_vs_ds_1m.pdf",
        "constructor_summary.csv", "layerwise_scaling.csv", "quality_curves_1m.csv",
        "constructor_figures_manifest.json",
    }
    assert expected <= {Path(p).name for p in generated}
    assert expected <= {p.name for p in out.iterdir()}

    manifest = json.loads((out / "constructor_figures_manifest.json").read_text())
    assert manifest["schema"] == "coursepaper.constructor_figures_manifest"
    assert manifest["target_recall"] == 0.95
    assert len(manifest["selections"]) == 8  # 3 mono + 3 layerwise + 2 FastHNSW
    assert manifest["input_sha256"]

    mono_1m = next(
        r for r in manifest["selections"]
        if r["method"] == "Monolithic HNSW" and r["scale"] == "1m"
    )
    assert mono_1m["construction_run_key"] == "mono-1m"
    assert mono_1m["quality_source_run_key"] == "historical-insert-1m"
    assert mono_1m["quality_linkage"] == "historical_source_run_matched_by_config_and_build_calc"

    layerwise_1m = next(
        r for r in manifest["selections"]
        if r["method"] == "Layerwise NN-Descent" and r["scale"] == "1m"
    )
    assert layerwise_1m["quality_linkage"] == "exact_construction_run_key"

    with (out / "constructor_summary.csv").open(newline="") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 8
    assert {r["method"] for r in summary} == {
        "Monolithic HNSW", "Layerwise NN-Descent", "FastHNSW pg2"
    }
