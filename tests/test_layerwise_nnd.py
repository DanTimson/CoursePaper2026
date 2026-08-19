from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

from ngmbench.index.fasthnsw import (
    PINNED_FASTKCNA_REVISION,
    FastHNSWError,
    parse_query_record,
    recall_at_k,
    select_stock_hnsw_construction,
)
from ngmbench.index.layerwise_nnd import (
    BUILD_INSTRUMENTATION,
    BUILD_RECORD_PREFIX,
    BUILD_SCHEMA,
    LayerwiseNNDError,
    LayerwiseNNDParams,
    parse_build_record,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "cpp/layerwise_nnd_hnsw_builder"
VALIDATOR = ROOT / "cpp/layerwise_nnd_hnsw_validate"
EVALUATOR = ROOT / "cpp/fast_hnsw_quality_eval"


def machine_record(**overrides) -> dict:
    record = {
        "schema": BUILD_SCHEMA, "version": 2, "instrumentation": BUILD_INSTRUMENTATION,
        "metric": "squared_l2_float32", "construction_threads": 1,
        "n": 40, "dim": 4, "M": 16, "initial_diversification_limit": 16,
        "base_degree_cap": 32, "upper_degree_cap": 16,
        "diversification_tie_rule": "stock-nearest-distance-then-descending-internal-id",
        "level_seed": 2024, "max_level": 2, "entry_point_internal": 0,
        "entry_point_label": 2, "level_rule": "validation-injected-level-vector",
        "candidate_rng_rule": "params-seed-mt19937-plus-per-invocation-srand-seed",
        "layer_occupancies": [40, 12, 3], "candidate_build_invocations": [1, 1, 1],
        "effective_K": [39, 11, 2], "effective_L": [39, 11, 2],
        "effective_S": [39, 11, 2], "effective_iterations": [0, 0, 0],
        "actual_iterations": [0, 0, 0], "effective_controls": [39, 11, 2],
        "diagnostic_upstream_n_comps": [0, 0, 0],
        "initial_selected_max_degree": [16, 11, 2],
        "final_max_degree": [32, 11, 2],
        "initial_node0_selected_neighbors": [list(range(39, 23, -1)), list(range(1, 12)), [1, 2]],
        "phase_totals": {"knng_candidate": 1698, "construction_search": 0,
                         "neighbor_prune": 100, "reverse_repair": 20,
                         "other_construction": 0},
        "layer_totals": [1800, 12, 6], "construction_total": 1818,
        "candidate_parameters": {"K": 500, "L": 500, "S": 12, "R": 100,
                                 "iter": 6, "seed": 2024, "delta": 0.002,
                                 "controls": 100, "recall_stop": 0.98},
        "structural_validation": {"membership": True, "no_self_or_duplicate": True,
                                  "degree_caps": True, "reciprocal": True},
    }
    record.update(overrides)
    return record


def parse_record(record: dict, *, injected: bool = True) -> dict:
    return parse_build_record(
        BUILD_RECORD_PREFIX + json.dumps(record), expected_n=40, expected_dim=4,
        expected_params=LayerwiseNNDParams(), allow_injected_levels=injected,
    )


def test_frozen_parameters_refuse_coerced_or_changed_values():
    LayerwiseNNDParams().validate_canonical()
    with pytest.raises(LayerwiseNNDError, match="types are strict"):
        LayerwiseNNDParams(threads=True).validate_canonical()
    with pytest.raises(LayerwiseNNDError, match="types are strict"):
        LayerwiseNNDParams(K=500.0).validate_canonical()
    with pytest.raises(LayerwiseNNDError, match="frozen/untuned"):
        LayerwiseNNDParams(K=499).validate_canonical()


def test_build_record_strict_hierarchy_independence_clamping_and_counts():
    checked = parse_record(machine_record())
    assert checked["candidate_build_invocations"] == [1, 1, 1]
    assert checked["initial_diversification_limit"] == 16
    assert checked["initial_selected_max_degree"] == [16, 11, 2]
    assert checked["final_max_degree"] == [32, 11, 2]
    assert checked["initial_node0_selected_neighbors"][0] == list(range(39, 23, -1))
    assert checked["phase_totals"]["construction_search"] == 0
    assert sum(checked["layer_totals"].values()) == checked["construction_total"]
    bad_cases = [
        ({"candidate_build_invocations": [1, 0, 1]}, "independent"),
        ({"initial_diversification_limit": 32}, "initial_diversification_limit"),
        ({"diversification_tie_rule": "ascending-internal-id"}, "diversification_tie_rule"),
        ({"initial_selected_max_degree": [17, 11, 2]}, "initial diversification"),
        ({"effective_K": [38, 11, 2]}, "size clamping"),
        ({"candidate_rng_rule": "shared-state"}, "RNG reset"),
        ({"actual_iterations": [1, 0, 0]}, "exceeds"),
        ({"phase_totals": {"knng_candidate": 1697, "construction_search": 1,
                            "neighbor_prune": 100, "reverse_repair": 20,
                            "other_construction": 0}}, "construction_search"),
        ({"layer_totals": [1800, 12, 5]}, "additive"),
        ({"structural_validation": {"anything": True}}, "structural"),
    ]
    for mutation, match in bad_cases:
        with pytest.raises(LayerwiseNNDError, match=match):
            parse_record(machine_record(**mutation))
    canonical = machine_record(level_rule="fastkcna-getLevel-stock-hnswlib-equivalent")
    parse_record(canonical, injected=False)
    with pytest.raises(LayerwiseNNDError, match="canonical builds"):
        parse_record(machine_record(), injected=False)


def layerwise_construction(tmp_path: Path) -> tuple[Path, dict]:
    index = tmp_path / "tiny.hnsw"; index.write_bytes(b"stock-index-fixture")
    sha = hashlib.sha256(index.read_bytes()).hexdigest()
    phases = {"knng_candidate": 10, "construction_search": 0,
              "neighbor_prune": 3, "reverse_repair": 2, "other_construction": 0}
    layers = {"0": 11, "1": 4}
    record = {
        "namespace": "layerwise-nnd-hnsw-canonical", "builder": "LayerwiseNNDescentHNSW",
        "algorithm": "LayerwiseNNDescentHNSW", "algo": "L-NND-HNSW", "canonical": True,
        "canonical_distance_counts_available": True, "exit_status": 0,
        "M": 16, "initial_diversification_limit": 16,
        "base_degree_cap": 32, "upper_degree_cap": 16,
        "diversification_tie_rule": "stock-nearest-distance-then-descending-internal-id",
        "level_seed": 2024,
        "build_calc": 15, "merge_calc": 0, "total_calc": 15,
        "distance_counts_by_phase": phases, "distance_counts_by_layer": layers,
        "canonical_layerwise_distance_counts": {
            "construction_total": 15, "phase_totals": phases, "layer_totals": layers,
        },
        "fastkcna": {"revision": PINNED_FASTKCNA_REVISION}, "run_key": "layerwise-tiny",
        "dataset": "tiny", "dataset_source": {"n": 40, "dim": 4},
        "output_index_path": str(index), "output_index_sha256": sha,
    }
    path = tmp_path / "layerwise_nnd_hnsw_canonical_tiny.jsonl"
    path.write_text(json.dumps(record) + "\n")
    return path, record


def test_shared_quality_selector_accepts_only_canonical_layerwise_profile(tmp_path):
    path, original = layerwise_construction(tmp_path)
    selected = select_stock_hnsw_construction(
        path, "layerwise-tiny", construction_kind="layerwise_nnd_hnsw",
    )
    assert selected["verified_output_index_sha256"] == original["output_index_sha256"]
    for field, value, match in [
        ("canonical", False, "canonical"),
        ("builder", "fastkcna-canonical", "builder"),
        ("algorithm", "fasthnsw", "algorithm"),
    ]:
        bad = copy.deepcopy(original); bad[field] = value; path.write_text(json.dumps(bad) + "\n")
        with pytest.raises(FastHNSWError, match=match):
            select_stock_hnsw_construction(path, "layerwise-tiny", construction_kind="layerwise_nnd_hnsw")
    bad = copy.deepcopy(original); bad["distance_counts_by_phase"]["construction_search"] = 1
    bad["distance_counts_by_phase"]["knng_candidate"] -= 1
    bad["canonical_layerwise_distance_counts"]["phase_totals"] = bad["distance_counts_by_phase"]
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(FastHNSWError, match="construction_search"):
        select_stock_hnsw_construction(path, "layerwise-tiny", construction_kind="layerwise_nnd_hnsw")


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    n = dim = 40
    # A deterministic one-dimensional chain gives the end-to-end stock search
    # test a connected tiny graph without relying on equal-distance tie cases.
    rows = [[float(i) if j == 0 else 0.0 for j in range(dim)] for i in range(n)]
    base = tmp_path / "base.fvecs"
    data = tmp_path / "base.lshkit"
    with base.open("wb") as fvecs, data.open("wb") as lshkit:
        lshkit.write(struct.pack("<III", 4, n, dim))
        for row in rows:
            packed = struct.pack("<" + "f" * dim, *row)
            fvecs.write(struct.pack("<i", dim)); fvecs.write(packed); lshkit.write(packed)
    levels_values = [2 if i in (2, 5, 9) else 1 if i < 12 else 0 for i in range(n)]
    levels = tmp_path / "levels.txt"; levels.write_text("\n".join(map(str, levels_values)) + "\n")
    query = tmp_path / "query.fvecs"
    with query.open("wb") as stream:
        for row in rows[:5]: stream.write(struct.pack("<i", dim)); stream.write(struct.pack("<" + "f" * dim, *row))
    gt = tmp_path / "gt.ivecs"
    with gt.open("wb") as stream:
        for i in range(5):
            labels = [i] + [j for j in range(n) if j != i][:9]
            stream.write(struct.pack("<i", 10)); stream.write(struct.pack("<10i", *labels))
    return base, data, levels, query, gt


def test_prepared_configs_are_frozen_separate_and_explicitly_overridable():
    for scale, n in (("sift10k", 10_000), ("sift100k", 100_000), ("sift1m", 1_000_000)):
        build = json.loads((ROOT / "config" / f"layerwise_nnd_hnsw_canonical_{scale}.json").read_text())
        quality = json.loads((ROOT / "config" / f"layerwise_nnd_hnsw_quality_{scale}.json").read_text())
        assert build["namespace"] == "layerwise-nnd-hnsw-canonical"
        assert build["tuning_status"] == "untuned canonical"
        assert build["dataset"]["nb"] == n
        assert build["candidate_parameters"] == {
            "K": 500, "L": 500, "S": 12, "R": 100, "iter": 6,
            "seed": 2024, "delta": 0.002, "controls": 100, "recall": 0.98,
            "M": 16, "threads": 1,
        }
        assert "layerwise_nnd_hnsw_canonical" in Path(build["results_path"]).name
        assert quality["namespace"] == "layerwise-nnd-hnsw-quality"
        assert quality["construction"]["results_path"] == build["results_path"]
        assert quality["construction"]["run_key"].startswith("OVERRIDE_")
        assert quality["eval"]["efs_array"] == [10, 50, 100, 200, 400]
        assert quality["eval"]["k"] == 10
        assert "layerwise_nnd_hnsw_quality" in Path(quality["results_path"]).name


def test_persisted_sift10k_smoke_has_additive_build_and_shared_quality_curve():
    build = json.loads((ROOT / "results/layerwise_nnd_hnsw_canonical_sift10k.jsonl").read_text())
    quality = json.loads((ROOT / "results/layerwise_nnd_hnsw_quality_sift10k.jsonl").read_text())
    assert build["build_calc"] == 45_296_135
    assert build["distance_counts_by_phase"]["construction_search"] == 0
    assert sum(build["distance_counts_by_phase"].values()) == build["build_calc"]
    assert sum(build["distance_counts_by_layer"].values()) == build["build_calc"]
    assert build["layer_occupancies"] == [10_000, 617, 41, 3, 1]
    assert build["candidate_build_invocations"] == [1, 1, 1, 1, 0]
    assert quality["namespace"] == "layerwise-nnd-hnsw-quality"
    assert quality["construction_run_key"] == build["run_key"]
    assert [point["ef"] for point in quality["recall_curve"]] == [10, 50, 100, 200, 400]
    assert build["initial_diversification_limit"] == 16
    assert build["initial_selected_max_degree"] == [16, 16, 12, 2, 0]
    assert build["final_max_degree"] == [32, 16, 12, 2, 0]
    assert build["output_index_sha256"] == "206f61574c0126e19ab63cc81edbe11ee3200e6af49c2476d0cd1d4e06f35cf3"
    assert quality["d_s@0.95"] == pytest.approx(491.75407655631676)
    assert all("query_seconds" not in point for point in quality["recall_curve"])


@pytest.mark.skipif(
    not BUILDER.is_file() or not os.access(BUILDER, os.X_OK),
    reason="build targeted C++ layerwise builder first",
)
def test_stock_initial_M_storage_capacity_and_equal_distance_tie(tmp_path):
    n = dim = 40
    # Injected mapping makes source label 2 internal node 0.  Its origin vector
    # has an exact tie to every orthogonal unit vector, so the selected order
    # distinguishes stock's descending-ID pair-priority tie from ascending ID.
    rows = [[0.0 if i == 2 else (1.0 if i == j else 0.0) for j in range(dim)] for i in range(n)]
    data = tmp_path / "tie.lshkit"
    with data.open("wb") as stream:
        stream.write(struct.pack("<III", 4, n, dim))
        for row in rows:
            stream.write(struct.pack("<" + "f" * dim, *row))
    levels_values = [2 if i in (2, 5, 9) else 1 if i < 12 else 0 for i in range(n)]
    levels = tmp_path / "levels.txt"
    levels.write_text("\n".join(map(str, levels_values)) + "\n")
    index = tmp_path / "tie.hnsw"
    proc = subprocess.run(
        [str(BUILDER), "--data", str(data), "--output", str(index),
         "--levels-file", str(levels)], capture_output=True, text=True, check=True,
    )
    record = parse_build_record(
        proc.stdout, expected_n=n, expected_dim=dim,
        expected_params=LayerwiseNNDParams(), allow_injected_levels=True,
    )
    assert record["initial_node0_selected_neighbors"][0] == list(range(39, 23, -1))
    assert record["initial_selected_max_degree"][0] == 16
    assert 16 < record["final_max_degree"][0] <= 32
    assert max(record["final_max_degree"][1:]) <= 16
    assert record["phase_totals"]["reverse_repair"] > 0


@pytest.mark.skipif(
    not all(path.is_file() and os.access(path, os.X_OK) for path in (BUILDER, VALIDATOR, EVALUATOR)),
    reason="build targeted C++ layerwise/evaluator binaries first",
)
def test_real_tiny_hierarchy_count_reset_stock_load_and_query(tmp_path):
    base, data, levels, query, gt = write_inputs(tmp_path)
    records = []
    indexes = []
    for suffix in ("a", "b"):
        index = tmp_path / f"index-{suffix}.hnsw"
        command = [str(BUILDER), "--data", str(data), "--output", str(index),
                   "--levels-file", str(levels)]
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        record = parse_build_record(
            proc.stdout, expected_n=40, expected_dim=40,
            expected_params=LayerwiseNNDParams(), allow_injected_levels=True,
        )
        records.append(record); indexes.append(index)
    assert records[0]["construction_total"] == records[1]["construction_total"]
    assert records[0]["phase_totals"] == records[1]["phase_totals"]
    assert records[0]["layer_totals"] == records[1]["layer_totals"]
    assert records[0]["phase_totals"]["knng_candidate"] == sum(n * (n - 1) for n in (40, 12, 3))
    assert records[0]["phase_totals"]["construction_search"] == 0
    assert max(records[0]["initial_selected_max_degree"]) <= 16
    assert records[0]["final_max_degree"][0] <= 32
    assert max(records[0]["final_max_degree"][1:]) <= 16
    assert hashlib.sha256(indexes[0].read_bytes()).digest() == hashlib.sha256(indexes[1].read_bytes()).digest()

    validation = subprocess.run(
        [str(VALIDATOR), "--index", str(indexes[0]), "--base", str(base),
         "--levels", str(levels), "--n", "40", "--dim", "40"],
        capture_output=True, text=True, check=True,
    )
    assert '"stock_load":true' in validation.stdout
    index_sha = hashlib.sha256(indexes[0].read_bytes()).hexdigest()
    query_proc = subprocess.run(
        [str(EVALUATOR), "--index", str(indexes[0]), "--query", str(query),
         "--base", str(base), "--ef", "50", "--k", "10", "--nq", "5",
         "--dim", "40", "--n", "40", "--identity-samples", "40",
         "--index-sha256", index_sha], capture_output=True, text=True, check=True,
    )
    query_record = parse_query_record(
        query_proc.stdout, expected_index_sha256=index_sha, expected_ef=50,
        expected_k=10, expected_nq=5, expected_n=40, expected_dim=40,
        expected_identity_samples=40,
    )
    with gt.open("rb") as stream:
        truth = []
        for _ in range(5):
            width = struct.unpack("<i", stream.read(4))[0]
            truth.append(struct.unpack("<" + "i" * width, stream.read(4 * width)))
    recall = recall_at_k(query_record["result_labels"], truth, k=10, n=40)
    assert 0.0 <= recall <= 1.0
    assert query_record["distance_evaluations_total"] > 0
    # Query/identity work cannot mutate the immutable construction record.
    assert records[0]["construction_total"] == records[1]["construction_total"]
