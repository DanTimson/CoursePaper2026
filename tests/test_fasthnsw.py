from __future__ import annotations

import copy
import json
import os
import stat
import struct
from pathlib import Path

import pytest

from ngmbench.cli_fasthnsw import main as cli_main
from ngmbench.index.fasthnsw import (
    PINNED_FASTKCNA_REVISION,
    PINNED_HNSWLIB_REVISION,
    QUERY_INSTRUMENTATION,
    QUERY_RECORD_PREFIX,
    QUERY_SCHEMA,
    FastHNSWError,
    inspect_vecs,
    parse_query_record,
    recall_at_k,
    select_canonical_construction,
)

ROOT = Path(__file__).resolve().parents[1]


def write_fvecs(path: Path, rows: list[list[float]]) -> Path:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(struct.pack("<i", len(row)))
            stream.write(struct.pack("<" + "f" * len(row), *row))
    return path


def write_ivecs(path: Path, rows: list[list[int]]) -> Path:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(struct.pack("<i", len(row)))
            stream.write(struct.pack("<" + "i" * len(row), *row))
    return path


def executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def fixture(tmp_path: Path):
    base = write_fvecs(tmp_path / "base.fvecs", [[float(i), float(i + 1)] for i in range(4)])
    query = write_fvecs(tmp_path / "query.fvecs", [[0.0, 1.0], [1.0, 2.0]])
    gt = write_ivecs(tmp_path / "gt.ivecs", [[0, 1, 2], [1, 2, 3]])
    index = tmp_path / "index.hnsw"; index.write_bytes(b"canonical-stock-index")
    import hashlib
    index_hash = hashlib.sha256(index.read_bytes()).hexdigest()
    base_info = inspect_vecs(base, kind="fvecs")
    record = {
        "namespace": "fastkcna-canonical", "builder": "fastkcna-canonical",
        "algorithm": "fasthnsw", "pg_type": 2,
        "canonical_distance_counts_available": True, "exit_status": 0,
        "build_calc": 15, "merge_calc": 0, "total_calc": 15,
        "distance_counts_by_phase": {"knng_candidate": 5, "construction_search": 4,
                                      "neighbor_prune": 3, "reverse_repair": 2,
                                      "other_construction": 1},
        "distance_counts_by_layer": {"0": 11, "1": 4},
        "fastkcna": {"revision": PINNED_FASTKCNA_REVISION, "build_index_sha256": "a" * 64},
        "run_key": "canonical-pg2", "dataset": "tiny", "dataset_source": base_info,
        "output_index_path": str(index), "output_index_sha256": index_hash,
        "counter_schema": {"schema": "coursepaper.fastkcna.distance_counts", "version": 1},
    }
    construction_results = tmp_path / "fastkcna_canonical_tiny.jsonl"
    construction_results.write_text(json.dumps(record) + "\n")
    evaluator = executable(tmp_path / "fast_hnsw_quality_eval", r'''
import json, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
ef = int(args["--ef"]); nq = int(args["--nq"]); k = int(args["--k"])
record = {
  "schema": "coursepaper.fasthnsw.query", "version": 1,
  "instrumentation": "stock-hnswlib-v0.8.0-counted-l2-v1",
  "hnswlib_revision": "3f3429661187e4c24a490a0f148fc6bc89042b3d",
  "metric": "squared_l2_float32", "query_threading": "single",
  "index_label_permutation_validated": True,
  "index_sha256": args.get("--index-sha256", ""),
  "k": k, "ef_search": ef, "query_count": nq,
  "distance_evaluations_total": ef * nq,
  "distance_evaluations_mean": float(ef),
  "index_element_count": int(args["--n"]), "dim": int(args["--dim"]),
  "identity_samples_checked": int(args.get("--identity-samples", "8")),
  "result_labels": [[0, 1][:k] for _ in range(nq)],
}
print("COURSEPAPER_FASTHNSW_QUERY " + json.dumps(record, separators=(",", ":")))
''')
    return base, query, gt, index, construction_results, evaluator, record


def machine_line(**overrides) -> str:
    record = {
        "schema": QUERY_SCHEMA, "version": 1, "instrumentation": QUERY_INSTRUMENTATION,
        "hnswlib_revision": PINNED_HNSWLIB_REVISION,
        "metric": "squared_l2_float32", "query_threading": "single",
        "index_label_permutation_validated": True,
        "index_sha256": "a" * 64, "k": 2, "ef_search": 4, "query_count": 2,
        "distance_evaluations_total": 10, "distance_evaluations_mean": 5.0,
        "index_element_count": 4, "dim": 2, "identity_samples_checked": 2,
        "result_labels": [[0, 1], [1, 2]],
    }
    record.update(overrides)
    return QUERY_RECORD_PREFIX + json.dumps(record)


def parse(stdout: str):
    return parse_query_record(
        stdout, expected_index_sha256="a" * 64, expected_ef=4,
        expected_k=2, expected_nq=2, expected_n=4, expected_dim=2,
        expected_identity_samples=2,
    )


def test_machine_record_parser_is_strict_and_preserves_exact_total():
    record = parse("human stderr should not be here\n" + machine_line() + "\n")
    assert record["distance_evaluations_total"] == 10
    assert record["distance_evaluations_mean"] == 5.0
    for stdout, match in [
        ("", "exactly one"),
        (machine_line() + "\n" + machine_line(), "exactly one"),
        (QUERY_RECORD_PREFIX + "{bad", "malformed"),
    ]:
        with pytest.raises(FastHNSWError, match=match):
            parse(stdout)
    mutations = [
        ({"schema": "wrong"}, "schema/version"),
        ({"hnswlib_revision": "wrong"}, "hnswlib_revision"),
        ({"metric": "inner_product"}, "metric"),
        ({"query_threading": "parallel"}, "query_threading"),
        ({"index_label_permutation_validated": False}, "index_label_permutation"),
        ({"ef_search": 5}, "ef_search mismatch"),
        ({"distance_evaluations_total": True}, "nonnegative integer"),
        ({"distance_evaluations_mean": 4.0}, "mean mismatch"),
        ({"result_labels": [[0, 1]]}, "exactly nq"),
        ({"result_labels": [[0], [1, 2]]}, "exactly k"),
        ({"result_labels": [[0, 4], [1, 2]]}, "invalid result label"),
    ]
    base = json.loads(machine_line().split(" ", 1)[1])
    for change, match in mutations:
        bad = dict(base); bad.update(change)
        with pytest.raises(FastHNSWError, match=match):
            parse(QUERY_RECORD_PREFIX + json.dumps(bad))


def test_recall_matches_main_body_duplicate_and_missing_semantics():
    gt = [(0, 1), (1, 2)]
    assert recall_at_k([[0, 1], [1, 3]], gt, k=2, n=4) == 0.75
    # Existing evaluator counts each returned matching occurrence; it does not
    # deduplicate the returned side.
    assert recall_at_k([[0, 0], [1, 1]], gt, k=2, n=4) == 1.0
    with pytest.raises(FastHNSWError, match="exactly k"):
        recall_at_k([[0], [1, 2]], gt, k=2, n=4)
    with pytest.raises(FastHNSWError, match="outside"):
        recall_at_k([[0, 5], [1, 2]], gt, k=2, n=4)


def test_vec_shape_checks_every_row_header(tmp_path):
    path = write_fvecs(tmp_path / "x.fvecs", [[0.0, 1.0], [2.0, 3.0]])
    assert inspect_vecs(path, kind="fvecs")["n"] == 2
    data = bytearray(path.read_bytes())
    struct.pack_into("<i", data, 12, 3)  # second record header
    path.write_bytes(data)
    with pytest.raises(FastHNSWError, match="row 1 header mismatch"):
        inspect_vecs(path, kind="fvecs")


def test_canonical_selector_requires_pg2_success_and_exact_index_hash(tmp_path):
    *_, results, evaluator, original = fixture(tmp_path)
    selected = select_canonical_construction(results, "canonical-pg2")
    assert selected["verified_output_index_sha256"] == original["output_index_sha256"]
    assert selected["construction_results_path"] == str(results.resolve())
    for field, value, match in [
        ("pg_type", 0, "pg_type"),
        ("canonical_distance_counts_available", False, "canonical_distance"),
        ("exit_status", 1, "exit_status"),
    ]:
        bad = copy.deepcopy(original); bad[field] = value
        results.write_text(json.dumps(bad) + "\n")
        with pytest.raises(FastHNSWError, match=match):
            select_canonical_construction(results, "canonical-pg2")
    results.write_text(json.dumps(original) + "\n")
    Path(original["output_index_path"]).write_bytes(b"tampered")
    with pytest.raises(FastHNSWError, match="SHA-256 mismatch"):
        select_canonical_construction(results, "canonical-pg2")


def test_cli_enriches_separate_record_without_rebuilding_or_mutating_source(tmp_path):
    base, query, gt, index, construction_results, evaluator, original = fixture(tmp_path)
    source_before = construction_results.read_bytes()
    results_path = tmp_path / "results/fasthnsw_quality_tiny.jsonl"
    config = {
        "namespace": "fasthnsw-quality",
        "binaries": {"evaluator": str(evaluator)},
        "construction": {"results_path": str(construction_results), "run_key": "canonical-pg2"},
        "dataset": {"name": "tiny", "dim": 2, "nb": 4, "base": str(base),
                    "query": str(query), "groundtruth": str(gt)},
        "eval": {"k": 2, "kk": 3, "nq": 2, "efs_array": [2, 4], "identity_samples": 2},
        "workdir": str(tmp_path / "work"), "results_path": str(results_path),
    }
    config_path = tmp_path / "quality.json"; config_path.write_text(json.dumps(config))
    assert cli_main(["--config", str(config_path)]) == 0
    record = json.loads(results_path.read_text())
    assert construction_results.read_bytes() == source_before
    # Canonical construction accounting/provenance survives unchanged.
    for field in ("builder", "build_calc", "merge_calc", "total_calc",
                  "distance_counts_by_phase", "distance_counts_by_layer", "fastkcna"):
        assert record[field] == original[field]
    assert record["namespace"] == "fasthnsw-quality"
    assert record["algo"] == "FastHNSW"
    assert record["analysis_dataset"] == "tiny"
    assert record["construction_namespace"] == "fastkcna-canonical"
    assert record["construction_run_key"] == "canonical-pg2"
    assert record["run_key"] == record["quality_run_key"]
    assert record["run_key"] != "canonical-pg2"
    assert record["recall@2"] == 0.75
    assert [point["ef"] for point in record["recall_curve"]] == [2, 4]
    assert [point["d_s"] for point in record["recall_curve"]] == [2.0, 4.0]
    assert [point["query_distance_total"] for point in record["recall_curve"]] == [4, 8]
    assert record["d_s@0.95"] is None
    assert record["quality_evaluator"]["query_threads"] == 1
    assert record["quality_evaluator"]["hnswlib_revision"] == PINNED_HNSWLIB_REVISION
    assert record["quality_artifacts"]["index"]["sha256"] == original["output_index_sha256"]
    assert all("result_labels" not in point for point in record["quality_evaluator"]["points"])
    # Idempotent result identity; no second row.
    assert cli_main(["--config", str(config_path)]) == 0
    assert len(results_path.read_text().splitlines()) == 1


def test_cli_rejects_dataset_and_result_namespace_mismatch(tmp_path):
    base, query, gt, index, construction_results, evaluator, original = fixture(tmp_path)
    config = {
        "namespace": "fasthnsw-quality", "binaries": {"evaluator": str(evaluator)},
        "construction": {"results_path": str(construction_results), "run_key": "canonical-pg2"},
        "dataset": {"name": "wrong", "dim": 2, "nb": 4, "base": str(base),
                    "query": str(query), "groundtruth": str(gt)},
        "eval": {"k": 2, "kk": 3, "nq": 2, "efs_array": [2], "identity_samples": 2},
        "workdir": str(tmp_path / "work"),
        "results_path": str(tmp_path / "results/fasthnsw_quality_tiny.jsonl"),
    }
    config_path = tmp_path / "bad.json"; config_path.write_text(json.dumps(config))
    with pytest.raises(FastHNSWError, match="construction dataset mismatch"):
        cli_main(["--config", str(config_path)])
    config["dataset"]["name"] = "tiny"
    config["results_path"] = str(tmp_path / "results/fastkcna_canonical_tiny.jsonl")
    config_path.write_text(json.dumps(config))
    with pytest.raises(FastHNSWError, match="distinct"):
        cli_main(["--config", str(config_path)])


def test_prepared_configs_select_existing_canonical_pg2_and_common_sweep():
    expected = {
        "sift100k": (100_000, "3981b337d2e4", "results/fastkcna_canonical_sift100k.jsonl"),
        "sift1m": (1_000_000, "ee2470d409d0", "results/fastkcna_canonical_sift1m.jsonl"),
    }
    for scale, (n, run_key, source) in expected.items():
        cfg = json.loads((ROOT / "config" / f"fasthnsw_quality_{scale}.json").read_text())
        assert cfg["namespace"] == "fasthnsw-quality"
        assert cfg["construction"] == {"results_path": source, "run_key": run_key}
        assert cfg["dataset"]["nb"] == n
        assert cfg["dataset"]["analysis_name"] == f"bigann{scale.removeprefix('sift')}"
        assert cfg["eval"] == {
            "k": 10, "kk": 100, "nq": 10000,
            "efs_array": [10, 50, 100, 200, 400], "identity_samples": 8,
        }
        assert "fasthnsw_quality" in Path(cfg["results_path"]).name
