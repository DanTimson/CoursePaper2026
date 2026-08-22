from __future__ import annotations

import json
from pathlib import Path

import pytest

from ngmbench import cli_build_budget


def test_run_key_changes_with_construction_semantics():
    base = {
        "builder": "hnswmerger-build-only",
        "dataset": "bigann1m",
        "n_parts": 2,
        "partition_method": "range",
        "m": 16,
        "ef_construction": 200,
        "threads": 1,
    }
    key = cli_build_budget._run_key(base)
    assert cli_build_budget._run_key({**base, "ef_construction": 32}) != key
    assert cli_build_budget._run_key({**base, "threads": 2}) != key
    assert cli_build_budget._run_key({**base, "m": 32}) != key


def test_build_budget_sums_actual_leaf_measurements_and_resumes(tmp_path, monkeypatch, capsys):
    for name in ("base.fvecs", "query.fvecs", "gt.ivecs"):
        (tmp_path / name).write_bytes(b"x")

    cfg = {
        "binaries": {"builds": "/fake/builds", "exps": "/fake/exps"},
        "hnsw": {"M": 16, "ef_construction": 32},
        "eval": {"k": 10, "kk": 100, "nq": 10, "efs_array": [10]},
        "threads": 1,
        "partitions": [1, 3],
        "results_path": str(tmp_path / "results.jsonl"),
        "datasets": [{
            "name": "tiny10",
            "dim": 4,
            "nb": 10,
            "base": str(tmp_path / "base.fvecs"),
            "query": str(tmp_path / "query.fvecs"),
            "groundtruth": str(tmp_path / "gt.ivecs"),
            "workdir": str(tmp_path / "work"),
        }],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    calls = []

    class FakeRunner:
        def __init__(self, paths, params):
            self.paths = paths
            self.params = params

        def build_leaf(self, lo, hi):
            calls.append((lo, hi, self.params.ef_construction))
            # Deliberately depends on both size and offset so P=3 is not
            # representable as P * B(N/P).
            calc = (hi - lo) * 10 + lo
            return str(tmp_path / f"leaf_{lo}_{hi}.hnsw"), {
                "build_calc": calc,
                "build_seconds": calc / 1000,
            }

    monkeypatch.setattr(cli_build_budget, "HNSWMergerRunner", FakeRunner)

    assert cli_build_budget.main(["--config", str(config_path)]) == 0
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2

    p1 = next(r for r in rows if r["n_parts"] == 1)
    assert p1["build_calc"] == 100
    assert p1["total_calc"] == 100
    assert p1["merge_calc"] == 0
    assert p1["ef_construction"] == 32

    p3 = next(r for r in rows if r["n_parts"] == 3)
    # contiguous_partitions(10,3) => [0,3), [3,6), [6,10)
    assert [(x["lrange"], x["rrange"]) for x in p3["leaf_builds"]] == [(0, 3), (3, 6), (6, 10)]
    assert p3["build_calc"] == 30 + 33 + 46
    assert p3["total_calc"] == p3["build_calc"]

    call_count = len(calls)
    assert cli_build_budget.main(["--config", str(config_path)]) == 0
    assert len(calls) == call_count  # log-level resume prevents any rebuild calls
    assert "skip cached log row" in capsys.readouterr().out


def test_build_budget_rejects_bad_partition_list(tmp_path):
    cfg = {
        "binaries": {"builds": "/fake/builds", "exps": "/fake/exps"},
        "partitions": [0],
        "datasets": [{"name": "unused"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="positive integers"):
        cli_build_budget.main(["--config", str(path)])
