from pathlib import Path

import pytest

from ngmbench.index.hnswmerger import CppParams, HNSWMergerRunner, Paths


def test_query_only_uses_rebuild_load_path_and_preserves_logical_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.fvecs"
    query = tmp_path / "query.fvecs"
    groundtruth = tmp_path / "groundtruth.ivecs"
    for path in (base, query, groundtruth):
        path.write_bytes(b"x")

    paths = Paths(
        builds_bin=str(tmp_path / "builds"),
        exps_bin=str(tmp_path / "exps"),
        base=str(base),
        query=str(query),
        groundtruth=str(groundtruth),
        workdir=str(tmp_path / "work"),
    )
    params = CppParams(
        dim=128,
        nb=10,
        M=16,
        ef_construction=200,
        k=10,
        kk=100,
        nq=10,
        efs_array=[10],
        thread=1,
    )
    runner = HNSWMergerRunner(paths, params)

    index = tmp_path / "existing.hnsw"
    index.write_bytes(b"existing-index")

    stdout = (
        "Merge method: REBUILD\n"
        "set ef = 10\n"
        "[search time: 0.002 s, pure query time: 0.001 s] ef=10\n"
        "search distance calls per query = 12.5000\n"
        "Intersection-merged index R@100 = 0.9000\n"
    )
    seen = {}

    def fake_run(binary: str, cfg_path: str) -> str:
        seen["binary"] = binary
        seen["config"] = Path(cfg_path).read_text()
        return stdout

    monkeypatch.setattr(runner, "_run", fake_run)

    out = runner.query_only(str(index), "INSERT", total_n=10)

    assert "merge_method = REBUILD\n" in seen["config"]
    assert "rerun = false\n" in seen["config"]
    assert "save_index = false\n" in seen["config"]
    assert f"index_path = {index}\n" in seen["config"]

    assert out["merge_method"] == "INSERT"
    assert out["recall_curve"] == [
        {
            "ef": 10,
            "recall": 0.9,
            "query_seconds": 0.001,
            "d_s": 12.5,
        }
    ]
