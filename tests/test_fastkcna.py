from __future__ import annotations

import json
import os
import stat
import struct
import subprocess
from pathlib import Path

import pytest

from ngmbench.cli_fastkcna import main as cli_main
from ngmbench.index.fastkcna import (
    DIAGNOSTIC_WARNING,
    FastKCNAError,
    FastKCNAParams,
    FastKCNAPaths,
    FastKCNARunner,
    check_fasthnsw_compatibility,
    prepare_lshkit,
)

ROOT = Path(__file__).resolve().parents[1]


def executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def fvecs(path: Path, n: int = 4, dim: int = 3) -> Path:
    with path.open("wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", dim))
            f.write(struct.pack("<" + "f" * dim, *[float(i + j) for j in range(dim)]))
    return path


def fake_checkout(tmp_path: Path, *, builder_mode: str = "ok", converter_mode: str = "ok") -> Path:
    root = tmp_path / "FastKCNA"
    code = root / "code"
    code.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    (root / "tracked").write_text("fixture")
    subprocess.run(["git", "-C", str(root), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)

    converter = r'''
import pathlib, struct, sys
counter = pathlib.Path(__file__).with_name("converter-count")
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if "MODE" == "fail":
    print("conversion failure", file=sys.stderr); raise SystemExit(7)
source, output = map(pathlib.Path, sys.argv[1:3])
raw = source.read_bytes(); dim = struct.unpack("<i", raw[:4])[0]
record = 4 + dim * 4; n = len(raw) // record
payload = b"".join(raw[i * record + 4:(i + 1) * record] for i in range(n))
output.write_bytes(struct.pack("<iii", 4, n, dim) + payload)
print(f"dim: {dim} n: {n}")
'''.replace("MODE", converter_mode)
    executable(code / "fvec2lshkit", converter)

    builder = r'''
import pathlib, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
if "MODE" == "fail":
    print("builder failure", file=sys.stderr); raise SystemExit(9)
if "MODE" != "missing":
    pathlib.Path(args["-index_path"]).write_bytes(b"index-output")
    pathlib.Path(args["-log_path"]).write_text("prune scan_rate,0.25,\nsearch scan_rate,0.5,\n")
print("iteration: 1 recall: 0.8 cost: 1.25")
print("diagnostic stderr", file=sys.stderr)
'''.replace("MODE", builder_mode)
    executable(code / "build_index", builder)
    return root


def paths(root: Path) -> FastKCNAPaths:
    return FastKCNAPaths.resolve({"checkout": str(root)}, environ={})


def params(pg_type: int = 0, threads: int = 2) -> FastKCNAParams:
    return FastKCNAParams(
        pg_type=pg_type, K=16, L=16, S=4, R=8, iter=2,
        search_L=16, search_K=16, nsg_R=4, step=2, loop_i=1,
        alpha=60, tau=0, nthreads=threads, controls=8, recall=0.9,
    )


def test_environment_and_config_resolution(tmp_path, monkeypatch):
    checkout = fake_checkout(tmp_path)
    resolved = FastKCNAPaths.resolve({}, environ={"FASTKCNA_ROOT": str(checkout)})
    assert resolved.checkout == checkout.resolve()
    assert resolved.build_index == (checkout / "code/build_index").resolve()
    explicit = FastKCNAPaths.resolve({
        "checkout": "$FK", "build_index": "$FK/code/build_index",
        "fvec2lshkit": "$FK/code/fvec2lshkit",
    }, environ={"FK": str(checkout)})
    assert explicit == resolved


def test_missing_backend_and_unresolved_environment_fail_clearly(tmp_path):
    with pytest.raises(FastKCNAError, match="checkout is missing"):
        FastKCNAPaths.resolve({"checkout": str(tmp_path / "absent")}, environ={})
    with pytest.raises(FastKCNAError, match="unresolved environment variable"):
        FastKCNAPaths.resolve({"checkout": "$NOT_SET"}, environ={})


def test_non_cli_fixed_defaults_cannot_be_misrepresented():
    with pytest.raises(ValueError, match="refusing misleading overrides"):
        FastKCNAParams(
            pg_type=0, K=16, L=16, S=4, R=8, iter=2,
            search_L=16, search_K=16, nsg_R=4, step=2, loop_i=1,
            alpha=60, tau=0, nthreads=1, controls=8, recall=0.9,
            seed=1,
        )


def test_command_construction_is_complete_for_both_pg_types(tmp_path):
    checkout = fake_checkout(tmp_path)
    runner = FastKCNARunner(paths(checkout), tmp_path / "work")
    for pg in (0, 2):
        command = runner.command(tmp_path / "data.lshkit", tmp_path / "out", tmp_path / "log", params(pg))
        assert command[0] == str((checkout / "code/build_index").resolve())
        pairs = dict(zip(command[1::2], command[2::2]))
        assert pairs["-pg_type"] == str(pg)
        assert pairs["-alpha"] == "60"
        assert pairs["-nthreads"] == "2"
        assert set(pairs) == {
            "-data_path", "-index_path", "-log_path", "-K", "-L", "-S", "-R",
            "-iter", "-search_L", "-nsg_R", "-search_K", "-step", "-loop_i",
            "-alpha", "-tau", "-nthreads", "-controls", "-recall", "-pg_type",
        }


def test_missing_input_fails_before_conversion(tmp_path):
    checkout = fake_checkout(tmp_path)
    with pytest.raises(FastKCNAError, match="input fvecs is missing"):
        prepare_lshkit(tmp_path / "missing.fvecs", tmp_path / "x.lshkit", paths(checkout))


def test_conversion_is_validated_and_idempotent(tmp_path):
    checkout = fake_checkout(tmp_path)
    source = fvecs(tmp_path / "tiny.fvecs")
    output = tmp_path / "cache/tiny.lshkit"
    first = prepare_lshkit(source, output, paths(checkout))
    second = prepare_lshkit(source, output, paths(checkout))
    assert first["cached"] is False and second["cached"] is True
    assert (checkout / "code/converter-count").read_text() == "1"
    assert struct.unpack("<iii", output.read_bytes()[:12]) == (4, 4, 3)
    same_size_corruption = bytearray(output.read_bytes())
    same_size_corruption[-1] ^= 0xFF
    output.write_bytes(same_size_corruption)
    third = prepare_lshkit(source, output, paths(checkout))
    assert third["cached"] is False
    assert (checkout / "code/converter-count").read_text() == "2"
    output.write_bytes(b"truncated")
    fourth = prepare_lshkit(source, output, paths(checkout))
    assert fourth["cached"] is False
    assert (checkout / "code/converter-count").read_text() == "3"


def test_conversion_nonzero_is_explicit(tmp_path):
    checkout = fake_checkout(tmp_path, converter_mode="fail")
    with pytest.raises(FastKCNAError, match="conversion failed.*exit=7"):
        prepare_lshkit(fvecs(tmp_path / "tiny.fvecs"), tmp_path / "tiny.lshkit", paths(checkout))


def prepared(tmp_path: Path, checkout: Path):
    source = fvecs(tmp_path / "tiny.fvecs")
    output = tmp_path / "tiny.lshkit"
    conversion = prepare_lshkit(source, output, paths(checkout))
    return output, conversion


def test_result_metadata_and_diagnostic_accounting_boundary(tmp_path):
    checkout = fake_checkout(tmp_path)
    data, conversion = prepared(tmp_path, checkout)
    result = FastKCNARunner(paths(checkout), tmp_path / "runs").run(data, params(2), "abc", conversion)
    assert result["exit_status"] == 0
    assert result["threads"] == 2
    assert len(result["fastkcna"]["revision"]) == 40
    assert result["command_shell"]
    assert Path(result["fastkcna_log_path"]).is_file()
    assert Path(result["output_index_path"]).is_file()
    assert result["diagnostic_fastkcna_counters"]["iteration_cost_ratios"] == [1.25]
    assert result["diagnostic_fastkcna_counters"]["prune_scan_rate"] == [[0.25]]
    assert result["accounting_warning"] == DIAGNOSTIC_WARNING
    assert result["canonical_distance_counts_available"] is False
    for forbidden in ("build_calc", "merge_calc", "total_calc"):
        assert forbidden not in result


@pytest.mark.parametrize("mode, message", [("fail", "build failed.*exit=9"), ("missing", "expected output is missing")])
def test_builder_failure_and_missing_output_are_explicit(tmp_path, mode, message):
    checkout = fake_checkout(tmp_path, builder_mode=mode)
    data, conversion = prepared(tmp_path, checkout)
    with pytest.raises(FastKCNAError, match=message):
        FastKCNARunner(paths(checkout), tmp_path / "runs").run(data, params(), "abc", conversion)


def test_cli_writes_only_new_fastkcna_namespace(tmp_path, monkeypatch):
    checkout = fake_checkout(tmp_path)
    source = fvecs(tmp_path / "tiny.fvecs")
    canonical = tmp_path / "results/canonical.jsonl"
    canonical.parent.mkdir()
    canonical.write_text('{"sentinel": true}\n')
    result_path = tmp_path / "results/fastkcna_exploratory_tiny.jsonl"
    config = {
        "namespace": "fastkcna-exploratory", "tuning_status": "untuned exploratory",
        "binaries": {"checkout": str(checkout)},
        "dataset": {"name": "tiny", "dim": 3, "nb": 4, "base": str(source)},
        "threads": 1, "fastkcna_params": params(0, 1).complete_metadata(),
        "workdir": str(tmp_path / "work"), "results_path": str(result_path),
    }
    # complete_metadata adds result-only fields not accepted by the constructor.
    config["fastkcna_params"] = {key: value for key, value in config["fastkcna_params"].items() if key not in {
        "fixed_upstream_defaults", "tuning_status", "hnswlib_ef_construction_equivalence"
    }}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    assert cli_main(["--config", str(config_path)]) == 0
    record = json.loads(result_path.read_text())
    assert record["namespace"] == "fastkcna-exploratory"
    assert "build_calc" not in record
    assert canonical.read_text() == '{"sentinel": true}\n'


def test_compatibility_smoke_records_success_or_exact_rejection(tmp_path):
    checkout = fake_checkout(tmp_path)
    data, conversion = prepared(tmp_path, checkout)
    index = Path(FastKCNARunner(paths(checkout), tmp_path / "runs").run(data, params(2), "pg2", conversion)["output_index_path"])
    base = fvecs(tmp_path / "base.fvecs")
    query = fvecs(tmp_path / "query.fvecs", n=2)
    gt = tmp_path / "gt.ivecs"; gt.write_bytes(struct.pack("<iii", 2, 0, 1))
    success = executable(tmp_path / "exps-ok", r'''
print("Merge method: REBUILD")
print("set ef = 10")
print("pure query time: 0.001 s] ef=10")
print("search distance calls per query = 12")
print("R@100 = 1.0")
''')
    out = check_fasthnsw_compatibility(
        exps_bin=success, index_path=index, base_path=base, query_path=query,
        groundtruth_path=gt, workdir=tmp_path / "compat-ok", dim=3, nb=4, nq=2,
    )
    assert out["compatible"] is True
    assert out["recall_curve"][0]["d_s"] == 12.0
    reject = executable(tmp_path / "exps-reject", 'import sys\nprint("Index seems to be corrupted or unsupported", file=sys.stderr)\nraise SystemExit(1)\n')
    out = check_fasthnsw_compatibility(
        exps_bin=reject, index_path=index, base_path=base, query_path=query,
        groundtruth_path=gt, workdir=tmp_path / "compat-no", dim=3, nb=4, nq=2,
    )
    assert out["compatible"] is False
    assert out["exit_status"] == 1
    assert "corrupted or unsupported" in out["stderr"]


def test_rebuild_thread_matrix_generates_distinct_monolithic_build_configs(tmp_path, monkeypatch):
    from ngmbench.index.hnswmerger import CppParams, HNSWMergerRunner, Paths

    for name in ("base.fvecs", "query.fvecs", "gt.ivecs"):
        (tmp_path / name).write_bytes(b"fixture")
    backend_paths = Paths(
        builds_bin=str(tmp_path / "builds"), exps_bin=str(tmp_path / "exps"),
        base=str(tmp_path / "base.fvecs"), query=str(tmp_path / "query.fvecs"),
        groundtruth=str(tmp_path / "gt.ivecs"), workdir=str(tmp_path / "work"),
    )
    generated = []

    def fake_run(self, binary, cfg_path):
        generated.append((self.cp.thread, Path(cfg_path), Path(cfg_path).read_text(), self.env["OMP_NUM_THREADS"]))
        return "distance calls = 123\n[1.5 s] build index\n"

    monkeypatch.setattr(HNSWMergerRunner, "_run", fake_run)
    for thread in (1, 2, 4, 8):
        runner = HNSWMergerRunner(
            backend_paths, CppParams(dim=128, nb=100_000, M=16, ef_construction=200, thread=thread)
        )
        runner.build_leaf(0, 100_000)

    assert [item[0] for item in generated] == [1, 2, 4, 8]
    assert [item[3] for item in generated] == ["1", "2", "4", "8"]
    for thread, cfg_path, cfg, _ in generated:
        assert cfg_path.name == f"build_0_100000_M16_efc200_t{thread}.cfg"
        assert "lrange = 0" in cfg and "rrange = 100000" in cfg
        assert "max_elements = 100000" in cfg
        assert "M = 16" in cfg and "ef_construction = 200" in cfg
        assert f"leaf_0_100000_M16_efc200_t{thread}.hnsw" in cfg


def test_prepared_configs_cover_overnight_and_thread_matrix():
    for scale, nb in (("sift100k", 100_000), ("sift1m", 1_000_000)):
        for pg in (0, 2):
            cfg = json.loads((ROOT / "config" / f"fastkcna_{scale}_pg{pg}.json").read_text())
            assert cfg["namespace"] == "fastkcna-exploratory"
            assert cfg["dataset"]["nb"] == nb
            assert cfg["fastkcna_params"]["pg_type"] == pg
            assert cfg["fastkcna_params"]["alpha"] == 60
            assert cfg["fastkcna_params"]["nsg_R"] == 16
            assert cfg["tuning_status"] == "untuned exploratory"
            assert "fastkcna" in Path(cfg["results_path"]).name
    matrix = json.loads((ROOT / "config/insert_thread_invariance_sift100k.json").read_text())
    spec = matrix["sweep"][0]
    assert spec["algo"] == ["REBUILD"] and spec["n_parts"] == [1]
    assert [point["thread"] for point in spec["params"]] == [1, 2, 4, 8]
    assert matrix["dataset"]["nb"] == 100_000
    assert matrix["hnsw"] == {"M": 16, "ef_construction": 200}
    assert matrix["experiment_metadata"]["build_range"] == [0, 100_000]
    assert matrix["experiment_metadata"]["from_empty"] is True
    assert "bigann100k.jsonl" not in matrix["results_path"]
    assert matrix["seed_metadata"]["value"] == 100
