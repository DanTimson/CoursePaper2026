from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_hnswmerger():
    path = ROOT / "ngmbench" / "index" / "hnswmerger.py"
    spec = importlib.util.spec_from_file_location("patched_hnswmerger", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_adaptive_lambda_schedule_m16():
    module = load_hnswmerger()
    f = module.adaptive_merge_lambda
    assert f(4, 16, 1, 1) == 4
    assert f(4, 16, 2, 1) == 7
    assert f(4, 16, 4, 1) == 10
    assert f(4, 16, 8, 1) == 13
    assert f(4, 16, 16, 1) == 16
    assert f(4, 16, 100, 1) == 16


def test_full_configs_are_single_thread_and_cover_all_p():
    for filename in (
        "total_cost_bigann10k.json",
        "total_cost_bigann100k.json",
        "total_cost_bigann1m.json",
    ):
        cfg = json.loads((ROOT / "config" / filename).read_text())
        assert cfg["threads"] == 1
        assert cfg["hnsw"] == {"M": 16, "ef_construction": 200}
        canonical = {
            spec["label"]: set(spec["n_parts"])
            for spec in cfg["sweep"]
            if spec["label"] in {"IGTM tuned", "CGTM tuned", "NGM search_ef=10"}
        }
        assert canonical
        assert all(parts == {2, 4, 8, 16} for parts in canonical.values())
