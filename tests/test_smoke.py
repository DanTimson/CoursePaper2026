"""Minimal correctness checks. Run with: pytest -q"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngmbench.cache import Cache
from ngmbench.config import ExperimentCfg, DatasetCfg, HNSWParams, MergeCfg, EvalCfg
from ngmbench.pipeline import run

DATA = DatasetCfg(name="synthetic", n=800, dim=12, n_clusters=5, n_queries=40, gt_k=10)
HP = HNSWParams(m=8, ef=10, ef_construction=20)
EV = EvalCfg(k=10, ef=40)


def _cfg(**kw):
    return ExperimentCfg(dataset=DATA, hnsw=HP, eval=EV, **kw)


def test_merge_algos_run_and_recall_reasonable(tmp_path):
    cache = Cache(str(tmp_path / "c"))
    for algo in ["NGM", "IGTM", "CGTM"]:
        for order in ["balanced", "sequential"]:
            r = run(_cfg(builder="merge",
                         merge=MergeCfg(algo=algo, n_parts=4, order=order)), cache)
            assert r["recall@10"] > 0.8
            assert r["total_calc"] > 0
            assert 0.0 <= r["frac_reachable_L0"] <= 1.0


def test_determinism(tmp_path):
    r1 = run(_cfg(builder="merge", merge=MergeCfg(algo="CGTM", n_parts=4)),
             Cache(str(tmp_path / "a")))
    r2 = run(_cfg(builder="merge", merge=MergeCfg(algo="CGTM", n_parts=4)),
             Cache(str(tmp_path / "b")))
    assert r1["merge_calc"] == r2["merge_calc"]


def test_baselines(tmp_path):
    cache = Cache(str(tmp_path / "c"))
    assert run(_cfg(builder="sigm"), cache)["recall@10"] > 0.8
    assert run(_cfg(builder="nndescent"), cache)["recall@10"] > 0.8
