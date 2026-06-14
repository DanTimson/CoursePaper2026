"""Smoke test: run every builder on a small synthetic dataset end-to-end.

Validates: leaf build, id-shift merge for NGM/IGTM/CGTM under both orders,
SIGM baseline, NN-Descent, recall computation, distance counting, and caching.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngmbench.cache import Cache, ResultsLog
from ngmbench.config import ExperimentCfg, DatasetCfg, HNSWParams, MergeCfg, EvalCfg
from ngmbench.pipeline import run

DATA = DatasetCfg(name="synthetic", n=1500, dim=16, n_clusters=6, n_queries=80, gt_k=10)
HNSW = HNSWParams(m=8, ef=10, ef_construction=24)
EVAL = EvalCfg(k=10, ef=48)

cache = Cache(".ngmbench_cache_smoke")
results = ResultsLog("results_smoke.jsonl")

cfgs = []
for algo in ["NGM", "IGTM", "CGTM"]:
    for order in ["balanced", "sequential"]:
        cfgs.append(ExperimentCfg(
            dataset=DATA, hnsw=HNSW, eval=EVAL, builder="merge",
            merge=MergeCfg(algo=algo, n_parts=4, partition_method="random", order=order),
        ))
cfgs.append(ExperimentCfg(dataset=DATA, hnsw=HNSW, eval=EVAL, builder="sigm"))
cfgs.append(ExperimentCfg(dataset=DATA, hnsw=HNSW, eval=EVAL, builder="nndescent"))

hdr = f"{'builder':9} {'algo':6} {'order':10} {'parts':5} {'recall@10':9} {'build':>9} {'merge':>9} {'total':>9} {'reach':>6}"
print(hdr); print("-" * len(hdr))
for cfg in cfgs:
    r = run(cfg, cache, results)
    print(f"{r['builder']:9} {r.get('algo','-'):6} {str(r.get('order','-')):10} "
          f"{r.get('n_parts','-')!s:5} {r.get('recall@10', float('nan')):.3f}     "
          f"{str(r.get('build_calc','-')):>9} {str(r.get('merge_calc','-')):>9} "
          f"{str(r.get('total_calc','-')):>9} {r.get('frac_reachable_L0', float('nan')):.3f}")

print("\nCache-hit rerun (should be instant, identical numbers):")
r2 = run(cfgs[2], cache)   # IGTM balanced again
print(f"  IGTM balanced recall@10={r2['recall@10']:.3f} total_calc={r2['total_calc']}")
