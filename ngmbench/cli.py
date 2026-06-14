"""CLI: expand a sweep config into runs and execute them with caching.

    python -m ngmbench.cli --config config/synthetic_demo.json

Config schema (JSON)::

    {
      "dataset": {...}, "hnsw": {...}, "eval": {...}, "seed": 0,
      "cache_dir": ".ngmbench_cache", "results_path": "results.jsonl",
      "sweep": [
        {"builder": "merge", "algo": ["NGM","IGTM","CGTM"],
         "order": ["balanced","sequential"], "n_parts": [2,4,8],
         "partition_method": ["random"], "params": {}},
        {"builder": "sigm"},
        {"builder": "nndescent"}
      ]
    }

Each merge spec is expanded over the cartesian product of its list-valued keys.
Already-cached stages are skipped, so re-running after adding a sweep point only
computes the new points.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Iterator, List

from .cache import Cache, ResultsLog
from .config import DatasetCfg, EvalCfg, ExperimentCfg, HNSWParams, MergeCfg


def _expand(spec: dict) -> Iterator[dict]:
    """Expand list-valued keys in a sweep spec into concrete dicts."""
    grid_keys = [k for k, v in spec.items() if isinstance(v, list)]
    if not grid_keys:
        yield dict(spec)
        return
    for combo in itertools.product(*[spec[k] for k in grid_keys]):
        out = dict(spec)
        out.update(dict(zip(grid_keys, combo)))
        yield out


def build_configs(conf: dict) -> List[ExperimentCfg]:
    ds = DatasetCfg(**conf.get("dataset", {}))
    hp = HNSWParams(**conf.get("hnsw", {}))
    ev = EvalCfg(**conf.get("eval", {}))
    seed = conf.get("seed", 0)
    cfgs: List[ExperimentCfg] = []
    for spec in conf.get("sweep", [{"builder": "merge"}]):
        for s in _expand(spec):
            builder = s.get("builder", "merge")
            if builder == "merge":
                merge = MergeCfg(
                    algo=s.get("algo", "IGTM"), n_parts=s.get("n_parts", 4),
                    partition_method=s.get("partition_method", "random"),
                    order=s.get("order", "balanced"), params=s.get("params", {}),
                )
                cfgs.append(ExperimentCfg(dataset=ds, hnsw=hp, eval=ev, merge=merge,
                                          seed=seed, builder="merge"))
            else:
                cfgs.append(ExperimentCfg(dataset=ds, hnsw=hp, eval=ev,
                                          seed=seed, builder=builder))
    return cfgs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a navigable-graph merge sweep.")
    ap.add_argument("--config", required=True, help="path to JSON sweep config")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        conf = json.load(f)

    cache = Cache(conf.get("cache_dir", ".ngmbench_cache"))
    results = ResultsLog(conf.get("results_path", "results.jsonl"))
    cfgs = build_configs(conf)

    print(f"{len(cfgs)} run(s). Results -> {results.path}")
    from .pipeline import run  # deferred so --help is fast
    for i, cfg in enumerate(cfgs, 1):
        rec = run(cfg, cache, results)
        tag = rec.get("algo", rec["builder"])
        r = rec.get(f"recall@{cfg.eval.k}")
        tc = rec.get("total_calc")
        print(f"[{i}/{len(cfgs)}] {rec['builder']:9} {tag:9} parts={rec['n_parts']} "
              f"order={rec['order']:10} recall@{cfg.eval.k}={r:.3f} total_calc={tc}")


if __name__ == "__main__":
    main()
