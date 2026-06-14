"""Pipeline: load -> partition -> build leaves -> merge -> evaluate -> record.

Two independently cached build stages:

  leaves : HNSWs on each partition, keyed by (dataset, hnsw params, n_parts,
           partition_method, seed). Shared across all merge algorithms/orders
           that use the same partitioning - built once, reused.
  index  : the merged root, keyed additionally by (algo, order, merge params).

Both bundles store their per-phase distance counts and timings, so a cache hit
returns those numbers without recomputation. The distance counter is rebound
onto loaded HNSWs so subsequent phases (merge, search) keep counting correctly.
"""
from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np

from .cache import Cache, ResultsLog
from .config import ExperimentCfg
from .data import Dataset, load_sift_like, make_synthetic, partition
from .distance import CountingDistance
from .evaluate import recall_at_k
from .index import (
    build_leaf,
    build_sigm,
    build_and_eval_nndescent,
    divide_and_conquer,
    quality_stats,
)


def vars_hnsw(cfg: ExperimentCfg) -> dict:
    return {"m": cfg.hnsw.m, "m0": cfg.hnsw.m0,
            "ef": cfg.hnsw.ef, "ef_construction": cfg.hnsw.ef_construction}


def load_dataset(cfg: ExperimentCfg, cache: Cache) -> Dataset:
    d = cfg.dataset

    def _compute():
        if d.name == "synthetic":
            return make_synthetic(d.n, d.dim, d.n_clusters, d.n_queries, d.gt_k, cfg.seed)
        return load_sift_like(d.name, d.base_path, d.query_path, d.gt_path, d.base_limit)

    return cache.get_or_compute("dataset", cfg.data_key(), _compute)


def _rebind(sub, dist: CountingDistance):
    sub.hnsw.distance_func = dist
    return sub


def load_leaves(cfg: ExperimentCfg, ds: Dataset, dist: CountingDistance, cache: Cache) -> dict:
    """Cached leaf-build stage. Returns {leaves, build_calc, build_seconds}."""
    def _compute():
        parts = partition(ds.base, cfg.merge.n_parts, cfg.merge.partition_method, cfg.seed)
        t0 = time.perf_counter()
        c0 = dist.snapshot()
        leaves = [build_leaf(ds.base, ids, dist, vars_hnsw(cfg), cfg.seed + i)
                  for i, ids in enumerate(parts)]
        return {"leaves": leaves,
                "build_calc": dist.snapshot() - c0,
                "build_seconds": time.perf_counter() - t0}

    bundle = cache.get_or_compute("leaves", cfg.leaves_key(), _compute)
    for sub in bundle["leaves"]:
        _rebind(sub, dist)               # leaves may be unpickled with a stale counter
    return bundle


def build_index_bundle(cfg: ExperimentCfg, ds: Dataset, dist: CountingDistance, cache: Cache) -> dict:
    if cfg.builder == "sigm":
        t0 = time.perf_counter()
        c0 = dist.snapshot()
        root = build_sigm(ds.base, dist, vars_hnsw(cfg), cfg.seed)
        return {"sub": root, "build_calc": dist.snapshot() - c0, "merge_calc": 0,
                "build_seconds": time.perf_counter() - t0, "merge_seconds": 0.0,
                "quality": quality_stats(root)}

    leaves_bundle = load_leaves(cfg, ds, dist, cache)
    random.seed(cfg.seed)            # CGTM uses random.choice; make merge deterministic
    np.random.seed(cfg.seed)
    t1 = time.perf_counter()
    c1 = dist.snapshot()
    root = divide_and_conquer(leaves_bundle["leaves"], cfg.merge.algo, dist,
                              cfg.merge.params, cfg.merge.order)
    return {"sub": root,
            "build_calc": leaves_bundle["build_calc"], "merge_calc": dist.snapshot() - c1,
            "build_seconds": leaves_bundle["build_seconds"],
            "merge_seconds": time.perf_counter() - t1,
            "quality": quality_stats(root)}


def run(cfg: ExperimentCfg, cache: Cache, results: Optional[ResultsLog] = None) -> dict:
    dist = CountingDistance("l2")
    ds = load_dataset(cfg, cache)

    if cfg.builder == "nndescent":
        rec = cache.get_or_compute(
            "nndescent", cfg.run_key(),
            lambda: build_and_eval_nndescent(ds.base, ds.queries, ds.groundtruth,
                                             k=cfg.eval.k, seed=cfg.seed))
        record = {**_meta(cfg, ds), **rec}
        if results:
            results.append(record)
        return record

    bundle = cache.get_or_compute(
        "index", cfg.index_key(),
        lambda: build_index_bundle(cfg, ds, dist, cache))
    _rebind(bundle["sub"], dist)         # for search-time counting in eval

    ev = recall_at_k(bundle["sub"], ds.queries, ds.groundtruth, dist,
                     k=cfg.eval.k, ef=cfg.eval.ef)

    record = {
        **_meta(cfg, ds),
        "algo": cfg.merge.algo if cfg.builder == "merge" else "SIGM",
        "build_calc": bundle["build_calc"], "merge_calc": bundle["merge_calc"],
        "total_calc": bundle["build_calc"] + bundle["merge_calc"],
        "build_seconds": bundle["build_seconds"], "merge_seconds": bundle["merge_seconds"],
        **bundle["quality"], **ev,
    }
    if results:
        results.append(record)
    return record


def _meta(cfg: ExperimentCfg, ds: Dataset) -> dict:
    return {
        "run_key": cfg.run_key(), "builder": cfg.builder, "dataset": ds.name,
        "n": ds.n, "dim": ds.dim,
        "n_parts": cfg.merge.n_parts if cfg.builder == "merge" else 1,
        "partition_method": cfg.merge.partition_method if cfg.builder == "merge" else "-",
        "order": cfg.merge.order if cfg.builder == "merge" else "-",
        "m": cfg.hnsw.m, "m0": cfg.hnsw.m0 if cfg.hnsw.m0 is not None else 2 * cfg.hnsw.m,
        "ef_construction": cfg.hnsw.ef_construction, "seed": cfg.seed,
    }
