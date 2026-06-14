"""Experiment configuration and stable hashing for stage caching.

A config hashes to a stable hex digest; the cache keys built artifacts on the
digests of the config slices they depend on, so an unchanged stage is skipped on
rerun.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class DatasetCfg:
    name: str = "synthetic"
    # synthetic params
    n: int = 2000
    dim: int = 16
    n_clusters: int = 8
    n_queries: int = 100
    gt_k: int = 10
    # real-dataset file paths (used when name != "synthetic")
    base_path: Optional[str] = None
    query_path: Optional[str] = None
    gt_path: Optional[str] = None
    base_limit: Optional[int] = None


@dataclass
class HNSWParams:
    m: int = 16
    m0: Optional[int] = None
    ef: int = 10
    ef_construction: int = 32


@dataclass
class MergeCfg:
    algo: str = "IGTM"               # NGM | IGTM | CGTM
    n_parts: int = 4
    partition_method: str = "random"  # random | kmeans
    order: str = "balanced"           # balanced | sequential
    params: dict = field(default_factory=dict)   # per-algo merge kwargs


@dataclass
class EvalCfg:
    k: int = 10
    ef: int = 64


@dataclass
class ExperimentCfg:
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    hnsw: HNSWParams = field(default_factory=HNSWParams)
    merge: MergeCfg = field(default_factory=MergeCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    seed: int = 0
    builder: str = "merge"            # merge | sigm | nndescent

    # ---- hashing helpers -------------------------------------------------- #
    def _digest(self, obj) -> str:
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def data_key(self) -> str:
        return self._digest(asdict(self.dataset))

    def leaves_key(self) -> str:
        return self._digest(
            {"data": asdict(self.dataset), "hnsw": asdict(self.hnsw),
             "n_parts": self.merge.n_parts, "pm": self.merge.partition_method,
             "seed": self.seed}
        )

    def index_key(self) -> str:
        return self._digest(
            {"leaves": self.leaves_key(), "merge": asdict(self.merge),
             "builder": self.builder, "seed": self.seed}
        )

    def run_key(self) -> str:
        return self._digest(asdict(self))


def cfg_from_dict(d: dict) -> ExperimentCfg:
    return ExperimentCfg(
        dataset=DatasetCfg(**d.get("dataset", {})),
        hnsw=HNSWParams(**d.get("hnsw", {})),
        merge=MergeCfg(**d.get("merge", {})),
        eval=EvalCfg(**d.get("eval", {})),
        seed=d.get("seed", 0),
        builder=d.get("builder", "merge"),
    )
