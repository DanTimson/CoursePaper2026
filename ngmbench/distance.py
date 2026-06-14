"""Distance functions with call counting.

The vendored HNSW routes every distance evaluation through a ``distance_func``
attribute (and passes one into ``neighborhood_construction``). By injecting a
``CountingDistance`` instance as that function into *every* HNSW we create, all
distance evaluations during build, merge, and search are counted — which is the
metric the paper reports.

A single counter is shared across an experiment run; snapshot it at phase
boundaries (build vs merge vs eval) to attribute counts per phase.
"""
from __future__ import annotations

import numpy as np


class CountingDistance:
    """Callable L2 distance that counts evaluations.

    Use one instance per experiment run. ``count`` is the running total;
    ``snapshot()`` returns the current total so you can diff across phases.
    """

    __slots__ = ("count", "_metric")

    def __init__(self, metric: str = "l2"):
        if metric != "l2":
            raise ValueError(f"unsupported metric: {metric!r}")
        self.count = 0
        self._metric = metric

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        self.count += 1
        # float() keeps the vendored heuristic's comparisons in plain Python
        # floats (it does min(map(...)) over distances).
        return float(np.linalg.norm(a - b))

    def snapshot(self) -> int:
        return self.count

    def reset(self) -> None:
        self.count = 0
