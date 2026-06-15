"""NN-Descent baseline via pynndescent.

NN-Descent builds a *flat* approximate k-NN graph by iterative local refinement
(neighbors-of-neighbors), with no hierarchy or navigability guarantee. That is
the conceptual contrast to merge construction. Because pynndescent is Numba-
compiled it does not expose a distance-evaluation count comparable to the pure-
Python merge family, so the apples-to-apples axis here is recall@k vs build
wall-clock; distance-count is reported as None.

We evaluate recall using pynndescent's own query (its prepared search over the
kNN graph). The kNN graph itself is exposed for an optional NSW-style greedy
search if you want a search procedure identical to HNSW's.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np


def build_and_eval_nndescent(
    base: np.ndarray,
    queries: np.ndarray,
    groundtruth: np.ndarray,
    k: int = 10,
    n_neighbors: int = 30,
    seed: int = 0,
    epsilons: Optional[list] = None,
) -> dict:
    from pynndescent import NNDescent

    t0 = time.perf_counter()
    index = NNDescent(
        base, metric="euclidean", n_neighbors=n_neighbors,
        random_state=seed, low_memory=True, verbose=False,
    )
    index.prepare()
    build_seconds = time.perf_counter() - t0

    def recall_at(nbrs):
        r = np.empty(len(queries))
        for i in range(len(queries)):
            truth = set(int(t) for t in groundtruth[i, :k])
            r[i] = len(truth.intersection(nbrs[i, :k].tolist())) / k
        return float(r.mean())

    # sweep query effort (epsilon) to get a recall-vs-speed curve comparable to
    # the C++ side's ef sweep. Higher epsilon = wider search = higher recall, slower.
    if epsilons is None:
        epsilons = [0.0, 0.1, 0.2, 0.4, 0.6]
    curve = []
    for eps in epsilons:
        tq = time.perf_counter()
        nbrs, _ = index.query(queries, k=k, epsilon=eps)
        qsec = time.perf_counter() - tq
        curve.append({"epsilon": eps, "recall": recall_at(nbrs), "query_seconds": qsec})

    best = max(c["recall"] for c in curve)
    return {
        "algo": "NNDescent",
        "build_seconds": build_seconds,
        "build_calc": None,            # not exposed by pynndescent (Numba)
        "merge_calc": None,
        f"recall@{k}": best,
        "recall_curve": curve,         # [{epsilon, recall, query_seconds}]
        "search_calc_per_query": None,
        "eval_k": k,
        "n_neighbors": n_neighbors,
        "n_layers": 1,
        "deg_mean": float(n_neighbors),
    }
