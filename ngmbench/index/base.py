"""Index construction primitives over the vendored HNSW.

A ``SubIndex`` pairs a vendored ``HNSW`` (keyed by *local* ids ``0..n-1``) with a
``global_ids`` array mapping local id -> global base-row id. Keeping local ids
contiguous from zero is what lets CGTM's ``curr_idx < len(hnsw_a.data)`` check
stay correct under arbitrary merge orders (see merge.py).
"""
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..distance import CountingDistance
from ..vendor_api import HNSW, heuristic


@dataclass
class SubIndex:
    hnsw: "HNSW"
    global_ids: np.ndarray   # (n,) local id -> global id

    @property
    def n(self) -> int:
        return len(self.global_ids)


def _new_hnsw(dist: CountingDistance, params: dict) -> "HNSW":
    return HNSW(
        distance_func=dist,
        m=params.get("m", 16),
        ef=params.get("ef", 10),
        ef_construction=params.get("ef_construction", 32),
        m0=params.get("m0", None),
        neighborhood_construction=heuristic,
    )


def build_leaf(
    X: np.ndarray,
    global_ids: np.ndarray,
    dist: CountingDistance,
    params: dict,
    seed: int,
) -> SubIndex:
    """Build one HNSW over the given rows of X by sequential insertion.

    Local ids are 0..len-1, in the order given by ``global_ids``.
    """
    random.seed(seed)
    np.random.seed(seed)
    h = _new_hnsw(dist, params)
    for local_id, g in enumerate(global_ids):
        h.add(local_id, X[g].astype(np.float32))
    return SubIndex(hnsw=h, global_ids=np.asarray(global_ids, dtype=np.int64))


def build_sigm(
    X: np.ndarray, dist: CountingDistance, params: dict, seed: int
) -> SubIndex:
    """SIGM baseline: a single HNSW over all points by sequential insertion."""
    return build_leaf(X, np.arange(X.shape[0]), dist, params, seed)


def shift_ids(sub: SubIndex, offset: int) -> SubIndex:
    """Return a copy of ``sub`` with every local id increased by ``offset``.

    Used to place B's ids at [nA, nA+nB) before a pairwise merge so the combined
    local id-space is contiguous from 0.
    """
    h = sub.hnsw
    new = HNSW(
        distance_func=h.distance_func, m=h.m, ef=h.ef,
        ef_construction=h.ef_construction, m0=h.m0,
        neighborhood_construction=h.neighborhood_construction,
    )
    new.data = {k + offset: v for k, v in h.data.items()}
    new.graphs = [
        {src + offset: [(dst + offset, d) for dst, d in nbrs] for src, nbrs in layer.items()}
        for layer in h.graphs
    ]
    new.enter_point = None if h.enter_point is None else h.enter_point + offset
    return SubIndex(hnsw=new, global_ids=sub.global_ids.copy())


# --------------------------------------------------------------------------- #
# Serialization (pickle of vectors + ragged graph dicts + meta)               #
# --------------------------------------------------------------------------- #
def serialize(sub: SubIndex, path: str) -> None:
    h = sub.hnsw
    state = {
        "data": {int(k): v.astype(np.float32) for k, v in h.data.items()},
        "graphs": [
            {int(src): [(int(dst), float(d)) for dst, d in nbrs] for src, nbrs in layer.items()}
            for layer in h.graphs
        ],
        "enter_point": None if h.enter_point is None else int(h.enter_point),
        "params": {"m": h.m, "m0": h.m0, "ef": h.ef, "ef_construction": h.ef_construction},
        "global_ids": sub.global_ids.astype(np.int64),
    }
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize(path: str, dist: CountingDistance) -> SubIndex:
    with open(path, "rb") as f:
        state = pickle.load(f)
    p = state["params"]
    h = HNSW(distance_func=dist, m=p["m"], ef=p["ef"],
             ef_construction=p["ef_construction"], m0=p["m0"],
             neighborhood_construction=heuristic)
    h.data = state["data"]
    h.graphs = state["graphs"]
    h.enter_point = state["enter_point"]
    return SubIndex(hnsw=h, global_ids=np.asarray(state["global_ids"], dtype=np.int64))


# --------------------------------------------------------------------------- #
# Graph-quality diagnostics                                                    #
# --------------------------------------------------------------------------- #
def quality_stats(sub: SubIndex) -> dict:
    h = sub.hnsw
    layer0 = h.graphs[0] if h.graphs else {}
    degrees = np.array([len(nbrs) for nbrs in layer0.values()]) if layer0 else np.array([0])
    reach = _reachable_from_entry(h)
    return {
        "n_nodes": len(h.data),
        "n_layers": len(h.graphs),
        "deg_mean": float(degrees.mean()),
        "deg_p50": float(np.percentile(degrees, 50)),
        "deg_p95": float(np.percentile(degrees, 95)),
        "deg_max": int(degrees.max()),
        "frac_reachable_L0": reach,
    }


def _reachable_from_entry(h: "HNSW") -> float:
    """Fraction of layer-0 nodes reachable from the entry point via greedy edges."""
    if not h.graphs or h.enter_point is None:
        return 0.0
    g0 = h.graphs[0]
    seen = set()
    stack = [h.enter_point]
    while stack:
        v = stack.pop()
        if v in seen or v not in g0:
            continue
        seen.add(v)
        for nb, _ in g0[v]:
            if nb not in seen:
                stack.append(nb)
    return len(seen) / max(1, len(g0))
