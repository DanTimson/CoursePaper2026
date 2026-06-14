"""Divide-and-conquer construction by pairwise merge.

Leaves are HNSWs built on partitions. Internal nodes merge two child indices
with one of NGM / IGTM / CGTM. The driver supports two merge orders:

  sequential : (((p0 . p1) . p2) . p3)  — left fold
  balanced   : ((p0 . p1) . (p2 . p3))  — binary tree (parallelizable)

For each pairwise merge we relabel B's ids to [nA, nA+nB) so the combined local
id-space is contiguous from zero; this is required for CGTM and harmless for the
others. The merged index's global_ids is the concatenation of the children's.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..distance import CountingDistance
from ..vendor_api import MERGE_FUNCS
from .base import SubIndex, shift_ids


def merge_pair(
    a: SubIndex, b: SubIndex, algo: str, dist: CountingDistance, merge_params: dict
) -> SubIndex:
    nA = a.n
    b_shifted = shift_ids(b, nA)                       # B ids -> [nA, nA+nB)
    merged_data = {**a.hnsw.data, **b_shifted.hnsw.data}
    fn = MERGE_FUNCS[algo]
    merged_hnsw = fn(a.hnsw, b_shifted.hnsw, merged_data, **merge_params)
    global_ids = np.concatenate([a.global_ids, b.global_ids])
    return SubIndex(hnsw=merged_hnsw, global_ids=global_ids)


def _reduce(
    leaves: List[SubIndex], algo: str, dist: CountingDistance,
    merge_params: dict, order: str,
) -> SubIndex:
    if len(leaves) == 1:
        return leaves[0]
    if order == "sequential":
        acc = leaves[0]
        for nxt in leaves[1:]:
            acc = merge_pair(acc, nxt, algo, dist, merge_params)
        return acc
    if order == "balanced":
        level = leaves
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    nxt.append(merge_pair(level[i], level[i + 1], algo, dist, merge_params))
                else:
                    nxt.append(level[i])
            level = nxt
        return level[0]
    raise ValueError(f"unknown merge order: {order!r}")


def divide_and_conquer(
    leaves: List[SubIndex],
    algo: str,
    dist: CountingDistance,
    merge_params: dict,
    order: str = "balanced",
) -> SubIndex:
    """Merge a list of leaf sub-indices into one. Returns the root SubIndex.

    Distance counts accrue into ``dist``; snapshot it around this call to get the
    merge-phase distance count.
    """
    return _reduce(leaves, algo, dist, merge_params, order)
