#!/usr/bin/env python3
"""Cross-validate the C++ HNSWMerger pipeline against the supervisor's pure-Python
reference (github.com/aponom84/merging-navigable-graphs), on identical data.

WHY: the two build DIFFERENTLY. hnswlib's getNeighborsByHeuristic2 keeps only the
candidates that survive the diversity test (often < M) and, when a point has < M
candidates, skips pruning entirely; the reference heuristic() in hnsw.py BACK-FILLS
the neighbour list up to k with the nearest rejected candidates. So the reference
builds DENSER graphs. Merge cost scales with input edge density, so the C++ numbers
are expected to sit below a Python-built reference - this harness measures by how
much, and whether the algorithm ORDERING is preserved.

This is deliberately run at small N (10k) where the Python build is tractable
(seconds); at 1M it would be hours. It is single-threaded and light on RAM, so it
coexists with a running C++ sweep.

    python xval_python_ref.py \
        --ref /path/to/merging-navigable-graphs \
        --base data/sift_scales/sift10k_base.fvecs \
        --query data/sift_scales/sift_query.fvecs \
        --gt data/sift_scales/sift10k_gt.ivecs \
        --out xval_10k.json

Then run the SAME (dataset, params) through the C++ harness and compare
'merge_distance_count' and the d_s curve.

UPSTREAM DISCIPLINE: every merge is his real function from merge_hnsw.py -
insertion_merge (SIGM), merge_naive (NGM), IGTM, CGTM - and the build is his
HNSW.add / heuristic and recall his calculate_recall. Nothing is reimplemented;
this driver only stages data and params. (Earlier version hand-rolled SIGM as
build+.add - wrong: insertion_merge IS his SIGM, and it deepcopies hnsw_a then
inserts b, so the count is insertion cost alone.)

Reports, per algorithm:
  build density  - mean neighbours/node at level 0 in the Python graphs
  merge cost     - total distance computations in the merge phase
  d_s @ ef       - search distance computations per query, at matched recall
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="clone of merging-navigable-graphs")
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-graphs", default=None, help="pickle the two built graphs here")
    ap.add_argument("--load-graphs", default=None, help="reuse graphs from a previous --save-graphs")
    ap.add_argument("--sweep", default=None,
                    help="JSON file: {algo: [param-dict, ...]} to run many configs on ONE build")
    ap.add_argument("--n", type=int, default=10000, help="base vectors to use (build is O(N log N) in PY)")
    ap.add_argument("--nq", type=int, default=1000, help="queries (cap for speed)")
    # match these to the C++ run you compare against
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--m0", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--efs", type=int, nargs="+", default=[10, 50, 100, 200, 400])
    # merge params (defaults = the j5,l7 point; override per comparison)
    ap.add_argument("--jump-ef", type=int, default=5)
    ap.add_argument("--local-ef", type=int, default=7)
    ap.add_argument("--next-step-k", type=int, default=3)
    ap.add_argument("--next-step-ef", type=int, default=3)
    ap.add_argument("--search-M", type=int, default=5)
    ap.add_argument("--search-ef", type=int, default=10, help="NGM (merge_naive) merge_ef")
    ap.add_argument("--sigm-efc", type=int, default=200, help="SIGM insertion ef_construction")
    ap.add_argument("--algos", nargs="+", default=["NGM", "IGTM", "CGTM", "SIGM"])
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.ref))
    # the reference's own modules
    import hnsw as refhnsw          # noqa: E402
    import merge_hnsw as refmerge   # noqa: E402
    from datasets import read_fvecs, read_ivecs, calculate_recall  # noqa: E402

    # ---- instrument distance: wrap the reference l2 with a global counter, and
    #      inject it as the distance_func so BUILD, MERGE and SEARCH all count.
    counter = {"n": 0}
    def l2(x, y):
        counter["n"] += 1
        d = x - y
        return float(np.dot(d, d))   # squared L2; ordering identical to hnswlib's L2Sqr

    base = np.array(list(read_fvecs(a.base)), dtype=np.float32)[:a.n]
    query = np.array(list(read_fvecs(a.query)), dtype=np.float32)[:a.nq]
    gt = np.array(list(read_ivecs(a.gt)), dtype=np.int32)[:a.nq]
    N = len(base)
    half = N // 2
    print(f"N={N}  half={half}  queries={len(query)}  ef_construction={a.ef_construction}")

    def build(vectors, id_offset):
        """Build a reference HNSW; return it plus a data dict keyed by GLOBAL id."""
        h = refhnsw.HNSW(distance_func=l2, m=a.M, m0=a.m0, ef=32,
                         ef_construction=a.ef_construction,
                         neighborhood_construction=refhnsw.heuristic)
        for i, v in enumerate(vectors):
            h.add(id_offset + i, v)
        return h

    def level0_density(h):
        g0 = h.graphs[0]
        degs = [len(v) for v in g0.values()]
        return float(np.mean(degs)), int(np.max(degs))

    import pickle
    t0 = time.time()
    if a.load_graphs and os.path.exists(a.load_graphs):
        hnsw_a, hnsw_b, build_calls = pickle.load(open(a.load_graphs, "rb"))
        # rebind the counting distance (pickle lost the closure)
        for h in (hnsw_a, hnsw_b):
            h.distance_func = l2
        print(f"loaded graphs from {a.load_graphs}  build_dist={build_calls:,}")
    else:
        counter["n"] = 0
        hnsw_a = build(base[:half], 0)
        hnsw_b = build(base[half:], half)
        build_calls = counter["n"]
        if a.save_graphs:
            for h in (hnsw_a, hnsw_b): h.distance_func = None   # closures don't pickle
            pickle.dump((hnsw_a, hnsw_b, build_calls), open(a.save_graphs, "wb"))
            for h in (hnsw_a, hnsw_b): h.distance_func = l2
            print(f"saved graphs to {a.save_graphs}")
    dens_a = level0_density(hnsw_a)
    dens_b = level0_density(hnsw_b)
    print(f"built 2x{half} in {time.time()-t0:.1f}s  build_dist={build_calls:,}  "
          f"L0 density a={dens_a[0]:.2f} b={dens_b[0]:.2f} (max cap {a.m0})")

    merged_data = {i: base[i] for i in range(N)}

    gt_list = [row[:a.k].tolist() for row in gt]
    def recall_at(hm, ef):
        counter["n"] = 0
        rec = calculate_recall(hm, list(query), groundtruth=gt_list, k=a.k, ef=ef)
        # calculate_recall may return a scalar or (recall, curve); normalise
        rec = rec[0] if isinstance(rec, tuple) else rec
        return float(rec), counter["n"] / len(query)

    results = {"config": vars(a), "N": N,
               "build": {"distance_count": build_calls,
                         "l0_density_a": dens_a[0], "l0_density_b": dens_b[0]},
               "algos": {}}

    from copy import deepcopy
    import random as _random

    def one_merge(algo, mp, ga, gb):
        if algo == "NGM":
            return refmerge.merge_naive(ga, gb, merged_data,
                                        merge_ef=mp.get("search_ef", a.search_ef))
        if algo == "IGTM":
            return refmerge.IGTM(ga, gb, merged_data,
                                 jump_ef=mp.get("jump_ef", a.jump_ef),
                                 local_ef=mp.get("local_ef", a.local_ef),
                                 next_step_k=mp.get("next_step_k", a.next_step_k),
                                 next_step_ef=mp.get("next_step_ef", a.next_step_ef),
                                 M=mp.get("search_M", a.search_M))
        if algo == "CGTM":
            return refmerge.CGTM(ga, gb, merged_data,
                                 jump_ef=mp.get("jump_ef", a.jump_ef),
                                 local_ef=mp.get("local_ef", a.local_ef),
                                 next_step_k=mp.get("next_step_k", a.next_step_k),
                                 M=mp.get("search_M", a.search_M))
        if algo == "SIGM":
            efc = mp.get("merge_ef_construction", -1)
            efc = efc if efc and efc > 0 else a.sigm_efc
            return refmerge.insertion_merge(ga, gb, ef_construction=efc)
        return None

    if a.sweep:
        jobs = [(algo, mp) for algo, plist in json.load(open(a.sweep)).items() for mp in plist]
    else:
        jobs = [(algo, {}) for algo in a.algos]

    for algo, mp in jobs:
        # independent copies per config -> order-independent and reproducible
        ga, gb = deepcopy(hnsw_a), deepcopy(hnsw_b)
        for _h in (ga, gb):
            _h.distance_func = l2
        _random.seed(42)
        counter["n"] = 0
        t = time.time()
        hm = one_merge(algo, mp, ga, gb)
        if hm is None:
            print(f"  skip {algo}: no reference entry point"); continue
        merge_calls = counter["n"]
        dens = level0_density(hm)
        curve = []
        for ef in a.efs:
            rec, ds = recall_at(hm, ef)
            curve.append({"ef": ef, "recall": rec, "d_s": ds})
        pid = ",".join(f"{k}{v}" for k,v in sorted(mp.items()) if v!=-1) or "default"
        results["algos"][f"{algo}|{pid}"] = {"algo": algo, "params": mp,
                                  "merge_distance_count": merge_calls,
                                  "l0_density": dens[0],
                                  "seconds": time.time() - t,
                                  "curve": curve}
        print(f"{algo:6} {pid:24}: merge_dist={merge_calls:,}  L0 density={dens[0]:.2f}  "
              f"({time.time()-t:.1f}s)  " +
              " ".join(f"ef{c['ef']}:r{c['recall']:.3f}/d{c['d_s']:.0f}" for c in curve))

    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("Compare 'merge_distance_count' and the d_s curve against the C++ run at "
          "the SAME (N, params). Agreement within a few % => the build-density gap "
          "washes out; a 20-30% gap => report C++ numbers as HNSWlib-construction, "
          "not reproduction of the Python reference.")


if __name__ == "__main__":
    main()
