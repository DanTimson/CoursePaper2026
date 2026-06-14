"""Generate publication figures from the results logs.

    python scripts/make_figures.py --results results_cpp.jsonl results_sift.jsonl --out docs/figures

Reads one or more JSONL result logs, applies the known corrections, and writes
PNG+PDF figures plus a summary CSV. Corrections (see README "data hazards"):

  * build_calc imputation — HNSWMerger reuses leaf indexes across algorithms at
    the same partition count, so only the first algorithm to run records the
    build cost; the rest log build_calc=0. We fill each row's build_calc from the
    (single) non-zero value at its partition count, and recompute total_calc.
    (Newer runs carry this forward via a sidecar and won't need imputing.)
  * TWO_MERGE merge_calc=0 — its merge isn't routed through the distance counter,
    so it's dropped from distance-count plots (kept for time/recall).
  * INSERT/baseline recall may be null on older runs (no query test on the
    single-index path); such rows are skipped where recall is required.

The merge-algorithm cost comparison uses merge_calc, not total_calc, because the
leaf-build cost is shared across algorithms at a given partition count and would
otherwise swamp the per-algorithm signal.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NQ = 10000  # SIFT query-set size, for QPS = NQ / query_seconds
MERGE_ALGOS = ["NGM", "IGTM", "CGTM"]
COLORS = {"NGM": "#d1495b", "IGTM": "#2e86de", "CGTM": "#16a085",
          "ES": "#e67e22", "TWO_MERGE": "#8e44ad", "INSERT": "#7f8c8d",
          "SIGM": "#7f8c8d", "NNDescent": "#27ae60"}
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(paths):
    rows, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skip missing {p})"); continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            key = r.get("run_key") or json.dumps(r, sort_keys=True)
            if key in seen:
                continue
            seen.add(key); rows.append(r)
    return rows


def correct(rows):
    # impute build_calc and build_seconds per (dataset, n_parts) for hnswmerger rows
    build_by, secs_by = {}, {}
    for r in rows:
        if r.get("builder") == "hnswmerger" and (r.get("build_calc") or 0) > 0:
            build_by[(r.get("dataset"), r.get("n_parts"))] = r["build_calc"]
            secs_by[(r.get("dataset"), r.get("n_parts"))] = r.get("build_seconds") or 0
    for r in rows:
        if r.get("builder") == "hnswmerger":
            key = (r.get("dataset"), r.get("n_parts"))
            if not (r.get("build_calc") or 0):
                r["build_calc"] = build_by.get(key, 0)
            if not (r.get("build_seconds") or 0):
                r["build_seconds"] = secs_by.get(key, 0)
        bc, mc = r.get("build_calc") or 0, r.get("merge_calc") or 0
        r["total_calc"] = bc + mc
        # TWO_MERGE has no usable distance count
        if r.get("algo") == "TWO_MERGE" and not mc:
            r["merge_calc_plot"] = math.nan
        else:
            r["merge_calc_plot"] = mc
    return rows


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def fig_merge_cost(rows, out):
    parts = sorted({r["n_parts"] for r in rows
                    if r.get("builder") == "hnswmerger" and r["algo"] in MERGE_ALGOS})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.8 / max(1, len(MERGE_ALGOS))
    for ai, algo in enumerate(MERGE_ALGOS):
        ys = []
        for p in parts:
            m = [r["merge_calc"] for r in rows if r.get("algo") == algo
                 and r.get("n_parts") == p and r.get("builder") == "hnswmerger"]
            ys.append((m[0] / 1e9) if m else 0)
        xs = [i + ai * w for i in range(len(parts))]
        ax.bar(xs, ys, w, label=algo, color=COLORS[algo])
    ax.set_xticks([i + w for i in range(len(parts))])
    ax.set_xticklabels([f"{p} partitions" for p in parts])
    ax.set_ylabel("merge-phase distance computations  (billions)")
    ax.set_title("Merge cost by algorithm and partition count (SIFT1M)")
    ax.legend(title="algorithm")
    _save(fig, out, "merge_cost")


def fig_partition_scaling(rows, out):
    parts = sorted({r["n_parts"] for r in rows
                    if r.get("builder") == "hnswmerger" and r["algo"] in MERGE_ALGOS})
    fig, (axc, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
    for algo in MERGE_ALGOS:
        mc = [next((r["merge_calc"] / 1e9 for r in rows if r.get("algo") == algo
                    and r.get("n_parts") == p), None) for p in parts]
        rc = [next((r.get("recall@10") for r in rows if r.get("algo") == algo
                    and r.get("n_parts") == p), None) for p in parts]
        axc.plot(parts, mc, "o-", color=COLORS[algo], label=algo)
        axr.plot(parts, rc, "o-", color=COLORS[algo], label=algo)
    # shared build cost (decreasing with partition count) on the cost panel
    bc = [next((r["build_calc"] / 1e9 for r in rows if r.get("n_parts") == p
                and r.get("builder") == "hnswmerger" and (r.get("build_calc") or 0) > 0), None)
          for p in parts]
    axc.plot(parts, bc, "k--", marker="s", label="build (shared)", alpha=0.6)
    axc.set_xticks(parts); axc.set_xlabel("partitions")
    axc.set_ylabel("distance computations (billions)")
    axc.set_title("Cost vs partition count"); axc.legend()
    axr.set_xticks(parts); axr.set_xlabel("partitions")
    axr.set_ylabel("recall@10 (best ef)")
    axr.set_title("Recall vs partition count"); axr.legend()
    fig.suptitle("Divide-and-conquer trades more (parallelizable) build for more merge cost and slight recall loss",
                 fontsize=11)
    _save(fig, out, "partition_scaling")


def fig_recall_vs_qps(rows, out, n_parts=2):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    methods = [r for r in rows if r.get("builder") == "hnswmerger"
               and r.get("n_parts") == n_parts and r.get("recall_curve")]
    order = ["IGTM", "CGTM", "NGM", "ES", "TWO_MERGE"]
    methods.sort(key=lambda r: order.index(r["algo"]) if r["algo"] in order else 99)
    for r in methods:
        pts = [(NQ / c["query_seconds"], c["recall"]) for c in r["recall_curve"]
               if c.get("query_seconds") and c.get("recall") is not None]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, "o-", color=COLORS.get(r["algo"], "#555"), label=r["algo"])
    # NN-Descent point(s) if present (single search setting)
    for r in rows:
        if r.get("builder") == "nndescent" and r.get("recall@10") is not None:
            ax.scatter([], [])  # placeholder; QPS unknown unless recorded
    ax.set_xscale("log")
    ax.set_xlabel("queries / second  (log)")
    ax.set_ylabel("recall@10")
    ax.set_title(f"Search quality vs speed at {n_parts} partitions (SIFT1M)")
    ax.legend(title="method")
    _save(fig, out, "recall_vs_qps")


def fig_construction_time(rows, out, n_parts=2):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sel = [r for r in rows if r.get("builder") == "hnswmerger"
           and (r.get("n_parts") == n_parts or r.get("algo") in ("INSERT", "REBUILD"))]
    order = {"INSERT": 0, "IGTM": 1, "CGTM": 2, "ES": 3, "NGM": 4, "TWO_MERGE": 5}
    sel.sort(key=lambda r: order.get(r["algo"], 99))
    names = [r["algo"] for r in sel]
    builds = [(r.get("build_seconds") or 0) for r in sel]
    merges = [r.get("merge_seconds") or 0 for r in sel]
    x = range(len(sel))
    ax.bar(x, builds, color="#bdc3c7", label="build (leaves, shared/parallelizable)")
    ax.bar(x, merges, bottom=builds, color="#e67e22", label="merge")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("construction wall-clock (s)")
    ax.set_title(f"Construction time at {n_parts} partitions (SIFT1M)")
    top = max((b + m) for b, m in zip(builds, merges)) if sel else 1
    ax.set_ylim(0, top * 1.22)
    ax.legend(loc="upper center", ncol=2, frameon=True)
    _save(fig, out, "construction_time")


def write_summary(rows, out):
    os.makedirs(out, exist_ok=True)
    cols = ["builder", "algo", "n_parts", "build_calc", "merge_calc", "total_calc",
            "build_seconds", "merge_seconds", "recall@10"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (str(r.get("algo")), r.get("n_parts", 0))):
            w.writerow(r)
    print("  wrote summary.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+",
                    default=["results_cpp.jsonl", "results_sift.jsonl", "results.jsonl"])
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)
    rows = correct(load(a.results))
    if not rows:
        print("no rows found"); return
    print(f"{len(rows)} rows -> {a.out}")
    fig_merge_cost(rows, a.out)
    fig_partition_scaling(rows, a.out)
    fig_recall_vs_qps(rows, a.out)
    fig_construction_time(rows, a.out)
    write_summary(rows, a.out)
    if not any(r.get("builder") == "nndescent" for r in rows):
        print("  note: no NN-Descent rows found — run config/sift1m.json (nndescent) "
              "to add it to the comparison.")


if __name__ == "__main__":
    main()
