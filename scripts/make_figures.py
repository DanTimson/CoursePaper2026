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

NQ_BY_DATASET = {"sift1m": 10000, "gist1m": 1000}  # query-set sizes, for QPS = nq / query_seconds


def nq_for(row):
    return row.get("nq") or NQ_BY_DATASET.get(row.get("dataset"), 10000)


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
    # last occurrence of a run_key wins (latest run supersedes earlier ones),
    # keeping the position of first appearance for stable ordering.
    by_key = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skip missing {p})"); continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            key = r.get("run_key") or json.dumps(r, sort_keys=True)
            by_key[key] = r
    return list(by_key.values())


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


def fig_merge_cost(rows, out, ds="SIFT1M"):
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
    ax.set_title(f"Merge cost by algorithm and partition count ({ds})")
    ax.legend(title="algorithm")
    _save(fig, out, "merge_cost")


def fig_partition_scaling(rows, out, ds="SIFT1M"):
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
    fig.suptitle(f"Divide-and-conquer trades more (parallelizable) build for more merge cost and slight recall loss  ({ds})",
                 fontsize=11)
    _save(fig, out, "partition_scaling")


def fig_recall_vs_qps(rows, out, ds="SIFT1M", n_parts=2):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    methods = [r for r in rows if r.get("builder") == "hnswmerger"
               and r.get("n_parts") == n_parts and r.get("recall_curve")]
    order = ["IGTM", "CGTM", "NGM", "ES", "TWO_MERGE"]
    methods.sort(key=lambda r: order.index(r["algo"]) if r["algo"] in order else 99)
    for r in methods:
        pts = [(nq_for(r) / c["query_seconds"], c["recall"]) for c in r["recall_curve"]
               if c.get("query_seconds") and c.get("recall") is not None]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, "o-", color=COLORS.get(r["algo"], "#555"), label=r["algo"])
    # NN-Descent curve if present (epsilon sweep with query_seconds)
    for r in rows:
        if r.get("builder") == "nndescent" and r.get("recall_curve"):
            pts = [(nq_for(r) / c["query_seconds"], c["recall"]) for c in r["recall_curve"]
                   if c.get("query_seconds") and c.get("recall") is not None]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, "s--", color=COLORS["NNDescent"], label="NN-Descent (flat k-NN)")
    ax.set_xscale("log")
    ax.set_xlabel("queries / second  (log)")
    ax.set_ylabel("recall@10")
    ax.set_title(f"Search quality vs speed at {n_parts} partitions ({ds})")
    ax.legend(title="method")
    _save(fig, out, "recall_vs_qps")


def fig_construction_time(rows, out, ds="SIFT1M", n_parts=2):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sel = [r for r in rows if (r.get("builder") == "hnswmerger"
           and (r.get("n_parts") == n_parts or r.get("algo") in ("INSERT", "REBUILD")))
           or r.get("builder") == "nndescent"]
    order = {"INSERT": 0, "NNDescent": 1, "IGTM": 2, "CGTM": 3, "ES": 4, "NGM": 5, "TWO_MERGE": 6}
    sel.sort(key=lambda r: order.get(r["algo"], 99))
    names = [("NN-Descent" if r["algo"] == "NNDescent" else r["algo"]) for r in sel]
    builds = [(r.get("build_seconds") or 0) for r in sel]
    merges = [r.get("merge_seconds") or 0 for r in sel]
    x = range(len(sel))
    ax.bar(x, builds, color="#bdc3c7", label="build  (shared leaves for merge; full build for INSERT / NN-Descent)")
    ax.bar(x, merges, bottom=builds, color="#e67e22", label="merge")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("construction wall-clock (s)")
    ax.set_title(f"Construction time at {n_parts} partitions ({ds})")
    top = max((b + m) for b, m in zip(builds, merges)) if sel else 1
    ax.set_ylim(0, top * 1.28)
    ax.legend(loc="upper center", ncol=1, frameon=True, fontsize=9)
    ax.text(0.5, -0.30, "NN-Descent is Numba-Python; the merge family is C++ — wall-clock is a ballpark, not a controlled comparison.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    _save(fig, out, "construction_time")


def fig_recall_vs_buildtime(rows, out, ds="SIFT1M", n_parts=2):
    """Cross-method overview: total construction cost vs achieved recall@10."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pts = []
    for r in rows:
        b = r.get("builder")
        if b == "hnswmerger" and r.get("recall@10") is not None and (
                r.get("n_parts") == n_parts or r.get("algo") in ("INSERT", "REBUILD")):
            t = (r.get("build_seconds") or 0) + (r.get("merge_seconds") or 0)
            kind = "baseline" if r["algo"] in ("INSERT", "REBUILD") else "merge"
            pts.append((r["algo"], t, r["recall@10"], kind))
        elif b == "nndescent" and r.get("recall@10") is not None:
            pts.append(("NNDescent", (r.get("build_seconds") or 0), r["recall@10"], "nndescent"))
    for algo, t, rec, kind in pts:
        marker = {"nndescent": "D", "baseline": "s"}.get(kind, "o")
        label = "NN-Descent" if algo == "NNDescent" else algo
        ax.scatter(t, rec, s=110, marker=marker, color=COLORS.get(algo, "#555"), zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(label, (t, rec), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlabel("total construction wall-clock (s)")
    ax.set_ylabel("recall@10  (best query effort)")
    ax.set_title(f"Construction cost vs achieved recall, {n_parts} partitions ({ds})")
    ax.text(0.5, -0.32,
            "Caveat: merge recall is best-ef on a navigable HNSW; NN-Descent recall is best-epsilon on a flat k-NN graph "
            "(\u2260 controlled).\nNN-Descent is Numba-Python vs C++ merge — time is a ballpark.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    _save(fig, out, "recall_vs_buildtime")


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
                    default=["results_cpp.jsonl", "results_sift.jsonl",
                             "results_gist_cpp.jsonl", "results_gist.jsonl", "results.jsonl"])
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)
    rows = correct(load(a.results))
    if not rows:
        print("no rows found"); return

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r.get("dataset") or "unknown"].append(r)

    for ds_key, ds_rows in sorted(by_ds.items()):
        ds = ds_key.upper()
        out = os.path.join(a.out, ds_key)
        print(f"{ds_key}: {len(ds_rows)} rows -> {out}")
        fig_merge_cost(ds_rows, out, ds)
        fig_partition_scaling(ds_rows, out, ds)
        fig_recall_vs_qps(ds_rows, out, ds)
        fig_construction_time(ds_rows, out, ds)
        fig_recall_vs_buildtime(ds_rows, out, ds)
        write_summary(ds_rows, out)
        if not any(r.get("builder") == "nndescent" for r in ds_rows):
            print(f"  note: no NN-Descent rows for {ds_key} — "
                  f"run config/{ds_key}.json (nndescent) to add it.")


if __name__ == "__main__":
    main()
