#!/usr/bin/env python3
"""Derived-quantity analyses over the merge sweep. Reads the raw per-scale JSONL
logs (which carry `dataset`, unlike summary.csv) plus the optional structural
density CSV, and emits three analyses + figures:

  1. lambda frontier   — HNSWMerger cost vs d_s across lambda, with marginal
                          returns (delta d_s per extra billion merge-computations)
                          -> quantifies "why lambda=4", diminishing returns.
  2. efc sensitivity    — same strategy at different ef_construction: how much
                          merge cost vs search quality (d_s) each efc buys.
  3. density vs d_s      — join structural mean-degree (graph_structure.csv) to
                          each strategy's d_s@target -> does denser graph search
                          cheaper, and is any strategy sparse-but-competitive.


    python scripts/analyse_trends.py \
        --results results/bigann10k.jsonl results/bigann100k.jsonl \
                  results/bigann1m.jsonl  results/bigann10m.jsonl \
        --density docs/figures/structure/graph_structure.csv \
        --target 0.95 --out docs/figures/trends

Writes lambda_frontier.png, efc_sensitivity.png (if >1 efc), density_vs_ds.png
(if --density), and trends.csv with every derived number. Scale comparison of
merge strategies lives in make_figures.py (merge_strategies_grid, scale_trend).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STRAT_LABEL = {"TWO_MERGE": "HNSWMerger", "IGTM": "IGTM", "CGTM": "CGTM",
               "NGM": "NGM", "SIGM": "SIGM", "INSERT": "Rebuild"}
COLORS = {"NGM": "#d1495b", "IGTM": "#2e86de", "CGTM": "#16a085",
          "TWO_MERGE": "#8e44ad", "INSERT": "#7f8c8d", "SIGM": "#34495e"}


def load(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skip missing {p})"); continue
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def scale_of(name):
    m = re.search(r"(\d+)\s*([km]?)$", (name or "").lower())
    if not m:
        return None
    return int(m.group(1)) * {"k": 1_000, "m": 1_000_000, "": 1}[m.group(2)]


def ds_at(r, target):
    cur = [(c.get("recall"), c.get("d_s")) for c in (r.get("recall_curve") or [])
           if c.get("recall") is not None and c.get("d_s") is not None]
    if len(cur) < 2:
        return None
    cur.sort()
    recs, dss = [c[0] for c in cur], [c[1] for c in cur]
    if target < recs[0] or target > recs[-1]:
        return None
    import bisect
    i = bisect.bisect_left(recs, target)
    if i == 0:
        return dss[0]
    r0, r1, d0, d1 = recs[i-1], recs[i], dss[i-1], dss[i]
    return d1 if r1 == r0 else d0 + (d1 - d0) * (target - r0) / (r1 - r0)


def _p(r, key, default=None):
    return (r.get("params") or {}).get(key, default)


def _efc(r):
    """ef_construction lives at the row top level (harness writes it there), not
    in params/merge_id. Fall back to params, then to 200."""
    v = r.get("ef_construction")
    if v is None:
        v = (r.get("params") or {}).get("ef_construction")
    return 200 if v is None else v


def _dss(r, target):
    """d_s@target from curve, else fall back to a stored d_s@0.95 field."""
    v = ds_at(r, target)
    if v is not None:
        return v
    return r.get("d_s@0.95")


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ---------- 1. lambda frontier -------------------------------------------------
def lambda_frontier(rows, out, target, csv_rows, ds_name):
    # one row per lambda at efc=200; dedup defensively (a stray efc-defaulted
    # duplicate lambda would otherwise create a lamX->lamX step).
    best = {}
    for r in rows:
        if r.get("algo") != "TWO_MERGE" or r.get("n_parts") != 2:
            continue
        if _efc(r) != 200:
            continue
        lam = _p(r, "merge_lambda")
        d, mc = _dss(r, target), r.get("merge_calc")
        if lam is None or not d or not mc:
            continue
        # if two rows share a lambda, keep the cheaper merge (the canonical one)
        prev = best.get(lam)
        if prev is None or mc < prev[0]:
            best[lam] = (mc, float(d))
    pts = sorted((lam, mc / 1e9, d) for lam, (mc, d) in best.items())
    if len(pts) < 2:
        print("  (skip lambda_frontier: <2 distinct lambda points)"); return
    lam = [p[0] for p in pts]; cost = [p[1] for p in pts]; dsv = [p[2] for p in pts]

    # marginal returns: delta d_s per extra billion merge-computations
    print("\n  lambda frontier (higher lambda = costlier merge, better graph):")
    for i, (l, c, d) in enumerate(pts):
        marg = ""
        if i > 0:
            dcost = c - cost[i-1]
            dds = dsv[i-1] - d           # positive = d_s dropped (improved)
            rate = dds / dcost if dcost else float("nan")
            marg = f"   d(d_s)/d(cost) = {rate:8.1f} per +1B  (cost +{dcost:.2f}B, d_s {dds:+.0f})"
            csv_rows.append({"analysis": "lambda_frontier", "dataset": ds_name,
                             "x": f"lam{lam[i-1]}->lam{l}", "value": rate,
                             "detail": f"dcost={dcost:.3f}B d_ds={dds:.1f}"})
        print(f"    lambda={l:<3} cost={c:6.2f}B  d_s@{target}={d:7.1f}{marg}")

    fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
    ax1.plot(lam, cost, "o-", color=COLORS["TWO_MERGE"], label="merge cost")
    ax1.set_xlabel("HNSWMerger lambda")
    ax1.set_ylabel("merge cost (billions)", color=COLORS["TWO_MERGE"])
    ax1.tick_params(axis="y", labelcolor=COLORS["TWO_MERGE"])
    ax1.set_xticks(lam)
    ax2 = ax1.twinx()
    ax2.plot(lam, dsv, "s--", color="#2c3e50", label=f"d_s@{target}")
    ax2.set_ylabel(f"search dist/query @ recall {target}", color="#2c3e50")
    ax2.tick_params(axis="y", labelcolor="#2c3e50")
    ax2.invert_yaxis()   # lower d_s (better) at top, so both curves "improve" upward
    ax2.spines["top"].set_visible(False)
    ax1.set_title(f"HNSWMerger lambda: cost vs search quality ({ds_name})")
    _save(fig, out, "lambda_frontier")


# ---------- 2. efc sensitivity -------------------------------------------------
def efc_sensitivity(rows, out, target, csv_rows, ds_name):
    """Same strategy across ef_construction. Report cost + d_s per efc."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    any_multi = False
    for algo in ["TWO_MERGE", "IGTM", "CGTM"]:
        sel = [r for r in rows if r.get("algo") == algo and r.get("n_parts") == 2]
        byefc = {}
        for r in sel:
            efc = _p(r, "ef_construction", 200)
            d = _dss(r, target); mc = r.get("merge_calc")
            if mc:
                byefc.setdefault(efc, []).append((mc / 1e9, d))
        if len(byefc) < 2:
            continue
        any_multi = True
        efcs = sorted(byefc)
        costs = [min(v)[0] for v in (byefc[e] for e in efcs)]
        ax.plot(efcs, costs, "o-", color=COLORS.get(algo, "#555"), label=STRAT_LABEL[algo])
        for e in efcs:
            c, d = min(byefc[e])
            print(f"  {STRAT_LABEL[algo]:10} efc={e:<4} cost={c:6.3f}B  d_s@{target}={d}")
            csv_rows.append({"analysis": "efc", "dataset": ds_name,
                             "x": f"{algo}_efc{e}", "value": c,
                             "detail": f"d_s={d}"})
    if not any_multi:
        print("  (skip efc_sensitivity: no strategy has >1 efc)"); plt.close(fig); return
    ax.set_xlabel("ef_construction")
    ax.set_ylabel("merge cost (billions)")
    ax.set_title(f"Merge cost vs ef_construction ({ds_name})")
    ax.legend(fontsize=9)
    _save(fig, out, "efc_sensitivity")


# ---------- 3. density vs d_s --------------------------------------------------
def density_vs_ds(rows, out, target, density_csv, csv_rows, ds_name):
    if not density_csv or not os.path.exists(density_csv):
        print("  (skip density_vs_ds: no --density csv)"); return
    dens = {}
    for r in csv.DictReader(open(density_csv)):
        dens[r["label"].upper()] = float(r["mean_degree"])
    # join by strategy: canonical (cheapest) config per algo
    pts = []
    for algo in ["TWO_MERGE", "IGTM", "CGTM", "NGM", "SIGM"]:
        key = STRAT_LABEL[algo].upper()
        # density csv may label HNSWMerger or TWO_MERGE; try both
        d_val = dens.get(key) or dens.get(algo)
        if d_val is None:
            continue
        cand = [(_dss(r, target), r) for r in rows
                if r.get("algo") == algo and r.get("n_parts") == 2]
        cand = [(d, r) for d, r in cand if d is not None]
        if not cand:
            continue
        dsv = min(cand)[0]
        pts.append((algo, d_val, float(dsv)))
        csv_rows.append({"analysis": "density_vs_ds", "dataset": ds_name,
                         "x": algo, "value": d_val, "detail": f"d_s={dsv:.1f}"})
    if len(pts) < 2:
        print("  (skip density_vs_ds: <2 joined points)"); return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for algo, deg, dsv in pts:
        ax.scatter(deg, dsv, s=110, color=COLORS.get(algo, "#555"), zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(STRAT_LABEL[algo], (deg, dsv), textcoords="offset points",
                    xytext=(7, 4), fontsize=9)
    ax.set_xlabel("merged-graph mean out-degree (level 0)")
    ax.set_ylabel(f"search dist/query @ recall {target}")
    ax.set_title(f"Graph density vs search cost ({ds_name})")
    print("\n  density vs d_s (denser graph should search cheaper -> lower d_s):")
    for algo, deg, dsv in sorted(pts, key=lambda p: p[1]):
        print(f"    {STRAT_LABEL[algo]:10} degree={deg:5.2f}  d_s@{target}={dsv:7.1f}")
    _save(fig, out, "density_vs_ds")



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--density", default=None)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--out", default="docs/figures/trends")
    a = ap.parse_args(argv)

    rows = load(a.results)
    if not rows:
        print("no rows"); return
    csv_rows = []

    # per-scale analyses use the largest scale present (richest sweep)
    by_scale = {}
    for r in rows:
        N = scale_of(r.get("dataset"))
        if N:
            by_scale.setdefault(N, []).append(r)
    big = max(by_scale) if by_scale else None
    big_rows = by_scale.get(big, rows)
    big_name = f"N={big:,}" if big else "data"

    print("== 1. lambda frontier ==")
    lambda_frontier(big_rows, a.out, a.target, csv_rows, big_name)
    print("\n== 2. efc sensitivity ==")
    efc_sensitivity(big_rows, a.out, a.target, csv_rows, big_name)
    print("\n== 3. density vs d_s ==")
    density_vs_ds(big_rows, a.out, a.target, a.density, csv_rows, big_name)


    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "trends.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["analysis", "dataset", "x", "value", "detail"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwrote {os.path.join(a.out, 'trends.csv')}")


if __name__ == "__main__":
    main()