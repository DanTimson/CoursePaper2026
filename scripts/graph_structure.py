#!/usr/bin/env python3
"""Analyse level-0 graph structure from dump_graph_level0 CSVs.

Consumes one or more `node_id,degree,neighbours` CSVs (as emitted by
dump_graph_level0.cpp) and reports, per graph:

  * out-degree distribution   — mean / median / max / fraction at the m0 cap
  * connectivity              — number of weakly-connected components (union-find
                                over the undirected support of the edge set) and
                                the largest component's share

and writes:
  * degree_distribution.png / .pdf  — overlaid degree histograms across graphs
  * graph_structure.csv             — one summary row per graph

WHY THIS EXISTS (two jobs):
  1. Cross-validation mechanism. The C++/Python merge-cost ratio (~0.4x for the
     traversal merges, steeper for NGM/SIGM) is explained by build density:
     hnswlib keeps only diversity-surviving neighbours (mean degree ~21 at
     maxM0=32), while the reference heuristic() BACK-FILLS to the cap (~32.0).
     Pass --ref-density 32.0 to draw that reference line and print the gap.
  2. The edge-heavy claim. Whether TWO_MERGE/HNSWMerger or ES produce denser or
     sparser merged graphs than INSERT was previously INFERRED from query time;
     dumping leaf / INSERT / merged graphs and running this settles it by
     measurement.

    python scripts/graph_structure.py \
        --csv c_leaf.csv c_insert.csv c_igtm.csv \
        --labels leaf INSERT IGTM \
        --ref-density 32.0 \
        --out docs/figures/structure
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """Return (degrees: np.ndarray[int], adjacency: list[list[int]])."""
    degrees, adj = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            deg = int(row[1])
            neigh = [int(x) for x in row[2].split()] if len(row) > 2 and row[2] else []
            degrees.append(deg)
            adj.append(neigh)
    return np.array(degrees, dtype=np.int64), adj


class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def connectivity(adj):
    """Weakly-connected components over the undirected support of the edges."""
    n = len(adj)
    uf = UF(n)
    for i, neigh in enumerate(adj):
        for j in neigh:
            if 0 <= j < n:          # guard against ids outside this dump
                uf.union(i, j)
    roots = {}
    for i in range(n):
        roots[uf.find(i)] = roots.get(uf.find(i), 0) + 1
    ncomp = len(roots)
    largest = max(roots.values()) if roots else 0
    return ncomp, largest


def analyse(path, label, maxM0):
    degrees, adj = load_csv(path)
    n = len(degrees)
    cap = maxM0 or int(degrees.max())
    at_cap = int((degrees >= cap).sum())
    ncomp, largest = connectivity(adj)
    return {
        "label": label,
        "path": os.path.basename(path),
        "nodes": n,
        "mean_degree": float(degrees.mean()),
        "median_degree": int(np.median(degrees)),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "cap": cap,
        "frac_at_cap": at_cap / n if n else 0.0,
        "components": ncomp,
        "largest_component_frac": largest / n if n else 0.0,
        "_degrees": degrees,
    }


def fig_degree_hist(summaries, out, ref_density=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colors = ["#2e86de", "#16a085", "#d1495b", "#8e44ad", "#e67e22", "#7f8c8d"]
    maxdeg = max(int(s["_degrees"].max()) for s in summaries)
    bins = np.arange(0, maxdeg + 2) - 0.5
    for i, s in enumerate(summaries):
        ax.hist(s["_degrees"], bins=bins, density=True, histtype="step",
                linewidth=1.8, color=colors[i % len(colors)],
                label=f"{s['label']} (mean {s['mean_degree']:.1f})")
    if ref_density:
        ax.axvline(ref_density, ls="--", color="#555", lw=1.4, alpha=0.8,
                   label=f"Python ref (back-filled, {ref_density:.0f})")
    ax.set_xlabel("out-degree at level 0")
    ax.set_ylabel("fraction of nodes")
    ax.set_title("Level-0 out-degree distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"degree_distribution.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote degree_distribution.png / .pdf")


def write_summary(summaries, out):
    os.makedirs(out, exist_ok=True)
    cols = ["label", "path", "nodes", "mean_degree", "median_degree",
            "min_degree", "max_degree", "cap", "frac_at_cap",
            "components", "largest_component_frac"]
    p = os.path.join(out, "graph_structure.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"  wrote graph_structure.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="one label per csv; defaults to filenames")
    ap.add_argument("--maxM0", type=int, default=0,
                    help="degree cap for frac-at-cap; 0 = use each graph's own max")
    ap.add_argument("--ref-density", type=float, default=None,
                    help="Python reference mean degree, drawn as a reference line")
    ap.add_argument("--out", default="docs/figures/structure")
    a = ap.parse_args(argv)

    labels = a.labels or [os.path.splitext(os.path.basename(p))[0] for p in a.csv]
    if len(labels) != len(a.csv):
        raise SystemExit(f"--labels ({len(labels)}) must match --csv ({len(a.csv)})")

    summaries = []
    for path, label in zip(a.csv, labels):
        s = analyse(path, label, a.maxM0)
        summaries.append(s)
        print(f"{label:12} n={s['nodes']:>7}  mean_deg={s['mean_degree']:6.3f}  "
              f"median={s['median_degree']:>2}  max={s['max_degree']:>2}  "
              f"at_cap={s['frac_at_cap']*100:5.2f}%  "
              f"components={s['components']}  largest={s['largest_component_frac']*100:.2f}%")

    if a.ref_density:
        print(f"\nreference (Python, back-filled): mean degree {a.ref_density:.2f}")
        for s in summaries:
            gap = a.ref_density - s["mean_degree"]
            print(f"  {s['label']:12} is {gap:+.2f} vs reference "
                  f"({s['mean_degree']/a.ref_density*100:.1f}% of its density)")

    fig_degree_hist(summaries, a.out, a.ref_density)
    write_summary(summaries, a.out)


if __name__ == "__main__":
    main()
