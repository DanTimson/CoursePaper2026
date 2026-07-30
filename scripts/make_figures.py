"""Generate publication figures from the results logs.

    python scripts/make_figures.py --results results/bigann10k.jsonl results/bigann100k.jsonl --out docs/figures

Reads one or more JSONL result logs, applies the known corrections, and writes
PNG+PDF figures plus a summary CSV. Corrections (see README "data hazards"):

  * build_calc imputation — HNSWMerger reuses leaf indexes across algorithms at
    the same partition count, so only the first algorithm to run records the
    build cost; the rest log build_calc=0. We fill each row's build_calc from the
    (single) non-zero value at its partition count, and recompute total_calc.
    (Newer runs carry this forward via a sidecar and won't need imputing.)
  * SIGM build charging — SIGM (Simple Insertion Graph Merge, the rebuild
    baseline) only builds leaf 0 and re-inserts the rest, so its own build_calc
    is not the shared P-leaf build. For a same-build TOTAL comparison we charge
    SIGM the shared P-leaf build (like the merges); the honest alternative
    (leaf 0 only, ~= monolithic INSERT) would make SIGM *cheaper* on total, which
    is why the reviewer's "merges beat rebuild" claim is read off merge_calc, not
    total_calc. SIGM/INSERT/REBUILD are excluded from the shared-build source so
    they cannot pollute it.
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


MERGE_ALGOS = ["SIGM", "NGM", "IGTM", "CGTM", "TWO_MERGE"]
# TWO_MERGE is the SIGMOD'26 HNSW-Merger algorithm (experiment.cpp calls
# hnswlib::HNSWMerger<float>(a, b, &space, lambda)); display it under its real name.
DISPLAY = {"TWO_MERGE": "HNSWMerger", "SIGM": "SIGM", "INSERT": "Rebuild"}


def disp(algo: str) -> str:
    return DISPLAY.get(algo, algo)
# algos that build all P leaves (so their build_calc is the shared per-partition
# build cost). SIGM (leaf 0 only) and INSERT/REBUILD (monolithic) are NOT here.
TRUE_MERGE = {"NGM", "IGTM", "CGTM", "ES", "TWO_MERGE"}
COLORS = {"NGM": "#d1495b", "IGTM": "#2e86de", "CGTM": "#16a085",
          "ES": "#e67e22", "TWO_MERGE": "#8e44ad", "INSERT": "#7f8c8d",
          "SIGM": "#34495e", "NNDescent": "#27ae60"}   # SIGM slate: distinct from NGM red and Rebuild grey
BUILD_GRAY = "#d5d8dc"


def ds_display(ds_key):
    """Dataset KEY stays bigann* (cache/dedup safety); TITLES read 'SIFT N (BIGANN)'.
    ANN_SIFT1M keeps its own name - it is a different collection, not a prefix."""
    import re
    m = re.match(r"bigann(\d+)([km]?)$", (ds_key or "").lower())
    if m:
        return f"SIFT {m.group(1)}{m.group(2).upper()} (BIGANN)"
    return (ds_key or "").upper()
ISO_TARGETS = [0.90, 0.95]  # recall levels for the iso-quality scatter
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


def _bkey(r):
    """Build-cost group: leaves depend on the build parameters, not just the
    partition count. Without M/ef_construction here, an efc sweep makes SIGM
    inherit another group's build_calc and its total_calc becomes nonsense."""
    return (r.get("dataset"), r.get("n_parts"), r.get("m"), r.get("ef_construction"))


def correct(rows):
    # shared P-leaf build/seconds per (dataset, n_parts), sourced ONLY from the
    # true merge algos so SIGM/INSERT/REBUILD can't overwrite it.
    build_by, secs_by = {}, {}
    for r in rows:
        if (r.get("builder") == "hnswmerger" and r.get("algo") in TRUE_MERGE
                and (r.get("build_calc") or 0) > 0):
            build_by[_bkey(r)] = r["build_calc"]
            secs_by[_bkey(r)] = r.get("build_seconds") or 0
    for r in rows:
        if r.get("builder") == "hnswmerger":
            key = _bkey(r)
            # SIGM: force the shared P-leaf build (same-build total comparison).
            # Others: impute only when the build wasn't recorded (leaf reuse -> 0).
            if r.get("algo") == "SIGM" or not (r.get("build_calc") or 0):
                if build_by.get(key):
                    r["build_calc"] = build_by[key]
                    r["build_seconds"] = secs_by.get(key, r.get("build_seconds") or 0)
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


# Which knobs each algorithm actually consumes. rec["params"] stores the full
# resolved set, so without this filter every row shows every knob - INSERT
# displaying "j40,l10,lam4,nse6,nsk6,sM40,sef40" despite reading none of them.
# Mirrors the C++ signatures in baseline2.h (note CGTM has no next_step_ef).
RELEVANT_PARAMS = {
    "NGM": {"search_ef"},
    "IGTM": {"jump_ef", "local_ef", "next_step_k", "next_step_ef", "search_M"},
    "CGTM": {"jump_ef", "local_ef", "next_step_k", "search_M"},
    "TWO_MERGE": {"merge_lambda"},
    "SIGM": {"merge_ef_construction"},
    "INSERT": set(), "REBUILD": set(), "ES": set(),
}


def _pid(r) -> str:
    """Short label for a parameter point; '' for a default/unswept row."""
    mp = r.get("params") or {}
    short = {"jump_ef": "j", "local_ef": "l", "next_step_k": "nsk",
             "next_step_ef": "nse", "search_M": "sM", "search_ef": "sef",
             "merge_ef_construction": "sigm_efc", "merge_lambda": "lam",
             "ef_construction": "efc", "M": "M"}
    keep = RELEVANT_PARAMS.get(r.get("algo"))
    if keep is not None:
        mp = {k: v for k, v in mp.items() if k in keep}
    parts = [f"{short.get(k, k)}{v}" for k, v in sorted(mp.items()) if v != -1]
    # build parameters live on the row itself, not in params
    if r.get("ef_construction") is not None:
        parts.insert(0, f"efc{r['ef_construction']}")
    if r.get("m") is not None:
        parts.insert(0, f"M{r['m']}")
    return ",".join(parts)


def _ds_at_recall(r, target):
    """Interpolate search distance computations per query at a target recall.

    Returns None if the row's curve never reaches `target` — a config that
    cannot hit the target quality has no iso-quality point and must not be
    silently plotted at its best-effort value.
    """
    cur = [(c.get("recall"), c.get("d_s")) for c in (r.get("recall_curve") or [])
           if c.get("recall") is not None and c.get("d_s") is not None]
    if len(cur) < 2:
        return None
    cur.sort()
    recs = [c[0] for c in cur]
    dss = [c[1] for c in cur]
    if target < recs[0] or target > recs[-1]:
        return None
    import bisect
    i = bisect.bisect_left(recs, target)
    if i == 0:
        return dss[0]
    r0, r1, d0, d1 = recs[i - 1], recs[i], dss[i - 1], dss[i]
    if r1 == r0:
        return d1
    return d0 + (d1 - d0) * (target - r0) / (r1 - r0)


def fig_iso_quality(rows, out, ds="SIFT1M", n_parts=2, target=0.95):
    """Merge cost at MATCHED search quality — the comparison the paper's table makes.

    x = search distance computations per query (d_s) interpolated at `target`
    recall; y = merge-phase distance computations. A config that merges cheaply
    by producing a worse graph moves RIGHT on the d_s axis, so a cheap-but-poor
    merge is visibly distinguished from a genuine efficiency gain.
    Rows that never reach `target` are dropped rather than plotted at their best.
    """
    pts = []
    for r in rows:
        if r.get("builder") != "hnswmerger" or r.get("n_parts") != n_parts:
            continue
        if r.get("algo") not in MERGE_ALGOS and r.get("algo") not in TRUE_MERGE:
            continue
        d_s = _ds_at_recall(r, target)
        mc = r.get("merge_calc")
        if d_s is None or not mc:
            continue
        lam = (r.get("params") or {}).get("merge_lambda")
        pts.append((r["algo"], _pid(r), d_s, mc / 1e9, lam))
    if not pts:
        print(f"  (skip iso_quality: no rows reach recall {target} with d_s)"); return
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    seen = set()
    for algo, pid, d_s, mc, lam in pts:
        ax.scatter(d_s, mc, s=90, color=COLORS.get(algo, "#555"), zorder=3,
                   edgecolor="white", linewidth=0.8,
                   label=disp(algo) if algo not in seen else None)
        seen.add(algo)
        # annotate HNSWMerger points with their lambda - that dial is the point
        # of this chart; other strategies show a single canonical config.
        if algo == "TWO_MERGE" and lam is not None:
            ax.annotate(f"\u03bb{lam}", (d_s, mc), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color=COLORS["TWO_MERGE"])
    ax.set_xlabel(f"search distance computations / query at recall@10 = {target}")
    ax.set_ylabel("merge-phase distance computations  (billions)")
    ax.set_title(f"Merge cost at matched search quality, {n_parts} partitions ({ds})")
    ax.legend(title="algorithm", fontsize=9)
    _save(fig, out, f"iso_quality_r{int(target*100)}")


def _shared_build_g(rows, p):
    """Shared P-leaf build (billions) at partition count p, from a true-merge row."""
    v = next((r.get("build_calc") for r in rows if r.get("builder") == "hnswmerger"
              and r.get("n_parts") == p and r.get("algo") in TRUE_MERGE
              and (r.get("build_calc") or 0) > 0), 0) or 0
    return v / 1e9


# ---- merge-STRATEGY view (SIGMOD nomenclature) -----------------------------
# Every method is one merge strategy with a single cost: Rebuild (=INSERT, full
# from-scratch build), SIGM (insertion seeded from one index), and the traversal
# merges NGM/IGTM/CGTM/HNSWMerger. On the distance axis:
#   Rebuild cost = INSERT total_calc (build from scratch, no separate merge)
#   SIGM cost    = its merge_calc (insertion of the other half)
#   others       = their merge_calc
# Build/merge are NOT split here - for Rebuild they are not separable, which is
# the whole point of the strategy framing.
STRATEGY_ORDER = ["TWO_MERGE", "IGTM", "CGTM", "NGM", "SIGM", "INSERT"]
STRATEGY_LABEL = {"TWO_MERGE": "HNSWMerger", "IGTM": "IGTM", "CGTM": "CGTM",
                  "NGM": "NGM", "SIGM": "SIGM", "INSERT": "Rebuild"}
# canonical config shown for each strategy (the iso-quality winner); baked into
# captions so the comparison charts need no per-point knob labels.
CANON_CONFIG = {"TWO_MERGE": "\u03bb=4", "IGTM": "j5,l7", "CGTM": "j15,l5",
                "NGM": "sef10", "SIGM": "efc=200", "INSERT": "\u2014"}

def _canon_caption(algos):
    parts = [f"{STRATEGY_LABEL[a]} {CANON_CONFIG[a]}" for a in algos
             if a in CANON_CONFIG and CANON_CONFIG[a] != "\u2014"]
    return "config: " + "; ".join(parts)


# Each strategy is summarized by its canonical config, not the cheapest row in a
# parameter sweep: SIGM at merge_ef_construction=-1 (inherit), HNSW-Merger at
# lambda=4. This keeps the per-strategy ordering consistent across scales, some of
# which sweep a knob and some of which do not.
def _is_canonical(r):
    p = r.get("params") or {}
    algo = r.get("algo")
    if algo == "TWO_MERGE":
        return p.get("merge_lambda") == 4          # HNSWMerger canonical lambda
    if algo == "SIGM":
        return p.get("merge_ef_construction", -1) in (-1, None)  # inherit ef
    return True   # NGM/IGTM/CGTM: single canonical config in the grid sweep


def _strategy_cost(rows, algo, n_parts=2):
    """Single per-strategy cost on the distance axis at efc=200, canonical config."""
    if algo == "INSERT":
        r = next((r for r in rows if r.get("algo") == "INSERT"
                  and r.get("builder") == "hnswmerger"
                  and _efc_row(r) == 200), None)
        return (r.get("total_calc") if r else None)
    cands = [r.get("merge_calc") for r in rows
             if r.get("algo") == algo and r.get("n_parts") == n_parts
             and r.get("builder") == "hnswmerger" and r.get("merge_calc")
             and _efc_row(r) == 200 and _is_canonical(r)]
    return min(cands) if cands else None


def _efc_row(r):
    """ef_construction from the row top level (not params/merge_id), default 200."""
    v = r.get("ef_construction")
    if v is None:
        v = (r.get("params") or {}).get("ef_construction")
    return 200 if v is None else v


def _scale_of(ds_name):
    """Extract N from a dataset name like bigann10k / bigann1m / bigann10m."""
    import re
    m = re.search(r"(\d+)\s*([km]?)$", (ds_name or "").lower())
    if not m:
        return None
    v = int(m.group(1)); u = m.group(2)
    return v * {"k": 1_000, "m": 1_000_000, "": 1}[u]


def fig_cross_dataset(dataset_files, out, scale_label="1M"):
    """Cross-dataset stability: canonical merge cost per strategy, NORMALIZED to
    each dataset's own Rebuild total, grouped by strategy across datasets at a
    fixed scale. Normalizing to Rebuild removes the ~10x absolute-cost spread
    between datasets (Deep 96-d vs GIST 960-d) so the *ordering and relative
    magnitude* are the visual point: the strategies cost a near-constant fraction
    of a full rebuild regardless of embedding type.

    dataset_files: list of (display_name, jsonl_path). Rebuild (INSERT total) is
    the per-dataset denominator; merges use their canonical config at efc=200.
    """
    merges = ["TWO_MERGE", "IGTM", "CGTM", "NGM", "SIGM"]
    data = {}   # display_name -> {strategy: fraction_of_rebuild}
    for name, path in dataset_files:
        if not os.path.exists(path):
            print(f"  (cross: skip missing {path})"); continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        reb = next((r.get("total_calc") for r in rows
                    if r.get("algo") == "INSERT" and r.get("builder") == "hnswmerger"
                    and _efc_row(r) == 200), None)
        if not reb:
            print(f"  (cross: no Rebuild for {name})"); continue
        frac = {}
        for algo in merges:
            c = _strategy_cost(rows, algo, 2)
            if c:
                frac[algo] = c / reb
        data[name] = frac
    if len(data) < 2:
        print("  (skip cross_dataset: <2 datasets)"); return

    names = list(data)
    nD = len(names)
    fig, ax = plt.subplots(figsize=(1.7 * nD + 3.5, 4.8))
    x = range(len(merges))
    width = 0.8 / nD
    # a distinct hue per dataset, strategies share the x-axis
    ds_colors = ["#8e44ad", "#2e86de", "#e67e22", "#16a085", "#c0392b", "#7f8c8d"]
    for di, name in enumerate(names):
        offs = [xi + (di - (nD - 1) / 2) * width for xi in x]
        vals = [data[name].get(a, 0) for a in merges]
        bars = ax.bar(offs, vals, width, label=name,
                      color=ds_colors[di % len(ds_colors)],
                      edgecolor="white", linewidth=0.4)
        # light value labels so the ~0.05/0.10/0.15/0.50 bands are quantifiable
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.5, color="#333",
                        rotation=90)
    ax.axhline(1.0, color="#555", lw=1.0, ls="--", alpha=0.6)
    ax.text(len(merges) - 0.5, 1.01, "Rebuild = 1.0", fontsize=8,
            color="#555", ha="right", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels([STRATEGY_LABEL[a] for a in merges])
    ax.set_ylabel("merge cost / dataset's own Rebuild cost")
    ax.set_title(f"Merge cost relative to full rebuild across datasets "
                 f"({scale_label})")
    ax.legend(title="dataset", fontsize=9)
    ax.set_ylim(0, max(0.6, max(v for d in data.values() for v in d.values()) * 1.15))
    _save(fig, out, "cross_dataset")




def fig_cross_scale(dataset_scale_files, out, focus="TWO_MERGE"):
    """Cost-relative-to-Rebuild vs scale, one line per dataset, for a single
    strategy (default HNSW-Merger). Shows the fraction is stable across BOTH
    dataset and scale - the strongest single statement of the generalization.

    dataset_scale_files: dict name -> {N: jsonl_path}.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ds_colors = ["#8e44ad", "#2e86de", "#e67e22", "#16a085", "#c0392b"]
    any_line = False
    for di, (name, scalemap) in enumerate(dataset_scale_files.items()):
        xs, ys = [], []
        for N in sorted(scalemap):
            path = scalemap[N]
            if not os.path.exists(path):
                continue
            rows = [json.loads(l) for l in open(path) if l.strip()]
            reb = next((r.get("total_calc") for r in rows
                        if r.get("algo") == "INSERT"
                        and r.get("builder") == "hnswmerger"
                        and _efc_row(r) == 200), None)
            c = _strategy_cost(rows, focus, 2)
            if reb and c:
                xs.append(N); ys.append(c / reb)
        if len(xs) >= 2:
            any_line = True
            ax.plot(xs, ys, "o-", color=ds_colors[di % len(ds_colors)], label=name)
    if not any_line:
        print("  (skip cross_scale: need >=2 scales per dataset)"); plt.close(fig); return
    ax.set_xscale("log")
    ax.set_xlabel("N (log)")
    ax.set_ylabel(f"{STRATEGY_LABEL.get(focus, focus)} cost / Rebuild cost")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{STRATEGY_LABEL.get(focus, focus)} cost relative to rebuild, "
                 f"across scale and dataset")
    ax.legend(title="dataset", fontsize=9)
    _save(fig, out, "cross_scale")


def fig_merge_strategies_grid(all_rows, out):
    """SIGMOD Fig.3-style small-multiples: one merge-cost bar panel per scale,
    Rebuild included as a strategy. The erosion of the merge advantage shows as
    the gap between Rebuild and the merges narrowing panel to panel - no slope
    fit, no exponent to misread, matching the reference paper's nomenclature."""
    by_scale = {}
    for r in all_rows:
        N = _scale_of(r.get("dataset"))
        if N and r.get("builder") == "hnswmerger":
            by_scale.setdefault(N, []).append(r)
    scales = sorted(by_scale)
    if not scales:
        print("  (skip merge_strategies_grid: no scale data)"); return
    ncol = len(scales)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 4.2), squeeze=False)
    for col, N in enumerate(scales):
        ax = axes[0][col]
        rows = by_scale[N]
        present = [(a, _strategy_cost(rows, a, 2)) for a in STRATEGY_ORDER]
        present = [(a, c) for a, c in present if c]
        present.sort(key=lambda kv: kv[1])
        vals = [c / 1e9 for _, c in present]
        cols = [COLORS.get(a, "#555") for a, _ in present]
        ax.bar(range(len(vals)), vals, color=cols, edgecolor="white", linewidth=0.5)
        reb = next((c for a, c in present if a == "INSERT"), None)
        for i, (a, c) in enumerate(present):
            if reb and a != "INSERT":
                sp = reb / c
                txt = f"{sp:.1f}\u00d7" if sp < 10 else f"{sp:.0f}\u00d7"
                ax.text(i, vals[i], txt, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([STRATEGY_LABEL[a] for a, _ in present], rotation=90, fontsize=7)
        ax.set_title(ds_display(f"bigann{N//1000000}m" if N >= 1_000_000
                                else f"bigann{N//1000}k"), fontsize=9)
        if col == 0:
            ax.set_ylabel("merge distance computations (billions)")
    fig.suptitle("Merge cost by strategy across scale  (\u00d7 = speedup vs Rebuild)",
                 fontsize=11)
    _save(fig, out, "merge_strategies_grid")


def fig_scale_trend(all_rows, out):
    """Per-strategy merge cost vs N (log-log), one line per strategy. This is the
    four-decade result and it needs no build/total split - merge cost is merge
    cost at every scale. Rebuild included as the top reference line."""
    by_scale = {}
    for r in all_rows:
        N = _scale_of(r.get("dataset"))
        if N and r.get("builder") == "hnswmerger":
            by_scale.setdefault(N, []).append(r)
    scales = sorted(by_scale)
    if len(scales) < 2:
        print("  (skip scale_trend: need >=2 scales)"); return
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for algo in STRATEGY_ORDER:
        xs, ys = [], []
        for N in scales:
            c = _strategy_cost(by_scale[N], algo, n_parts=2)
            if c:
                xs.append(N); ys.append(c / 1e9)
        if len(xs) >= 2:
            ax.plot(xs, ys, "o-", color=COLORS.get(algo, "#555"),
                    label=STRATEGY_LABEL[algo], linewidth=1.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dataset size N  (log)")
    ax.set_ylabel("merge distance computations, billions  (log)")
    ax.set_title("Merge cost by strategy across scale")
    ax.legend(title="strategy", fontsize=9)
    shown = [a for a in STRATEGY_ORDER if any(_strategy_cost(v, a, 2)
             for v in by_scale.values())]
    ax.text(0.5, -0.16, _canon_caption(shown), transform=ax.transAxes,
            ha="center", fontsize=7.5, color="#666")
    _save(fig, out, "scale_trend")


def fig_param_sweep(rows, out, ds="SIFT1M", n_parts=2, target=0.95):
    """Knob-space charts for the strategies with enough sampling to warrant one.
    Only NGM (search_ef, 3 points) qualifies; HNSWMerger's lambda dial lives in
    fig_iso_quality, and IGTM/CGTM were sampled too sparsely (tuned-vs-default,
    stated in caption) to draw a trend through."""
    ngm = [r for r in rows if r.get("algo") == "NGM" and r.get("n_parts") == n_parts
           and r.get("builder") == "hnswmerger" and r.get("merge_calc")]
    def sef(r):
        return (r.get("params") or {}).get("search_ef")
    ngm = [r for r in ngm if sef(r) is not None]
    if len(ngm) < 2:
        print("  (skip param_sweep: NGM search_ef has <2 points)"); return
    ngm.sort(key=sef)
    xs = [sef(r) for r in ngm]
    cost = [r["merge_calc"] / 1e9 for r in ngm]
    ds = [_ds_at_recall(r, target) for r in ngm]

    fig, axc = plt.subplots(figsize=(6.8, 4.4))
    axc.plot(xs, cost, "o-", color=COLORS["NGM"], label="merge cost")
    axc.set_xlabel("NGM search_ef")
    axc.set_ylabel("merge distance computations (billions)", color=COLORS["NGM"])
    axc.tick_params(axis="y", labelcolor=COLORS["NGM"])
    axc.set_xticks(xs)
    if any(v is not None for v in ds):
        axd = axc.twinx()
        axd.plot(xs, ds, "s--", color="#2c3e50", label=f"d_s @ recall {target}")
        axd.set_ylabel(f"search dist/query @ recall {target}", color="#2c3e50")
        axd.tick_params(axis="y", labelcolor="#2c3e50")
        axd.spines["top"].set_visible(False)
    axc.set_title(f"NGM: cost and search quality vs search_ef ({ds})")
    _save(fig, out, "param_sweep_ngm")


def fig_merge_strategies(rows, out, ds="SIFT1M", n_parts=2):
    """One bar per merge strategy - Rebuild and SIGM promoted to first-class
    strategies alongside the traversal merges (SIGMOD Fig.3 nomenclature). Single
    cost axis; no build/merge split."""
    present = [(a, _strategy_cost(rows, a, n_parts)) for a in STRATEGY_ORDER]
    present = [(a, c) for a, c in present if c]
    if len(present) < 2:
        print("  (skip merge_strategies: <2 strategies)"); return
    present.sort(key=lambda kv: kv[1])
    labels = [STRATEGY_LABEL[a] for a, _ in present]
    vals = [c / 1e9 for _, c in present]
    cols = [COLORS.get(a, "#555") for a, _ in present]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="white", linewidth=0.6)
    reb = next((c for a, c in present if a == "INSERT"), None)
    for i, (bar, (a, c)) in enumerate(zip(bars, present)):
        lbl = f"{vals[i]:.2f}"
        if reb and a != "INSERT":
            sp = reb / c
            lbl += f"\n{sp:.1f}\u00d7" if sp < 10 else f"\n{sp:.0f}\u00d7"  # speedup vs Rebuild
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                lbl, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("merge distance computations  (billions)")
    ax.set_title(f"Merge cost by strategy ({ds}, {n_parts} partitions)")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.text(0.5, -0.22, _canon_caption([a for a, _ in present]),
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#666")
    _save(fig, out, "merge_strategies")


def fig_merge_cost(rows, out, ds="SIFT1M"):
    parts = sorted({r["n_parts"] for r in rows
                    if r.get("builder") == "hnswmerger" and r["algo"] in MERGE_ALGOS})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.8 / max(1, len(MERGE_ALGOS))
    for ai, algo in enumerate(MERGE_ALGOS):
        ys = []
        for p in parts:
            m = [r["merge_calc"] for r in rows if r.get("algo") == algo
                 and r.get("n_parts") == p and r.get("builder") == "hnswmerger"
                 and r.get("merge_calc") is not None]
            ys.append((m[0] / 1e9) if m else 0)
        xs = [i + ai * w for i in range(len(parts))]
        ax.bar(xs, ys, w, label=disp(algo), color=COLORS[algo])
    ax.set_xticks([i + w * (len(MERGE_ALGOS) - 1) / 2 for i in range(len(parts))])
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
        axc.plot(parts, mc, "o-", color=COLORS[algo], label=disp(algo))
        # SIGM (and any baseline) has no recall curve -> skip it on the recall axis
        if any(v is not None for v in rc):
            axr.plot(parts, rc, "o-", color=COLORS[algo], label=disp(algo))
    # shared build cost (decreasing with partition count) on the cost panel
    bc = [next((r["build_calc"] / 1e9 for r in rows if r.get("n_parts") == p
                and r.get("builder") == "hnswmerger" and r.get("algo") in TRUE_MERGE
                and (r.get("build_calc") or 0) > 0), None)
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


def _plabel(r) -> str:
    """Parameter label for use INSIDE one build group: drop the M/efc prefix,
    which is constant across the panel and would just repeat in every tick."""
    pid = _pid(r)
    parts = [p for p in pid.split(",") if not (p.startswith("M") and p[1:].isdigit())
             and not p.startswith("efc")]
    return ",".join(parts)


def fig_recall_vs_ds(rows, out, ds="SIFT1M", n_parts=2):
    """Search quality vs search COST on the distance-computation axis: recall@10
    against d_s (search distance computations per query). The implementation-
    independent companion to fig_recall_vs_qps - same curves, but x is work done
    rather than wall-clock throughput, so it is not confounded by threading or
    the C++/Python split. NN-Descent is omitted: its search is not d_s-
    instrumented, and it was never a controlled time comparison either."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    methods = [r for r in rows if r.get("builder") == "hnswmerger"
               and r.get("n_parts") == n_parts and r.get("recall_curve")]
    order = ["TWO_MERGE", "IGTM", "CGTM", "NGM", "ES"]
    methods.sort(key=lambda r: order.index(r["algo"]) if r["algo"] in order else 99)
    drawn = 0
    for r in methods:
        pts = [(c["d_s"], c["recall"]) for c in r["recall_curve"]
               if c.get("d_s") and c.get("recall") is not None]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, "o-", color=COLORS.get(r["algo"], "#555"), label=disp(r["algo"]))
        drawn += 1
    if not drawn:
        print("  (skip recall_vs_ds: no d_s curves)"); plt.close(fig); return
    ax.set_xlabel("search distance computations / query  (d_s)")
    ax.set_ylabel("recall@10")
    ax.set_title(f"Search quality vs search cost at {n_parts} partitions ({ds})")
    ax.legend(title="strategy")
    _save(fig, out, "recall_vs_ds")


def write_summary(rows, out):
    os.makedirs(out, exist_ok=True)
    cols = ["builder", "algo", "n_parts", "params_id", "build_calc", "merge_calc",
            "total_calc", "build_seconds", "merge_seconds", "recall@10", "d_s@0.95"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (str(r.get("algo")), r.get("n_parts", 0),
                                             _pid(r))):
            w.writerow({**r, "params_id": _pid(r), "d_s@0.95": _ds_at_recall(r, 0.95)})
    print("  wrote summary.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", nargs="+", default=None,
                    help="cross-dataset mode: NAME=path.jsonl pairs at one scale")
    ap.add_argument("--cross-scale", default="1M", help="scale label for --cross title")
    ap.add_argument("--cross-scale-lines", nargs="+", default=None,
                    help="cross-scale line mode: NAME:N=path.jsonl tuples")
    ap.add_argument("--cross-focus", default="TWO_MERGE",
                    help="strategy for --cross-scale-lines (default TWO_MERGE)")
    ap.add_argument("--results", nargs="+",
                    default=["results/bigann10k.jsonl", "results/bigann100k.jsonl",
                             "results/bigann1m.jsonl", "results/bigann10m.jsonl"])
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)

    if a.cross:
        pairs = []
        for tok in a.cross:
            if "=" not in tok:
                print(f"--cross expects NAME=path.jsonl, got {tok!r}"); return
            name, path = tok.split("=", 1)
            pairs.append((name, path))
        fig_cross_dataset(pairs, a.out, a.cross_scale)
        return

    if a.cross_scale_lines:
        import collections
        dsf = collections.defaultdict(dict)
        for tok in a.cross_scale_lines:
            # NAME:N=path  e.g. SIFT:100000=results/bigann100k.jsonl
            head, path = tok.split("=", 1)
            name, N = head.split(":", 1)
            dsf[name][int(N)] = path
        fig_cross_scale(dict(dsf), a.out, a.cross_focus)
        return
    rows = correct(load(a.results))
    if not rows:
        print("no rows found"); return

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r.get("dataset") or "unknown"].append(r)

    for ds_key, ds_rows in sorted(by_ds.items()):
        ds = ds_display(ds_key)
        out = os.path.join(a.out, ds_key)
        print(f"{ds_key}: {len(ds_rows)} rows -> {out}")
        fig_merge_strategies(ds_rows, out, ds)
        for t in ISO_TARGETS:
            fig_iso_quality(ds_rows, out, ds, n_parts=2, target=t)
        fig_param_sweep(ds_rows, out, ds, n_parts=2)
        fig_merge_cost(ds_rows, out, ds)
        fig_partition_scaling(ds_rows, out, ds)
        fig_recall_vs_qps(ds_rows, out, ds)
        fig_recall_vs_ds(ds_rows, out, ds)
        write_summary(ds_rows, out)
        if not any(r.get("builder") == "nndescent" for r in ds_rows):
            print(f"  note: no NN-Descent rows for {ds_key} — "
                  f"run config/{ds_key}.json (nndescent) to add it.")

    # cross-dataset: merge cost vs scale (all bigann* scales together)
    fig_scale_trend(rows, os.path.join(a.out, "_scale"))
    fig_merge_strategies_grid(rows, os.path.join(a.out, "_scale"))


if __name__ == "__main__":
    main()