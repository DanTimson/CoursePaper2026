#!/usr/bin/env python3
"""End-to-end operation accounting for partition-build + merge experiments.

The build budget is always taken from the independent BUILD_ONLY sweep:

    budget(N, P) = B(N) - sum_j B(S_j)

Merge logs contribute only C_A(N, P).  The script never infers leaf-build work
from a merge result.

Primary figures:
  1. P=2 component curves across 10K, 100K, 1M.
  2. Build / merge / total versus P for every dataset and strategy.
  3. Total/rebuild ratio versus P.
  4. Merge-budget utilization C_A / budget.
  5. Break-even speedup required of a merge.
  6. Optional quality and per-tree-level diagnostics.

Example:
    python scripts/plot_total_cost_experiment.py \
      --build-budget docs/figures/build_budget/build_budget.csv \
      --merge-results results/total_cost_bigann10k.jsonl \
                      results/total_cost_bigann100k.jsonl \
                      results/total_cost_bigann1m.jsonl \
      --out docs/figures/total_cost
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def expand_paths(values: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        matches = [Path(p) for p in glob.glob(value)]
        out.extend(matches or [Path(value)])
    return [p for p in out if p.exists()]


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("run_key") or f"{path}:{line_no}"
                by_key[key] = row
    return list(by_key.values())


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def as_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def parse_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def dataset_label(name: str) -> str:
    labels = {
        "bigann10k": "SIFT 10K",
        "bigann100k": "SIFT 100K",
        "bigann1m": "SIFT 1M",
        "bigann10m": "SIFT 10M",
    }
    return labels.get(name.lower(), name)


def auto_variant(row: dict[str, Any]) -> str:
    if row.get("variant"):
        return str(row["variant"])
    algo = row.get("algo") or row.get("algorithm") or "unknown"
    params = parse_params(row.get("params"))
    order = row.get("order", "balanced")
    if algo in {"TWO_MERGE", "HNSW-MERGER"}:
        lam = params.get("merge_lambda", 4)
        mode = params.get("merge_lambda_mode", "fixed")
        suffix = "large-first" if order == "sequential" else order
        return f"HNSW-Merger {mode} λ0={lam}, {suffix}"
    if algo == "NGM":
        return f"NGM search_ef={params.get('search_ef', 40)}"
    if algo == "IGTM":
        return "IGTM tuned"
    if algo == "CGTM":
        return "CGTM tuned"
    return str(algo)


def build_index(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    out = {}
    for row in rows:
        key = (
            row["dataset"],
            as_int(row["n_parts"]),
            as_int(row["m"]),
            as_int(row["ef_construction"]),
            as_int(row.get("threads"), 1),
        )
        out[key] = row
    return out


def interpolate_y_at_x(points: list[tuple[float, float]], target: float) -> Optional[float]:
    clean = sorted((x, y) for x, y in points if x is not None and y is not None)
    if not clean:
        return None
    for x, y in clean:
        if x == target:
            return y
    for (x0, y0), (x1, y1) in zip(clean, clean[1:]):
        if x0 <= target <= x1 and x1 != x0:
            t = (target - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


def quality_metrics(row: dict[str, Any], ef: int = 100, recall_target: float = 0.95) -> dict[str, Any]:
    curve = row.get("recall_curve") or []
    recall_at_ef = None
    for point in curve:
        if as_int(point.get("ef"), -1) == ef and point.get("recall") is not None:
            recall_at_ef = float(point["recall"])
            break
    ds_points = [
        (float(point["recall"]), float(point["d_s"]))
        for point in curve
        if point.get("recall") is not None and point.get("d_s") is not None
    ]
    ds_at_target = interpolate_y_at_x(ds_points, recall_target)
    max_recall = max(
        (float(point["recall"]) for point in curve if point.get("recall") is not None),
        default=None,
    )
    return {
        f"recall_at_ef_{ef}": recall_at_ef,
        f"ds_at_recall_{recall_target:.2f}": ds_at_target,
        "max_recall": max_recall,
    }


def join_rows(
    build_rows: list[dict[str, Any]],
    merge_rows: list[dict[str, Any]],
    *,
    strict_build_check: bool,
) -> list[dict[str, Any]]:
    builds = build_index(build_rows)
    output: list[dict[str, Any]] = []
    mismatches = []

    for row in merge_rows:
        if row.get("builder") not in (None, "", "hnswmerger"):
            continue
        p = as_int(row.get("n_parts"), 1)
        if p <= 1 or row.get("merge_calc") in (None, ""):
            continue
        dataset = str(row.get("dataset"))
        key = (
            dataset,
            p,
            as_int(row.get("m"), 16),
            as_int(row.get("ef_construction"), 200),
            as_int(row.get("threads"), 1),
        )
        build = builds.get(key)
        if build is None:
            print(f"skip merge row without independent build budget: {key}")
            continue

        B = as_int(build["monolithic_build_calc"])
        L = as_int(build["leaf_build_calc"])
        C = as_int(row["merge_calc"])
        budget = B - L
        run_build = as_int(row.get("build_calc"), 0)
        if run_build and run_build != L:
            mismatches.append((auto_variant(row), key, run_build, L))

        record = {
            "dataset": dataset,
            "dataset_label": dataset_label(dataset),
            "n": as_int(build["n"]),
            "n_parts": p,
            "m": key[2],
            "ef_construction": key[3],
            "threads": key[4],
            "algorithm": row.get("algo") or row.get("algorithm"),
            "variant": auto_variant(row),
            "order": row.get("order", "balanced"),
            "params": json.dumps(parse_params(row.get("params")), sort_keys=True),
            "monolithic_build_calc": B,
            "leaf_build_calc": L,
            "merge_calc": C,
            "total_calc": L + C,
            "build_ratio": L / B,
            "merge_ratio": C / B,
            "total_ratio": (L + C) / B,
            "budget_calc": budget,
            "budget_ratio": budget / B,
            "budget_utilization": C / budget if budget > 0 else math.inf,
            "net_advantage_calc": budget - C,
            "net_advantage_ratio": (budget - C) / B,
            "partition_wins": C < budget,
            "run_build_calc": run_build,
            "merge_seconds": as_float(row.get("merge_seconds")),
            "build_seconds": as_float(row.get("build_seconds")),
            "merge_steps": row.get("merge_steps") or [],
            **quality_metrics(row),
        }
        output.append(record)

    if mismatches:
        message = "\n".join(
            f"  {variant} {key}: run build={actual:,}, budget build={expected:,}"
            for variant, key, actual, expected in mismatches
        )
        if strict_build_check:
            raise RuntimeError("merge-run build counters disagree with BUILD_ONLY data:\n" + message)
        print("WARNING: build counter mismatches; BUILD_ONLY values remain authoritative:\n" + message)

    return output


def save_figure(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def ordered_datasets(rows: list[dict[str, Any]]) -> list[str]:
    ns = {}
    for row in rows:
        ns[row["dataset"]] = min(ns.get(row["dataset"], row["n"]), row["n"])
    return [name for name, _ in sorted(ns.items(), key=lambda item: item[1])]


def ordered_variants(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "IGTM tuned",
        "CGTM tuned",
        "NGM search_ef=10",
        "HNSW-Merger fixed λ0=4, balanced",
        "HNSW-Merger fixed λ0=4, large-first",
        "HNSW-Merger adaptive λ0=4, large-first",
    ]
    present = {row["variant"] for row in rows}
    return [v for v in preferred if v in present] + sorted(present - set(preferred))


def axes_grid(count: int, width: float = 6.0, height: float = 4.2):
    cols = 2 if count > 1 else 1
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    flat = list(axes.flat)
    for ax in flat[count:]:
        ax.set_visible(False)
    return fig, flat[:count]


def plot_p2_components(rows: list[dict[str, Any]], out: Path) -> None:
    subset = [row for row in rows if row["n_parts"] == 2]
    variants = ordered_variants(subset)
    if not variants:
        return
    fig, axes = axes_grid(len(variants))
    for ax, variant in zip(axes, variants):
        group = sorted((row for row in subset if row["variant"] == variant), key=lambda row: row["n"])
        if not group:
            continue
        x = [row["n"] for row in group]
        ax.plot(x, [row["build_ratio"] for row in group], marker="o", label="Two leaf builds")
        ax.plot(x, [row["merge_ratio"] for row in group], marker="s", label="Merge")
        ax.plot(x, [row["total_ratio"] for row in group], marker="^", label="Total")
        ax.axhline(1.0, linestyle="--", linewidth=1, label="Monolithic build")
        ax.set_xscale("log")
        ax.set_xticks(x, [dataset_label(row["dataset"]) for row in group])
        ax.set_ylabel("Operations / monolithic build")
        ax.set_title(variant)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Two-partition construction components across scale", y=1.02)
    fig.tight_layout()
    save_figure(fig, out, "p2_components_by_scale")


def plot_p2_totals(rows: list[dict[str, Any]], out: Path) -> None:
    subset = [row for row in rows if row["n_parts"] == 2]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    datasets = ordered_datasets(subset)
    by_ds = {row["dataset"]: row for row in subset}
    build_points = []
    for ds in datasets:
        row = next((r for r in subset if r["dataset"] == ds), None)
        if row:
            build_points.append(row)
    if build_points:
        ax.plot(
            [row["n"] for row in build_points],
            [row["build_ratio"] for row in build_points],
            marker="o",
            linestyle="--",
            label="Two leaf builds only",
        )
    for variant in ordered_variants(subset):
        group = sorted((row for row in subset if row["variant"] == variant), key=lambda row: row["n"])
        ax.plot(
            [row["n"] for row in group],
            [row["total_ratio"] for row in group],
            marker="o",
            label=variant,
        )
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Monolithic build")
    ax.set_xscale("log")
    ticks = sorted({row["n"] for row in subset})
    tick_labels = [
        dataset_label(next(row["dataset"] for row in subset if row["n"] == n))
        for n in ticks
    ]
    ax.set_xticks(ticks, tick_labels)
    ax.set_ylabel("Total operations / monolithic build")
    ax.set_title("Two-partition end-to-end work across scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    save_figure(fig, out, "p2_total_ratio_by_scale")


def plot_components_by_p(rows: list[dict[str, Any]], out: Path) -> None:
    for dataset in ordered_datasets(rows):
        subset = [row for row in rows if row["dataset"] == dataset]
        variants = ordered_variants(subset)
        if not variants:
            continue
        fig, axes = axes_grid(len(variants))
        for ax, variant in zip(axes, variants):
            group = sorted((row for row in subset if row["variant"] == variant), key=lambda row: row["n_parts"])
            x = [row["n_parts"] for row in group]
            ax.plot(x, [row["build_ratio"] for row in group], marker="o", label="Leaf builds")
            ax.plot(x, [row["merge_ratio"] for row in group], marker="s", label="Merge")
            ax.plot(x, [row["total_ratio"] for row in group], marker="^", label="Total")
            ax.axhline(1.0, linestyle="--", linewidth=1, label="Monolithic build")
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(set(x)))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("Partitions P")
            ax.set_ylabel("Operations / monolithic build")
            ax.set_title(variant)
            ax.grid(True, alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.suptitle(f"{dataset_label(dataset)}: build, merge, and total work", y=1.02)
        fig.tight_layout()
        save_figure(fig, out, f"components_by_partitions_{dataset}")


def plot_multi_dataset(rows: list[dict[str, Any]], out: Path, *, metric: str, title: str, ylabel: str, name: str, hline: Optional[float] = None) -> None:
    eligible = [
        row for row in rows
        if row.get(metric) is not None
        and not (isinstance(row.get(metric), float) and math.isnan(row[metric]))
    ]
    datasets = ordered_datasets(eligible)
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.6), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        subset = [row for row in eligible if row["dataset"] == dataset]
        for variant in ordered_variants(subset):
            group = sorted((row for row in subset if row["variant"] == variant), key=lambda row: row["n_parts"])
            if not group:
                continue
            ax.plot(
                [row["n_parts"] for row in group],
                [row[metric] for row in group],
                marker="o",
                label=variant,
            )
        if hline is not None:
            ax.axhline(hline, linestyle="--", linewidth=1)
        ticks = sorted({row["n_parts"] for row in subset})
        if ticks:
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Partitions P")
        ax.set_ylabel(ylabel)
        ax.set_title(dataset_label(dataset))
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), fontsize=8)
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    save_figure(fig, out, name)


def plot_break_even(build_rows: list[dict[str, Any]], out: Path) -> None:
    usable = [row for row in build_rows if as_int(row["n_parts"]) > 1 and as_float(row["budget_ratio"]) > 0]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_ds[row["dataset"]].append(row)
    for dataset, group in sorted(by_ds.items(), key=lambda item: as_int(item[1][0]["n"])):
        group.sort(key=lambda row: as_int(row["n_parts"]))
        ax.plot(
            [as_int(row["n_parts"]) for row in group],
            [1.0 / as_float(row["budget_ratio"]) for row in group],
            marker="o",
            label=dataset_label(dataset),
        )
    ticks = sorted({as_int(row["n_parts"]) for row in usable})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Partitions P")
    ax.set_ylabel("Required merge speedup vs rebuild")
    ax.set_title("Break-even threshold imposed by the measured build saving")
    ax.legend()
    ax.grid(True, alpha=0.25)
    save_figure(fig, out, "required_merge_speedup")


def flatten_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        for step in row.get("merge_steps") or []:
            out.append({
                "dataset": row["dataset"],
                "n": row["n"],
                "n_parts": row["n_parts"],
                "variant": row["variant"],
                "order": row["order"],
                "monolithic_build_calc": row["monolithic_build_calc"],
                **step,
                "step_ratio_to_rebuild": as_int(step.get("merge_calc")) / row["monolithic_build_calc"],
            })
    return out


def plot_merge_levels(step_rows: list[dict[str, Any]], out: Path) -> None:
    if not step_rows:
        return
    max_p = max(as_int(row["n_parts"]) for row in step_rows)
    subset = [row for row in step_rows if as_int(row["n_parts"]) == max_p]
    if not subset:
        return
    datasets = sorted({row["dataset"] for row in subset}, key=lambda ds: min(as_int(r["n"]) for r in subset if r["dataset"] == ds))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.6), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        ds_rows = [row for row in subset if row["dataset"] == dataset]
        variants = sorted({row["variant"] for row in ds_rows})
        for variant in variants:
            group = [row for row in ds_rows if row["variant"] == variant]
            by_level: dict[int, float] = defaultdict(float)
            for row in group:
                by_level[as_int(row["level"])] += as_float(row["step_ratio_to_rebuild"], 0.0)
            levels = sorted(by_level)
            ax.plot(levels, [by_level[level] for level in levels], marker="o", label=variant)
        ax.set_xlabel("Merge-tree level")
        ax.set_ylabel("Level merge operations / rebuild")
        ax.set_title(dataset_label(dataset))
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), fontsize=8)
    fig.suptitle(f"Where merge work is spent at P={max_p}", y=1.04)
    fig.tight_layout()
    save_figure(fig, out, f"merge_cost_by_level_p{max_p}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-budget", required=True)
    parser.add_argument("--merge-results", nargs="*", default=[])
    parser.add_argument(
        "--merge-comparison-csv",
        help="Optional legacy/current summary CSV; useful for previewing P=2 before new JSONL runs.",
    )
    parser.add_argument("--out", default="docs/figures/total_cost")
    parser.add_argument("--strict-build-check", action="store_true")
    args = parser.parse_args(argv)

    build_rows = load_csv(Path(args.build_budget))
    merge_rows = load_jsonl(expand_paths(args.merge_results))
    if args.merge_comparison_csv:
        for row in load_csv(Path(args.merge_comparison_csv)):
            row.setdefault("algo", row.get("algorithm"))
            row.setdefault("builder", "hnswmerger")
            merge_rows.append(row)
    if not merge_rows:
        raise RuntimeError("no merge results found")

    joined = join_rows(
        build_rows,
        merge_rows,
        strict_build_check=args.strict_build_check,
    )
    if not joined:
        raise RuntimeError("no merge rows matched the independent build-budget table")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Keep nested step objects out of the flat summary.
    flat_summary = [{k: v for k, v in row.items() if k != "merge_steps"} for row in joined]
    write_csv(out / "total_cost_summary.csv", flat_summary)
    steps = flatten_steps(joined)
    write_csv(out / "merge_steps.csv", steps)

    plot_p2_components(joined, out)
    plot_p2_totals(joined, out)
    plot_components_by_p(joined, out)
    plot_multi_dataset(
        joined, out,
        metric="total_ratio",
        title="End-to-end construction work",
        ylabel="Total operations / monolithic build",
        name="total_ratio_by_partitions",
        hline=1.0,
    )
    plot_multi_dataset(
        joined, out,
        metric="budget_utilization",
        title="How much of the measured merge budget is consumed",
        ylabel="Merge cost / build saving",
        name="merge_budget_utilization",
        hline=1.0,
    )
    plot_break_even(build_rows, out)
    plot_multi_dataset(
        joined, out,
        metric="recall_at_ef_100",
        title="Final-index recall at ef=100",
        ylabel="Recall@k",
        name="quality_recall_at_ef100",
    )
    plot_multi_dataset(
        joined, out,
        metric="ds_at_recall_0.95",
        title="Search work at matched recall 0.95",
        ylabel="Search distance computations / query",
        name="quality_ds_at_recall095",
    )
    plot_merge_levels(steps, out)

    print(f"wrote {out / 'total_cost_summary.csv'}")
    if steps:
        print(f"wrote {out / 'merge_steps.csv'}")


if __name__ == "__main__":
    main()
