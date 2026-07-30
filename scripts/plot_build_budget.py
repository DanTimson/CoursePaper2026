#!/usr/bin/env python3
"""Analyse directly measured partition-build budgets.

Primary quantities:

    L(N,P)       = sum of the P measured leaf-build counts
    budget(N,P)  = B(N) - L(N,P)
    total_A(N,P) = L(N,P) + merge_A(N,P)

No leaf cost is inferred from a merge algorithm.

Example:
    python scripts/plot_build_budget.py \
      --build-results results/build_budget_bigann.jsonl \
      --merge-results results/bigann10k.jsonl results/bigann100k.jsonl \
                      results/bigann1m.jsonl results/bigann10m.jsonl \
      --out docs/figures/build_budget
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(p) for p in glob.glob(value)]
        paths.extend(matches or [Path(value)])
    return [p for p in paths if p.exists()]


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("dataset"),
        int(row.get("n", 0)),
        int(row.get("dim", 0)),
        int(row.get("m", 0)),
        int(row.get("ef_construction", 0)),
        int(row.get("threads", 1)),
        row.get("partition_method", "range"),
    )


def canonical_params(row: dict[str, Any]) -> str:
    return json.dumps(row.get("params") or {}, sort_keys=True, separators=(",", ":"))


def summarise_builds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    builds = [
        row for row in rows
        if row.get("algo") == "BUILD_ONLY" and row.get("build_calc") is not None
    ]
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in builds:
        grouped[build_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        mono = next((row for row in group if int(row.get("n_parts", 0)) == 1), None)
        if mono is None:
            print(f"skip {key}: no directly measured P=1 monolithic build")
            continue
        B = int(mono["build_calc"])
        N = int(mono["n"])
        for row in sorted(group, key=lambda r: int(r["n_parts"])):
            P = int(row["n_parts"])
            L = int(row["build_calc"])
            budget = B - L
            leaf_counts = [
                int(leaf["build_calc"])
                for leaf in row.get("leaf_builds", [])
                if leaf.get("build_calc") is not None
            ]
            leaf_mean = float(np.mean(leaf_counts)) if leaf_counts else math.nan
            leaf_std = float(np.std(leaf_counts, ddof=1)) if len(leaf_counts) > 1 else 0.0
            leaf_cv = leaf_std / leaf_mean if leaf_mean else math.nan
            output.append(
                {
                    "dataset": row["dataset"],
                    "n": N,
                    "dim": row.get("dim"),
                    "m": row.get("m"),
                    "ef_construction": row.get("ef_construction"),
                    "threads": row.get("threads"),
                    "partition_method": row.get("partition_method"),
                    "n_parts": P,
                    "monolithic_build_calc": B,
                    "leaf_build_calc": L,
                    "leaf_build_ratio": L / B if B else math.nan,
                    "budget_calc": budget,
                    "budget_ratio": budget / B if B else math.nan,
                    "budget_per_point": budget / N if N else math.nan,
                    "leaf_mean_calc": leaf_mean,
                    "leaf_min_calc": min(leaf_counts) if leaf_counts else None,
                    "leaf_max_calc": max(leaf_counts) if leaf_counts else None,
                    "leaf_cv": leaf_cv,
                }
            )
    return output


def fit_budget(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["n_parts"]) > 1:
            grouped[
                (
                    row["dataset"],
                    row["m"],
                    row["ef_construction"],
                    row["threads"],
                    row["partition_method"],
                )
            ].append(row)

    fits = []
    for key, group in grouped.items():
        if len(group) < 2:
            continue
        x = np.log(np.asarray([int(row["n_parts"]) for row in group], dtype=float))
        y = np.asarray([float(row["budget_per_point"]) for row in group], dtype=float)
        design = np.column_stack([np.ones_like(x), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = design @ np.asarray([intercept, slope])
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 if ss_tot == 0 and ss_res == 0 else (math.nan if ss_tot == 0 else 1 - ss_res / ss_tot)
        fits.append(
            {
                "dataset": key[0],
                "m": key[1],
                "ef_construction": key[2],
                "threads": key[3],
                "partition_method": key[4],
                "intercept": float(intercept),
                "slope_per_ln_p": float(slope),
                "r_squared": float(r2),
                "points": len(group),
            }
        )
    return fits


def compare_merges(
    build_rows: list[dict[str, Any]],
    merge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    budgets = {
        (
            row["dataset"],
            int(row["n_parts"]),
            int(row["m"]),
            int(row["ef_construction"]),
            int(row["threads"]),
        ): row
        for row in build_rows
        if int(row["n_parts"]) > 1
    }
    output = []
    for row in merge_rows:
        if int(row.get("n_parts", 1)) <= 1:
            continue
        if row.get("merge_calc") is None:
            continue
        key = (
            row.get("dataset"),
            int(row.get("n_parts", 0)),
            int(row.get("m", 0)),
            int(row.get("ef_construction", 0)),
            int(row.get("threads", 1)),
        )
        build = budgets.get(key)
        if build is None:
            continue
        merge_calc = int(row.get("merge_calc") or 0)
        budget = int(build["budget_calc"])
        B = int(build["monolithic_build_calc"])
        L = int(build["leaf_build_calc"])
        output.append(
            {
                "dataset": row.get("dataset"),
                "n": row.get("n"),
                "m": row.get("m"),
                "ef_construction": row.get("ef_construction"),
                "threads": row.get("threads"),
                "algorithm": row.get("algo"),
                "n_parts": row.get("n_parts"),
                "order": row.get("order"),
                "params": canonical_params(row),
                "monolithic_build_calc": B,
                "leaf_build_calc": L,
                "budget_calc": budget,
                "merge_calc": merge_calc,
                "merge_to_budget_ratio": merge_calc / budget if budget > 0 else math.inf,
                "total_calc": L + merge_calc,
                "total_to_rebuild_ratio": (L + merge_calc) / B if B else math.nan,
                "net_advantage_calc": budget - merge_calc,
                "partition_wins": bool(merge_calc < budget),
            }
        )
    return output


def save_figure(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_build_ratio(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, group in sorted(by_dataset.items(), key=lambda item: min(r["n"] for r in item[1])):
        group.sort(key=lambda row: int(row["n_parts"]))
        ax.plot(
            [row["n_parts"] for row in group],
            [row["leaf_build_ratio"] for row in group],
            marker="o",
            label=dataset,
        )
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["n_parts"]) for row in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of partitions P")
    ax.set_ylabel("Measured leaf-build work / monolithic build work")
    ax.set_title("Partition-build work from direct leaf measurements")
    ax.legend(title="Dataset")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out, "partition_build_ratio")


def plot_budget_ratio(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, group in sorted(by_dataset.items(), key=lambda item: min(r["n"] for r in item[1])):
        group.sort(key=lambda row: int(row["n_parts"]))
        ax.plot(
            [row["n_parts"] for row in group],
            [row["budget_ratio"] for row in group],
            marker="o",
            label=dataset,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["n_parts"]) for row in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of partitions P")
    ax.set_ylabel("(B(N) - sum leaf builds) / B(N)")
    ax.set_title("Directly measured merge budget")
    ax.legend(title="Dataset")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out, "partition_budget_ratio")


def plot_budget_per_point(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["n_parts"]) > 1:
            by_dataset[row["dataset"]].append(row)
    for dataset, group in sorted(by_dataset.items(), key=lambda item: min(r["n"] for r in item[1])):
        group.sort(key=lambda row: int(row["n_parts"]))
        ax.plot(
            [math.log(int(row["n_parts"])) for row in group],
            [row["budget_per_point"] for row in group],
            marker="o",
            label=dataset,
        )
    ticks = sorted({int(row["n_parts"]) for row in rows if int(row["n_parts"]) > 1})
    ax.set_xticks([math.log(p) for p in ticks], [str(p) for p in ticks])
    ax.set_xlabel("Number of partitions P (positioned by ln P)")
    ax.set_ylabel("Build saving per point")
    ax.set_title("Test of the predicted ln(P) build-saving relation")
    ax.legend(title="Dataset")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out, "partition_budget_per_point")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-results", nargs="+", required=True)
    parser.add_argument("--merge-results", nargs="*", default=[])
    parser.add_argument("--out", default="docs/figures/build_budget")
    args = parser.parse_args(argv)

    out = Path(args.out)
    build_source = load_jsonl(expand_paths(args.build_results))
    builds = summarise_builds(build_source)
    if not builds:
        raise RuntimeError("no BUILD_ONLY rows found")

    fits = fit_budget(builds)
    merges = compare_merges(
        builds,
        load_jsonl(expand_paths(args.merge_results)) if args.merge_results else [],
    )

    write_csv(out / "build_budget.csv", builds)
    write_csv(out / "budget_ln_p_fits.csv", fits)
    write_csv(out / "merge_budget_comparison.csv", merges)

    plot_build_ratio(builds, out)
    plot_budget_ratio(builds, out)
    plot_budget_per_point(builds, out)

    print(f"wrote {out / 'build_budget.csv'}")
    print(f"wrote {out / 'budget_ln_p_fits.csv'}")
    if merges:
        print(f"wrote {out / 'merge_budget_comparison.csv'}")


if __name__ == "__main__":
    main()
