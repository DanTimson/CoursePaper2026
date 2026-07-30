#!/usr/bin/env python3
"""Analyse exact and fitted total distance-computation budgets.

Example:
    python scripts/analyse_total_operations.py \
      --results results/bigann10k.jsonl results/bigann100k.jsonl \
                results/bigann1m.jsonl results/bigann10m.jsonl \
      --out results/operation_analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any

# Allow direct execution from a checkout without requiring installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ngmbench.operation_model import (  # noqa: E402
    canonical_params,
    deduplicate_rows,
    equal_partition_crossover,
    fit_log_per_point,
    group_build_samples,
    group_merge_samples,
    observed_comparisons,
)


def _expand_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        matches = [Path(value) for value in glob.glob(item)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(item)
            if path.exists():
                paths.append(path)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("no result files matched --results")
    return unique


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no}: expected a JSON object")
                rows.append(value)
    return deduplicate_rows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_int(value: int | float) -> str:
    return f"{int(round(value)):,}"


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _short_params(params_json: str) -> str:
    params = json.loads(params_json)
    relevant = [
        "merge_lambda",
        "jump_ef",
        "local_ef",
        "next_step_k",
        "next_step_ef",
        "search_M",
        "search_ef",
        "merge_ef_construction",
    ]
    return ", ".join(f"{name}={params[name]}" for name in relevant if name in params)


def fit_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    build_fits: dict[tuple[Any, ...], Any] = {}
    build_output: list[dict[str, Any]] = []
    for key, samples in group_build_samples(rows).items():
        if len({n for n, _ in samples}) < 2:
            continue
        family, builder, dim, m, efc, threads = key
        model = fit_log_per_point(samples, label=f"{family}:build:M={m}:efc={efc}")
        build_fits[key] = model
        build_output.append(
            {
                "dataset_family": family,
                "builder": builder,
                "dim": dim,
                "m": m,
                "ef_construction": efc,
                "threads": threads,
                **model.to_dict(),
            }
        )

    merge_output: list[dict[str, Any]] = []
    for key, samples in group_merge_samples(rows).items():
        if len({n for n, _ in samples}) < 2:
            continue
        (
            family,
            builder,
            dim,
            m,
            efc,
            threads,
            algo,
            n_parts,
            partition_method,
            order,
            params,
        ) = key
        merge_model = fit_log_per_point(
            samples,
            label=f"{family}:merge:{algo}:M={m}:efc={efc}:P={n_parts}",
        )
        build_key = (family, builder, dim, m, efc, threads)
        build_model = build_fits.get(build_key)
        crossover = (
            equal_partition_crossover(build_model, merge_model, n_parts)
            if build_model is not None and n_parts > 1
            else None
        )
        row: dict[str, Any] = {
            "dataset_family": family,
            "builder": builder,
            "dim": dim,
            "m": m,
            "ef_construction": efc,
            "threads": threads,
            "algorithm": algo,
            "n_parts": n_parts,
            "partition_method": partition_method,
            "order": order,
            "params": params,
            **merge_model.to_dict(),
        }
        if crossover is not None:
            row.update(
                build_beta=build_model.beta,
                build_budget_per_point=crossover.build_budget_per_point,
                crossover_n=crossover.threshold_n,
                crossover_interpretation=crossover.interpretation,
            )
        merge_output.append(row)
    return build_output, merge_output


def render_markdown(
    paths: list[Path],
    comparisons: list[dict[str, Any]],
    build_fits: list[dict[str, Any]],
    merge_fits: list[dict[str, Any]],
) -> str:
    winners = [row for row in comparisons if row["partition_wins"]]
    lines = [
        "# Total operation analysis",
        "",
        "This report treats recorded `*_calc` values as exact finite-instance distance-call counts. "
        "The log-linear fits are descriptive/extrapolative models, not implementation-level lower bounds.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    lines.extend(
        [
            "",
            "## Exact measured inequality",
            "",
            "For every partition run, the exact condition is:",
            "",
            "`merge_calc < mono_build_calc - partition_build_calc`.",
            "",
            f"Measured comparisons: **{len(comparisons)}**; partition+merge wins: **{len(winners)}**.",
            "",
            "| dataset | M | efc | algorithm | P | merge params | build saving/pt | merge/pt | net/pt | result |",
            "|---|---:|---:|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        comparisons,
        key=lambda value: (
            value["n"],
            value["ef_construction"],
            value["algorithm"],
            value["params"],
        ),
    ):
        result = "WIN" if row["partition_wins"] else "LOSS"
        lines.append(
            "| {dataset} | {m} | {ef_construction} | {algorithm} | {n_parts} | {params} | "
            "{saving} | {merge} | {net} | {result} |".format(
                dataset=row["dataset"],
                m=row["m"],
                ef_construction=row["ef_construction"],
                algorithm=row["algorithm"],
                n_parts=row["n_parts"],
                params=_short_params(row["params"]),
                saving=_fmt_float(row["build_saving_per_point"]),
                merge=_fmt_float(row["merge_per_point"]),
                net=_fmt_float(row["net_advantage_per_point"]),
                result=result,
            )
        )

    lines.extend(
        [
            "",
            "## Fitted build models",
            "",
            "Model: `build_calc / N = alpha + beta * ln(N)`.",
            "",
            "| family | M | efc | alpha | beta | R² | RMSE/pt | N range |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(build_fits, key=lambda value: (value["dataset_family"], value["m"], value["ef_construction"])):
        lines.append(
            "| {dataset_family} | {m} | {ef_construction} | {alpha} | {beta} | {r_squared} | "
            "{rmse} | {n_min}–{n_max} |".format(
                dataset_family=row["dataset_family"],
                m=row["m"],
                ef_construction=row["ef_construction"],
                alpha=_fmt_float(row["alpha"]),
                beta=_fmt_float(row["beta"]),
                r_squared=_fmt_float(row["r_squared"], 4),
                rmse=_fmt_float(row["rmse_per_point"]),
                n_min=_fmt_int(row["n_min"]),
                n_max=_fmt_int(row["n_max"]),
            )
        )

    lines.extend(
        [
            "",
            "## Fitted merge crossovers",
            "",
            "For equal leaves, the fitted build saving per point is `build_beta * ln(P)`. "
            "The reported N* solves `merge_alpha + merge_beta * ln(N*) = build_beta * ln(P)`.",
            "",
            "| family | M | efc | algorithm | P | params | merge alpha | merge beta | budget/pt | N* | R² |",
            "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        merge_fits,
        key=lambda value: (
            value["dataset_family"],
            value["m"],
            value["ef_construction"],
            value["algorithm"],
            value["params"],
        ),
    ):
        lines.append(
            "| {dataset_family} | {m} | {ef_construction} | {algorithm} | {n_parts} | {params} | "
            "{alpha} | {beta} | {budget} | {cross} | {r_squared} |".format(
                dataset_family=row["dataset_family"],
                m=row["m"],
                ef_construction=row["ef_construction"],
                algorithm=row["algorithm"],
                n_parts=row["n_parts"],
                params=_short_params(row["params"]),
                alpha=_fmt_float(row["alpha"]),
                beta=_fmt_float(row["beta"]),
                budget=_fmt_float(row.get("build_budget_per_point")),
                cross=_fmt_int(row["crossover_n"]) if row.get("crossover_n") is not None and math.isfinite(row["crossover_n"]) else "—",
                r_squared=_fmt_float(row["r_squared"], 4),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- A measured WIN/LOSS is exact for that dataset ordering, random graph realization, code revision, and configuration.",
            "- N* is model-dependent and should be reported with the fitted range and residual error; it is not a proof of asymptotic behavior.",
            "- A theorem that merge must eventually lose would additionally need a justified positive lower bound on merge work per point that grows with ln(N). The current implementation's stopping conditions are data-dependent, so the configuration values alone do not provide that bound.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        nargs="+",
        default=["results/bigann*.jsonl"],
        help="JSONL files or glob patterns",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/operation_analysis"),
        help="output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = _expand_paths(args.results)
    rows = load_jsonl(paths)
    comparisons = [item.to_dict() for item in observed_comparisons(rows)]
    build_fits, merge_fits = fit_rows(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "observed_total_budget.csv", comparisons)
    write_csv(args.out / "build_model_fits.csv", build_fits)
    write_csv(args.out / "merge_model_fits.csv", merge_fits)
    report = render_markdown(paths, comparisons, build_fits, merge_fits)
    (args.out / "report.md").write_text(report, encoding="utf-8")

    winners = sum(bool(row["partition_wins"]) for row in comparisons)
    print(f"loaded {len(rows)} unique runs from {len(paths)} files")
    print(f"paired {len(comparisons)} partition runs; {winners} total-operation wins")
    print(f"wrote {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
