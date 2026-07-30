"""Distance-computation models for HNSW build/merge experiments.

This module deliberately separates two claims:

1. Exact, finite-instance accounting from the instrumented C++ counters.
   For an observed run, partition + merge beats a monolithic build iff

       merge_calc < mono_build_calc - partition_build_calc.

2. A phenomenological log-linear model fitted to several scales:

       count(N) / N = alpha + beta * log(N).

The fitted model is useful for interpolation and crossover estimates.  It is
not a deterministic lower bound on the HNSW implementation: ef_construction,
M, and merge beam parameters cap or influence data-dependent searches, but do
not force a fixed number of distance evaluations per point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

JsonRow = Mapping[str, Any]


def balanced_partition_sizes(n: int, p: int) -> tuple[int, ...]:
    """Return the sizes of ``p`` balanced non-empty partitions of ``n`` items."""
    if n <= 0:
        raise ValueError("n must be positive")
    if p <= 0:
        raise ValueError("p must be positive")
    if p > n:
        raise ValueError("p cannot exceed n when partitions must be non-empty")
    q, r = divmod(n, p)
    return tuple(q + (1 if i < r else 0) for i in range(p))


def dataset_family(name: str) -> str:
    """Map scale-labelled datasets such as bigann10k/bigann1m to one family."""
    value = name.strip().lower()
    return re.sub(r"(?:[-_]?\d+(?:\.\d+)?[km]?)$", "", value) or value


def canonical_params(params: Mapping[str, Any] | None) -> str:
    """Stable JSON representation suitable for grouping merge configurations."""
    return json.dumps(params or {}, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class LogPerPointModel:
    """OLS fit of ``total_count / N = alpha + beta * ln(N)``."""

    label: str
    alpha: float
    beta: float
    r_squared: float
    rmse_per_point: float
    observations: int
    n_min: int
    n_max: int

    def predict_per_point(self, n: int) -> float:
        if n <= 0:
            raise ValueError("n must be positive")
        return self.alpha + self.beta * math.log(n)

    def predict_total(self, n: int, *, clamp_nonnegative: bool = True) -> float:
        value = n * self.predict_per_point(n)
        return max(0.0, value) if clamp_nonnegative else value

    def partition_build_total(self, n: int, p: int) -> float:
        """Modelled total for separately building balanced partitions."""
        return sum(self.predict_total(size) for size in balanced_partition_sizes(n, p))

    def partition_build_saving(self, n: int, p: int) -> float:
        return self.predict_total(n) - self.partition_build_total(n, p)

    def equal_partition_saving_per_point(self, p: int) -> float:
        """Exact model identity for equal leaves: beta * ln(P)."""
        if p <= 0:
            raise ValueError("p must be positive")
        return self.beta * math.log(p)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_log_per_point(
    samples: Iterable[tuple[int, int | float]],
    *,
    label: str,
) -> LogPerPointModel:
    """Fit a log-linear per-point model from ``(N, total_count)`` samples."""
    cleaned: list[tuple[int, float]] = []
    for n, total in samples:
        n_int = int(n)
        total_float = float(total)
        if n_int <= 0:
            raise ValueError(f"invalid N={n_int}")
        if total_float < 0 or not math.isfinite(total_float):
            raise ValueError(f"invalid count={total_float}")
        cleaned.append((n_int, total_float))

    if len(cleaned) < 2:
        raise ValueError("at least two scales are required for a fit")
    if len({n for n, _ in cleaned}) < 2:
        raise ValueError("at least two distinct N values are required for a fit")

    cleaned.sort()
    x = np.log(np.asarray([n for n, _ in cleaned], dtype=float))
    y = np.asarray([total / n for n, total in cleaned], dtype=float)
    design = np.column_stack((np.ones_like(x), x))
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.asarray([alpha, beta])
    residual = y - fitted
    ss_res = float(np.dot(residual, residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.dot(centered, centered))
    r_squared = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (
        0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    )
    rmse = math.sqrt(ss_res / len(cleaned))

    return LogPerPointModel(
        label=label,
        alpha=float(alpha),
        beta=float(beta),
        r_squared=float(r_squared),
        rmse_per_point=float(rmse),
        observations=len(cleaned),
        n_min=cleaned[0][0],
        n_max=cleaned[-1][0],
    )


@dataclass(frozen=True)
class ObservedComparison:
    """Exact comparison for one measured partition/merge run."""

    dataset: str
    n: int
    dim: int
    builder: str
    algorithm: str
    n_parts: int
    m: int
    ef_construction: int
    threads: int
    params: str
    mono_build_calc: int
    partition_build_calc: int
    merge_calc: int
    partition_total_calc: int
    build_saving_calc: int
    net_advantage_calc: int
    merge_to_budget_ratio: float | None
    partition_wins: bool
    run_key: str | None

    @property
    def build_saving_per_point(self) -> float:
        return self.build_saving_calc / self.n

    @property
    def merge_per_point(self) -> float:
        return self.merge_calc / self.n

    @property
    def net_advantage_per_point(self) -> float:
        return self.net_advantage_calc / self.n

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(
            build_saving_per_point=self.build_saving_per_point,
            merge_per_point=self.merge_per_point,
            net_advantage_per_point=self.net_advantage_per_point,
        )
        return row


def _base_key(row: JsonRow) -> tuple[Any, ...]:
    return (
        row.get("builder"),
        row.get("dataset"),
        int(row.get("n", 0)),
        int(row.get("dim", 0)),
        int(row.get("m", 0)),
        int(row.get("ef_construction", 0)),
        int(row.get("threads", 1)),
        row.get("partition_method"),
        row.get("order"),
    )


def observed_comparisons(rows: Iterable[JsonRow]) -> list[ObservedComparison]:
    """Pair every partition run with its matching measured monolithic INSERT.

    Returned values are exact consequences of the recorded counters.  Positive
    ``net_advantage_calc`` means partition build + merge used fewer distance
    evaluations than the matching monolithic build.
    """
    materialized = list(rows)
    monolithic: dict[tuple[Any, ...], JsonRow] = {}
    for row in materialized:
        if row.get("algo") == "INSERT" and int(row.get("n_parts", 1)) == 1:
            monolithic[_base_key(row)] = row

    output: list[ObservedComparison] = []
    for row in materialized:
        n_parts = int(row.get("n_parts", 1))
        if n_parts <= 1 or row.get("algo") == "INSERT":
            continue
        mono = monolithic.get(_base_key(row))
        if mono is None:
            continue

        mono_build = int(mono["build_calc"])
        partition_build = int(row["build_calc"])
        merge = int(row.get("merge_calc", 0))
        total = int(row.get("total_calc", partition_build + merge))
        saving = mono_build - partition_build
        advantage = mono_build - total
        ratio = (merge / saving) if saving > 0 else None

        output.append(
            ObservedComparison(
                dataset=str(row.get("dataset", "")),
                n=int(row["n"]),
                dim=int(row.get("dim", 0)),
                builder=str(row.get("builder", "")),
                algorithm=str(row.get("algo", "")),
                n_parts=n_parts,
                m=int(row.get("m", 0)),
                ef_construction=int(row.get("ef_construction", 0)),
                threads=int(row.get("threads", 1)),
                params=canonical_params(row.get("params")),
                mono_build_calc=mono_build,
                partition_build_calc=partition_build,
                merge_calc=merge,
                partition_total_calc=total,
                build_saving_calc=saving,
                net_advantage_calc=advantage,
                merge_to_budget_ratio=ratio,
                partition_wins=advantage > 0,
                run_key=row.get("run_key"),
            )
        )
    return sorted(
        output,
        key=lambda item: (
            item.dataset,
            item.ef_construction,
            item.algorithm,
            item.params,
        ),
    )


@dataclass(frozen=True)
class CrossoverEstimate:
    """Crossover implied by two fitted log-per-point models."""

    build_budget_per_point: float
    merge_alpha: float
    merge_beta: float
    threshold_n: float | None
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def equal_partition_crossover(
    build_model: LogPerPointModel,
    merge_model: LogPerPointModel,
    p: int,
) -> CrossoverEstimate:
    """Solve ``merge_pp(N) = build_beta * ln(P)`` under the fitted models."""
    if p <= 1:
        raise ValueError("p must be greater than one")
    budget = build_model.equal_partition_saving_per_point(p)
    alpha = merge_model.alpha
    beta = merge_model.beta

    if math.isclose(beta, 0.0, abs_tol=1e-12):
        wins = alpha < budget
        return CrossoverEstimate(
            build_budget_per_point=budget,
            merge_alpha=alpha,
            merge_beta=beta,
            threshold_n=None,
            interpretation=(
                "model predicts partition+merge wins at all N"
                if wins
                else "model predicts partition+merge loses at all N"
            ),
        )

    exponent = (budget - alpha) / beta
    try:
        threshold = math.exp(exponent)
    except OverflowError:
        threshold = math.inf

    if beta > 0:
        interpretation = "model predicts a win below N* and a loss above N*"
    else:
        interpretation = "model predicts a loss below N* and a win above N*"
    return CrossoverEstimate(
        build_budget_per_point=budget,
        merge_alpha=alpha,
        merge_beta=beta,
        threshold_n=threshold,
        interpretation=interpretation,
    )


def group_build_samples(rows: Iterable[JsonRow]) -> dict[tuple[Any, ...], list[tuple[int, int]]]:
    """Group monolithic INSERT samples by dataset family and build config."""
    grouped: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        if row.get("algo") != "INSERT" or int(row.get("n_parts", 1)) != 1:
            continue
        key = (
            dataset_family(str(row.get("dataset", ""))),
            row.get("builder"),
            int(row.get("dim", 0)),
            int(row.get("m", 0)),
            int(row.get("ef_construction", 0)),
            int(row.get("threads", 1)),
        )
        grouped[key].append((int(row["n"]), int(row["build_calc"])))
    return dict(grouped)


def group_merge_samples(rows: Iterable[JsonRow]) -> dict[tuple[Any, ...], list[tuple[int, int]]]:
    """Group merge counts by dataset family and exact merge configuration."""
    grouped: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        if int(row.get("n_parts", 1)) <= 1 or row.get("algo") == "INSERT":
            continue
        key = (
            dataset_family(str(row.get("dataset", ""))),
            row.get("builder"),
            int(row.get("dim", 0)),
            int(row.get("m", 0)),
            int(row.get("ef_construction", 0)),
            int(row.get("threads", 1)),
            str(row.get("algo", "")),
            int(row.get("n_parts", 1)),
            row.get("partition_method"),
            row.get("order"),
            canonical_params(row.get("params")),
        )
        grouped[key].append((int(row["n"]), int(row.get("merge_calc", 0))))
    return dict(grouped)


def deduplicate_rows(rows: Sequence[JsonRow]) -> list[dict[str, Any]]:
    """Keep the last copy of each run_key; preserve anonymous rows separately."""
    keyed: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        run_key = copy.get("run_key")
        if run_key:
            keyed[str(run_key)] = copy
        else:
            anonymous.append(copy)
    return anonymous + list(keyed.values())
