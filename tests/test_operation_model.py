import math

import pytest

from ngmbench.operation_model import (
    balanced_partition_sizes,
    equal_partition_crossover,
    fit_log_per_point,
    observed_comparisons,
)


def test_balanced_partition_sizes():
    assert balanced_partition_sizes(10, 3) == (4, 3, 3)
    assert sum(balanced_partition_sizes(1_000_001, 2)) == 1_000_001


def test_equal_partition_saving_identity():
    alpha = 100.0
    beta = 7.0
    samples = []
    for n in (10_000, 100_000, 1_000_000):
        samples.append((n, n * (alpha + beta * math.log(n))))
    model = fit_log_per_point(samples, label="synthetic build")
    n = 1_000_000
    p = 4
    observed = model.partition_build_saving(n, p) / n
    assert observed == pytest.approx(beta * math.log(p), rel=1e-12)


def test_observed_budget_inequality():
    common = {
        "builder": "hnswmerger",
        "dataset": "bigann10k",
        "dim": 128,
        "n": 10_000,
        "m": 16,
        "ef_construction": 200,
        "threads": 1,
        "partition_method": "range",
        "order": "balanced",
    }
    rows = [
        {
            **common,
            "algo": "INSERT",
            "n_parts": 1,
            "build_calc": 19_031_205,
            "merge_calc": 0,
            "total_calc": 19_031_205,
            "params": {},
        },
        {
            **common,
            "algo": "TWO_MERGE",
            "n_parts": 2,
            "build_calc": 15_868_251,
            "merge_calc": 1_033_751,
            "total_calc": 16_902_002,
            "params": {"merge_lambda": 4},
        },
    ]
    result = observed_comparisons(rows)[0]
    assert result.build_saving_calc == 3_162_954
    assert result.net_advantage_calc == 2_129_203
    assert result.partition_wins
    assert result.merge_to_budget_ratio == pytest.approx(1_033_751 / 3_162_954)


def test_crossover_solution():
    build = fit_log_per_point(
        [(n, n * (100 + 20 * math.log(n))) for n in (10_000, 100_000, 1_000_000)],
        label="build",
    )
    merge = fit_log_per_point(
        [(n, n * (-10 + 2 * math.log(n))) for n in (10_000, 100_000, 1_000_000)],
        label="merge",
    )
    result = equal_partition_crossover(build, merge, 2)
    expected = math.exp((20 * math.log(2) - (-10)) / 2)
    assert result.threshold_n == pytest.approx(expected)
    assert "win below" in result.interpretation
