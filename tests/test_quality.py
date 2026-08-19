from __future__ import annotations

import json
from pathlib import Path

import pytest

from ngmbench.quality import ds_at_recall


ROOT = Path(__file__).resolve().parents[1]


def test_ds_at_recall_interpolates_sorted_valid_points():
    curve = [
        {"recall": 0.98, "d_s": 300.0},
        {"recall": None, "d_s": 999.0},
        {"recall": 0.90, "d_s": 100.0},
        {"recall": 0.94, "d_s": 180.0},
        {"recall": 0.96, "d_s": None},
    ]
    assert ds_at_recall(curve, 0.95) == pytest.approx(210.0)
    assert ds_at_recall(curve, 0.90) == 100.0


def test_ds_at_recall_requires_two_points_and_a_measured_bracket():
    curve = [{"recall": 0.90, "d_s": 100.0}, {"recall": 0.94, "d_s": 180.0}]
    assert ds_at_recall(curve, 0.89) is None
    assert ds_at_recall(curve, 0.95) is None
    assert ds_at_recall([curve[0]], 0.90) is None
    assert ds_at_recall(None, 0.95) is None


def _result_record(path: str, run_key: str) -> dict:
    records = [
        json.loads(line)
        for line in (ROOT / path).read_text().splitlines()
        if line.strip()
    ]
    matches = [record for record in records if record.get("run_key") == run_key]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    "path,run_key,expected",
    [
        ("results/bigann100k.jsonl", "2e4d96d22b0e", 792.5877314606739),
        ("results/bigann1m.jsonl", "5daad962da48", 1512.7745510460245),
        ("results/bigann1m.jsonl", "9eb587f50cee", 1379.4200343511447),
    ],
)
def test_ds_at_recall_preserves_current_real_record_values(path, run_key, expected):
    record = _result_record(path, run_key)
    assert ds_at_recall(record["recall_curve"], 0.95) == pytest.approx(expected)
