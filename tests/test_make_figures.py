import pytest

from scripts.make_figures import (
    _ds_curve_peer,
    _is_canonical,
    _shared_build_g,
    _strategy_cost,
    _qps_curve_peer,
)


def _row(algo, *, params=None, n_parts=2, order="balanced", merge_calc=100,
         total_calc=None, build_calc=10, efc=200, run_key=None):
    return {
        "builder": "hnswmerger",
        "algo": algo,
        "n_parts": n_parts,
        "order": order,
        "ef_construction": efc,
        "params": params or {},
        "merge_calc": merge_calc,
        "build_calc": build_calc,
        "total_calc": total_calc if total_calc is not None else build_calc + merge_calc,
        "run_key": run_key,
    }


def test_fasthnsw_is_distance_peer_but_not_qps_peer():
    fasthnsw = {
        "namespace": "fasthnsw-quality",
        "builder": "fastkcna-canonical",
        "algo": "FastHNSW",
        "recall_curve": [
            {
                "recall": 0.95,
                "d_s": 500.0,
                "query_distance_total": 5_000,
                "query_count": 10,
            }
        ],
    }
    layerwise = {
        "namespace": "layerwise-nnd-hnsw-quality",
        "builder": "LayerwiseNNDescentHNSW",
        "algo": "L-NND-HNSW",
        "recall_curve": [{"recall": 0.95, "d_s": 550.0}],
    }
    hnswmerger = {
        "builder": "hnswmerger",
        "algo": "TWO_MERGE",
        "n_parts": 2,
        "recall_curve": [
            {"recall": 0.95, "d_s": 600.0, "query_seconds": 0.25}
        ],
    }
    rows = [fasthnsw, layerwise, hnswmerger]

    assert [r["algo"] for r in rows if _qps_curve_peer(r)] == ["TWO_MERGE"]
    assert [r["algo"] for r in rows if _ds_curve_peer(r)] == [
        "FastHNSW",
        "L-NND-HNSW",
        "TWO_MERGE",
    ]


def test_canonical_strategy_parameters_are_explicit():
    assert _is_canonical(_row("NGM", params={"search_ef": 10}))
    assert not _is_canonical(_row("NGM", params={"search_ef": 20}))

    assert _is_canonical(_row("IGTM", params={
        "jump_ef": 5, "local_ef": 7, "next_step_k": 3,
        "next_step_ef": 3, "search_M": 5,
        # irrelevant resolved defaults must not affect identity
        "search_ef": 40, "merge_lambda": 4,
    }))
    assert not _is_canonical(_row("IGTM", params={
        "jump_ef": 40, "local_ef": 10, "next_step_k": 6,
        "next_step_ef": 6, "search_M": 40,
    }))

    assert _is_canonical(_row("CGTM", params={
        "jump_ef": 15, "local_ef": 5, "next_step_k": 3,
        "search_M": 5, "next_step_ef": 999,  # CGTM does not consume this knob
    }))
    assert not _is_canonical(_row("CGTM", params={
        "jump_ef": 20, "local_ef": 5, "next_step_k": 3, "search_M": 3,
    }))

    assert _is_canonical(_row("SIGM", params={"merge_ef_construction": -1}))
    assert not _is_canonical(_row("SIGM", params={"merge_ef_construction": 32}))


def test_hnswmerger_canonical_identity_rejects_adaptive_and_large_first():
    assert _is_canonical(_row("TWO_MERGE", params={
        "merge_lambda": 4, "merge_lambda_mode": "fixed",
    }))
    assert not _is_canonical(_row("TWO_MERGE", params={
        "merge_lambda": 4, "merge_lambda_mode": "adaptive",
    }))
    assert not _is_canonical(_row("TWO_MERGE", order="sequential", params={
        "merge_lambda": 4, "merge_lambda_mode": "fixed",
    }))


def test_strategy_cost_uses_declared_canonical_row_not_cheapest_candidate():
    rows = [
        _row("NGM", params={"search_ef": 10}, merge_calc=200, run_key="canonical"),
        _row("NGM", params={"search_ef": 20}, merge_calc=100, run_key="cheaper"),
    ]
    assert _strategy_cost(rows, "NGM") == 200


def test_distinct_duplicate_canonical_rows_fail_loudly():
    rows = [
        _row("NGM", params={"search_ef": 10}, merge_calc=200, run_key="a"),
        _row("NGM", params={"search_ef": 10}, merge_calc=190, run_key="b"),
    ]
    with pytest.raises(ValueError, match="ambiguous canonical rows"):
        _strategy_cost(rows, "NGM")


def test_same_run_key_duplicate_is_deduplicated_last_occurrence_wins():
    rows = [
        _row("NGM", params={"search_ef": 10}, merge_calc=200, run_key="same"),
        _row("NGM", params={"search_ef": 10}, merge_calc=190, run_key="same"),
    ]
    assert _strategy_cost(rows, "NGM") == 190


def test_shared_build_is_efc200_specific_and_consistent():
    rows = [
        _row("NGM", params={"search_ef": 10}, build_calc=300, efc=64, run_key="64"),
        _row("NGM", params={"search_ef": 10}, build_calc=500, efc=200, run_key="200a"),
        _row("IGTM", params={
            "jump_ef": 5, "local_ef": 7, "next_step_k": 3,
            "next_step_ef": 3, "search_M": 5,
        }, build_calc=500, efc=200, run_key="200b"),
    ]
    assert _shared_build_g(rows, 2) == pytest.approx(500 / 1e9)
