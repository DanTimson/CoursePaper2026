from scripts.make_figures import _ds_curve_peer, _qps_curve_peer


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
