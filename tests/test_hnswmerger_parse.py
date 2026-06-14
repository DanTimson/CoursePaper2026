"""Validate the HNSWMerger stdout parsers against real captured output.

The IGTM and NGM blocks below are verbatim from actual ./exps runs (200k SIFT
slice). Run: pytest -q tests/test_hnswmerger_parse.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from ngmbench.index.hnswmerger import parse_exps, parse_builds

IGTM_OUT = """Configuration:
  Merge Method: IGTM
Merge method: IGTM
Iteration 1/1
time for new layer: 0.396677
Total time for insertion: 17.430 s
distance calls = 104428612
Index not saved, rerun with save_index flag in config to save the index.
Start searching
set ef = 10
[search time: 0.410 s, pure query time: 0.408 s] ef=10
Intersection-merged index R@100 = 0.1392
[search time: 0.402 s, pure query time: 0.401 s] ef=10
Intersection-merged index R@100 = 0.1392
[search time: 0.422 s, pure query time: 0.421 s] ef=10
Intersection-merged index R@100 = 0.1392
set ef = 50
[search time: 1.283 s, pure query time: 1.281 s] ef=50
Intersection-merged index R@100 = 0.1831
[search time: 1.279 s, pure query time: 1.277 s] ef=50
Intersection-merged index R@100 = 0.1831
[search time: 1.282 s, pure query time: 1.280 s] ef=50
Intersection-merged index R@100 = 0.1831
set ef = 100
[search time: 2.254 s, pure query time: 2.251 s] ef=100
Intersection-merged index R@100 = 0.1890
[search time: 2.222 s, pure query time: 2.219 s] ef=100
Intersection-merged index R@100 = 0.1890
[search time: 2.840 s, pure query time: 2.837 s] ef=100
Intersection-merged index R@100 = 0.1890
"""

NGM_OUT = """Configuration:
  Merge Method: NGM
Merge method: NGM
Total time for insertion: 34.952 s
distance calls = 203166328
set ef = 10
[search time: 0.520 s, pure query time: 0.518 s] ef=10
Intersection-merged index R@100 = 0.1535
"""

BUILDS_OUT = """loaded base vectors: 1000000 vectors of dimension 128
[1.234 s] build index (dataset size = 0 - 100000)
distance calls = 5551234
"""


def test_igtm_parse():
    r = parse_exps(IGTM_OUT, expect_method="IGTM")
    assert r["merge_method"] == "IGTM"
    assert r["merge_calc"] == 104428612
    assert r["merge_seconds"] == 17.430
    assert len(r["recall_curve"]) == 3                      # ef 10/50/100
    c = {x["ef"]: x for x in r["recall_curve"]}
    assert c[10]["recall"] == pytest.approx(0.1392)
    assert c[100]["recall"] == pytest.approx(0.1890)
    # 3 timing repeats averaged
    assert c[10]["query_seconds"] == pytest.approx((0.408 + 0.401 + 0.421) / 3, abs=1e-3)


def test_ngm_parse_and_costs_more():
    r = parse_exps(NGM_OUT, expect_method="NGM")
    assert r["merge_method"] == "NGM"
    assert r["merge_calc"] == 203166328
    assert r["merge_calc"] > 104428612                      # the falsification check


def test_method_mismatch_raises():
    with pytest.raises(ValueError):
        parse_exps(IGTM_OUT, expect_method="NGM")           # catches a mislabeled config


def test_builds_parse():
    b = parse_builds(BUILDS_OUT)
    assert b["build_calc"] == 5551234
    assert b["build_seconds"] == 1.234
