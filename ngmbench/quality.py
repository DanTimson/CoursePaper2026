"""Shared search-quality derivations used by analysis and publication output."""
from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping
from typing import Any


def ds_at_recall(
    recall_curve: Iterable[Mapping[str, Any]] | None,
    target: float,
) -> float | None:
    """Linearly interpolate ``d_s`` at ``target`` recall.

    Invalid curve points are ignored.  At least two valid measured points must
    bracket the target; values outside the measured recall range are not
    extrapolated or replaced with a best-effort endpoint.
    """
    curve = sorted(
        (point.get("recall"), point.get("d_s"))
        for point in (recall_curve or ())
        if point.get("recall") is not None and point.get("d_s") is not None
    )
    if len(curve) < 2:
        return None

    recalls = [recall for recall, _ in curve]
    if target < recalls[0] or target > recalls[-1]:
        return None

    index = bisect.bisect_left(recalls, target)
    if index == 0:
        return curve[0][1]

    recall0, ds0 = curve[index - 1]
    recall1, ds1 = curve[index]
    if recall1 == recall0:
        return ds1
    return ds0 + (ds1 - ds0) * (target - recall0) / (recall1 - recall0)
