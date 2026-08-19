#!/usr/bin/env python3
"""Bounded deterministic pg0/pg2 validation for canonical FastKCNA counters."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from ngmbench.index.fastkcna import (
    FastKCNAParams,
    FastKCNAPaths,
    FastKCNARunner,
    prepare_lshkit,
)


def write_fvecs(path: Path, data: np.ndarray) -> None:
    payload = b"".join(
        struct.pack("<i", data.shape[1]) + np.asarray(row, dtype="<f4").tobytes()
        for row in data
    )
    path.write_bytes(payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastkcna-root", required=True)
    parser.add_argument("--workdir", default="/tmp/coursepaper-fastkcna-distance-validation")
    parser.add_argument("--multithread", type=int, default=4)
    args = parser.parse_args(argv)
    if args.multithread <= 1:
        parser.error("--multithread must be greater than one")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    # Arithmetic data avoids dependence on an external dataset or RNG implementation.
    base = np.asarray(
        [[((i * 17 + j * 7) % 101) / 10.0 for j in range(8)] for i in range(64)],
        dtype="<f4",
    )
    source = workdir / "tiny64.fvecs"
    write_fvecs(source, base)
    paths = FastKCNAPaths.resolve({"checkout": args.fastkcna_root})
    conversion = prepare_lshkit(source, workdir / "tiny64.lshkit", paths)
    common = dict(
        K=10, L=12, S=4, R=10, iter=2, search_L=8, search_K=12,
        nsg_R=4, step=2, loop_i=1, alpha=60, tau=0,
        controls=5, recall=0.98,
    )
    summary = {"scope": "tiny canonical distance validation only", "runs": []}
    for pg_type in (0, 2):
        for nthreads in (1, args.multithread):
            params = FastKCNAParams(pg_type=pg_type, nthreads=nthreads, **common)
            record = FastKCNARunner(paths, workdir / "runs").run(
                workdir / "tiny64.lshkit", params, f"pg{pg_type}-t{nthreads}",
                conversion=conversion, require_canonical=True,
            )
            total = record["build_calc"]
            assert total > 0
            assert sum(record["distance_counts_by_phase"].values()) == total
            assert sum(record["distance_counts_by_layer"].values()) == total
            assert record["merge_calc"] == 0 and record["total_calc"] == total
            summary["runs"].append({
                "pg_type": pg_type, "nthreads": nthreads, "total": total,
                "phases": record["distance_counts_by_phase"],
                "layers": record["distance_counts_by_layer"],
                "diagnostic_upstream_n_comps": record[
                    "canonical_fastkcna_distance_counts"
                ].get("diagnostic_upstream_n_comps"),
            })
    output = workdir / "distance_validation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "summary": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
