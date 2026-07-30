"""C++-backend index construction: shells out to a patched HNSWMerger build.

The pure-Python merge implementations (base/merge/nndescent) were removed once the
C++ backend and the reference cross-validation (scripts/xval_python_ref.py) became
the live path; only the subprocess driver remains here.
"""
from .hnswmerger import (
    CppParams,
    Paths,
    run_hnswmerger,
    parse_builds,
    parse_exps,
    contiguous_partitions,
)

__all__ = [
    "CppParams",
    "Paths",
    "run_hnswmerger",
    "parse_builds",
    "parse_exps",
    "contiguous_partitions",
]