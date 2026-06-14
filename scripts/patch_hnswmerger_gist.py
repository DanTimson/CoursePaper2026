#!/usr/bin/env python3
"""Add a GIST1M workload to a cloned HNSWMerger's test_config.h (idempotent).

HNSWMerger ships SIFT/DEEP/TURING workloads only; GIST needs four small edits to
WorkloadType, its parser, its to-string, and setDefaultsByWorkload. This patches
those in place. Re-running is a no-op once GIST1M is present.

    python scripts/patch_hnswmerger_gist.py /path/to/HNSWMerger/HNSW-Merger/test_config.h

After patching, recompile:  cd HNSW-Merger && make exp && make build
"""
import re
import sys


def patch(path: str) -> None:
    src = open(path).read()
    if "GIST1M" in src:
        print("GIST1M already present — nothing to do.")
        return
    orig = src

    # 1) enum WorkloadType { SIFT1M, ... }  -> add GIST1M after SIFT1M
    src = re.sub(r"(enum\s+WorkloadType\s*\{\s*\n\s*SIFT1M,\s*\n)",
                 r"\1    GIST1M,\n", src, count=1)

    # 2) parseWorkloadType: add a branch
    src = re.sub(r'(WorkloadType\s+parseWorkloadType[^\{]*\{\s*\n)',
                 r'\1    if (s == "GIST1M") return GIST1M;\n', src, count=1)

    # 3) workloadTypeToString: add a case
    src = re.sub(r'(std::string\s+workloadTypeToString[^\{]*\{\s*\n\s*switch\s*\([^\)]*\)\s*\{\s*\n)',
                 r'\1    case GIST1M: return "GIST1M";\n', src, count=1)

    # 4) setDefaultsByWorkload: add a case block (dim 960; GIST has 1000 queries)
    src = re.sub(r'(void\s+setDefaultsByWorkload[^\{]*\{\s*\n\s*switch\s*\([^\)]*\)\s*\{\s*\n)',
                 r'\1    case GIST1M:\n'
                 r'        cfg.dim = 960;\n'
                 r'        cfg.max_elements = 1e6;\n'
                 r'        cfg.nb = 1e6;\n'
                 r'        cfg.k = 100;\n'
                 r'        cfg.kk = 100;\n'
                 r'        cfg.nq = 1000;\n'
                 r'        break;\n', src, count=1)

    if src == orig:
        print("ERROR: no anchors matched — test_config.h layout differs from expected. "
              "Patch by hand (add GIST1M to the enum, parser, to-string, and "
              "setDefaultsByWorkload with dim=960).", file=sys.stderr)
        sys.exit(1)

    edits = sum(t in src for t in ['GIST1M,', 'return GIST1M', 'return "GIST1M"', 'cfg.dim = 960'])
    if edits < 4:
        print(f"ERROR: only {edits}/4 edits applied — check the file manually.", file=sys.stderr)
        sys.exit(1)

    open(path, "w").write(src)
    print(f"Patched {path}: added GIST1M (enum, parser, to-string, defaults). "
          f"Recompile with `make exp && make build`.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    patch(sys.argv[1])
