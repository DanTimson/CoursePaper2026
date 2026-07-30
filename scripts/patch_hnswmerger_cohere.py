#!/usr/bin/env python3
"""Add a COHERE1M workload to a cloned HNSWMerger's test_config.h (idempotent).

Mirrors patch_hnswmerger_gist.py. Cohere/wikipedia-22-12 embeddings are 768-d, L2,
with 1000 held-out queries. Four edits: WorkloadType enum, its parser, its
to-string, and setDefaultsByWorkload.

    python scripts/patch_hnswmerger_cohere.py /path/to/HNSW-Merger/test_config.h
    cd HNSW-Merger && make exp && make build
"""
import re
import sys

DIM, NQ = 768, 1000
NAME = "COHERE1M"


def main(path):
    src = open(path).read()
    if NAME in src:
        print(f"{NAME} already present — nothing to do."); return
    orig = src
    src = re.sub(r"(enum\s+WorkloadType\s*\{\s*\n\s*SIFT1M,\s*\n)",
                 rf"\1    {NAME},\n", src, count=1)
    src = re.sub(r'(WorkloadType\s+\w+\s*\([^)]*\)\s*\{[^}]*?SIFT1M[^}]*?\n)',
                 rf'\1    if (s == "{NAME}") return {NAME};\n', src, count=1)
    src = re.sub(r'(case\s+SIFT1M:\s*return\s*"SIFT1M";\s*\n)',
                 rf'\1    case {NAME}: return "{NAME}";\n', src, count=1)
    src = re.sub(r'(void\s+setDefaultsByWorkload[^\{]*\{\s*\n\s*switch\s*\([^\)]*\)\s*\{\s*\n)',
                 rf'\1    case {NAME}:\n'
                 rf'        cfg.dim = {DIM};\n'
                 r'        cfg.max_elements = 1e6;\n'
                 r'        cfg.nb = 1e6;\n'
                 r'        cfg.k = 100;\n'
                 r'        cfg.kk = 100;\n'
                 rf'        cfg.nq = {NQ};\n'
                 r'        break;\n', src, count=1)
    if src == orig:
        print("ERROR: no anchors matched — test_config.h layout differs. Patch by "
              f"hand (add {NAME} to enum, parser, to-string, setDefaultsByWorkload "
              f"with dim={DIM}).", file=sys.stderr)
        sys.exit(1)
    edits = sum(t in src for t in [f'{NAME},', f'return {NAME}',
                                   f'return "{NAME}"', f'cfg.dim = {DIM}'])
    if edits < 4:
        print(f"ERROR: only {edits}/4 edits applied — check the file manually.",
              file=sys.stderr)
        sys.exit(1)
    open(path, "w").write(src)
    print(f"Patched {NAME} into {path}. Recompile: make exp && make build")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
