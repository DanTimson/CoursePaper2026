"""Adapter for the HNSWMerger C++ tool (Kimchuls/HNSWMerger), used as upstream.

We drive its two binaries via generated key=value config files:

  builds  (build_index.cpp)  -- builds one HNSW over a contiguous id-range,
                                 points labelled by global row id; prints its
                                 own `distance calls`.
  exps    (experiment.cpp)   -- merges exactly TWO indexes (for NGM/IGTM/CGTM/
                                 ES/TWO_MERGE), prints merge time, `distance
                                 calls`, and a recall-vs-ef sweep.

Because NGM/IGTM/CGTM are strictly pairwise, divide-and-conquer over k>2
partitions is driven here: build k leaves, then merge pairwise with
save_index=true, feeding each saved result back in. merge_calc/merge_seconds are
summed across the pairwise steps; build_calc/build_seconds across the leaves.

Partitioning is by contiguous id-range (HNSWMerger's own approach), so the C++
side needs no relabeling — labels are global row ids throughout.

Parsers below are validated against real stdout in tests/test_hnswmerger_parse.py.
"""
from __future__ import annotations

import os
import json
import math
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# paper algo name -> HNSWMerger merge_method enum
ALGO_TO_METHOD = {
    "NGM": "NGM", "IGTM": "IGTM", "CGTM": "CGTM",
    "ES": "ES", "ELASTIC": "ES",
    "TWO_MERGE": "TWO_MERGE", "HNSW-MERGER": "TWO_MERGE",
    "SIGM": "INSERT", "INSERT": "INSERT", "REBUILD": "REBUILD",
}

# workload_type providing dim/k/kk/nq defaults; we still set dim/nb explicitly
# dim -> upstream workload enum name (provides metric + k/kk/nq defaults; dim/nb
# are still set explicitly per run). SIFT/DEEP/TURING ship in upstream test_config.h;
# GIST and COHERE require the corresponding scripts/patch_hnswmerger_*.py patch.
# NOTE: verify these enum spellings against your test_config.h - upstream uses the
# "<NAME>1M" convention (SIFT1M, not SIFT10M); a wrong string silently falls back.
WORKLOAD = {128: "SIFT1M", 96: "DEEP10M", 100: "TURING10M",
            960: "GIST1M", 768: "COHERE1M"}


# --------------------------------------------------------------------------- #
# stdout parsers                                                              #
# --------------------------------------------------------------------------- #
_RE_METHOD = re.compile(r"Merge [Mm]ethod:\s*(\w+)")
_RE_DIST = re.compile(r"distance calls\s*=\s*(\d+)")
_RE_MERGE_T = re.compile(r"Total time for insertion:\s*([\d.]+)\s*s")
_RE_BUILD_T = re.compile(r"\[([\d.]+)\s*s\]\s*build index")
_RE_EF = re.compile(r"set ef\s*=\s*(\d+)")
_RE_QT = re.compile(r"pure query time:\s*([\d.]+)\s*s\]\s*ef=(\d+)")
_RE_RECALL = re.compile(r"R@\d+\s*=\s*([\d.]+)")   # label says R@100 but value is R@k
_RE_DS = re.compile(r"search distance calls per query\s*=\s*([\d.]+)")   # iso-quality metric d_s


def parse_builds(stdout: str) -> Dict:
    """Parse one ./builds run -> {build_calc, build_seconds}."""
    dist = _RE_DIST.search(stdout)
    bt = _RE_BUILD_T.search(stdout)
    return {
        "build_calc": int(dist.group(1)) if dist else None,
        "build_seconds": float(bt.group(1)) if bt else None,
    }


def parse_exps(stdout: str, expect_method: Optional[str] = None) -> Dict:
    """Parse one ./exps run.

    Returns merge_method (echoed), merge_calc, merge_seconds, and the recall-vs-ef
    curve [{ef, recall, query_seconds}] with the 3 fixed timing repeats averaged.
    Raises if the echoed method disagrees with expect_method (catches mislabeled
    config files like an IGTM config left in an 'ngm' file).
    """
    method = _RE_METHOD.search(stdout)
    method = method.group(1) if method else None
    if expect_method and method and method != expect_method:
        raise ValueError(f"HNSWMerger ran {method!r} but expected {expect_method!r} "
                         f"-- check the merge_method line in the generated config")
    dist = _RE_DIST.search(stdout)
    mt = _RE_MERGE_T.search(stdout)

    # build the recall/timing curve: walk lines, tracking current ef
    curve: Dict[int, Dict[str, list]] = {}
    cur_ef = None
    for line in stdout.splitlines():
        m = _RE_EF.search(line)
        if m:
            cur_ef = int(m.group(1)); curve.setdefault(cur_ef, {"qt": [], "rec": [], "ds": []})
            continue
        q = _RE_QT.search(line)
        if q:
            curve.setdefault(int(q.group(2)), {"qt": [], "rec": [], "ds": []})["qt"].append(float(q.group(1)))
            continue
        d = _RE_DS.search(line)
        if d and cur_ef is not None:
            curve[cur_ef]["ds"].append(float(d.group(1)))
            continue
        r = _RE_RECALL.search(line)
        if r and cur_ef is not None:
            curve[cur_ef]["rec"].append(float(r.group(1)))
    recall_curve = [
        {"ef": ef,
         "recall": float(np.mean(v["rec"])) if v["rec"] else None,
         "query_seconds": float(np.mean(v["qt"])) if v["qt"] else None,
         "d_s": float(np.mean(v["ds"])) if v["ds"] else None}
        for ef, v in sorted(curve.items())
    ]
    return {
        "merge_method": method,
        "merge_calc": int(dist.group(1)) if dist else None,
        "merge_seconds": float(mt.group(1)) if mt else None,
        "recall_curve": recall_curve,
    }


# --------------------------------------------------------------------------- #
# config writers                                                              #
# --------------------------------------------------------------------------- #
def _write_kv(path: str, kv: Dict) -> None:
    with open(path, "w") as f:
        for k, v in kv.items():
            f.write(f"{k} = {v}\n")


@dataclass
class Paths:
    builds_bin: str          # .../HNSW-Merger/builds
    exps_bin: str            # .../HNSW-Merger/exps
    base: str                # sift_base.fvecs
    query: str               # sift_query.fvecs
    groundtruth: str         # sift_groundtruth.ivecs
    workdir: str             # scratch dir for indexes + configs

    def __post_init__(self):
        # Normalise every path once, here, so configs stay portable:
        #   $VARS   -> os.path.expandvars  (e.g. $HNSWMERGER_BIN for the separate
        #             binaries clone; repo-internal paths need no var and stay
        #             relative)
        #   ~       -> expanduser
        #   relative-> resolved against cwd (the repo root)
        # This matters because _run executes the binaries with cwd=dirname(binary),
        # inside the HNSWMerger tree, so a bare relative path would otherwise
        # resolve against the wrong repo.
        for f in ("builds_bin", "exps_bin", "base", "query", "groundtruth", "workdir"):
            v = getattr(self, f)
            if v:
                v = os.path.expandvars(os.path.expanduser(v))
                if "$" in v:
                    raise ValueError(
                        f"Paths.{f}: unresolved environment variable in {v!r}. "
                        f"Set it (e.g. export HNSWMERGER_BIN=/path/to/HNSW-Merger) "
                        f"or see .env.example.")
                setattr(self, f, os.path.abspath(v))
        for f in ("base", "query", "groundtruth"):
            v = getattr(self, f)
            if not os.path.exists(v):
                raise FileNotFoundError(
                    f"Paths.{f} does not exist: {v}\n"
                    f"  (resolved from the config against cwd={os.getcwd()})")


@dataclass
class CppParams:
    dim: int
    nb: int                  # full base size
    M: int = 16
    ef_construction: int = 200
    k: int = 10
    kk: int = 100
    nq: int = 10000
    efs_array: List[int] = field(default_factory=lambda: [10, 50, 100, 200])
    thread: int = 1                  # drives OMP_NUM_THREADS *and* the cfg thread key
    # merge knobs. NGM uses search_ef; IGTM uses all five;
    # CGTM's signature has NO next_step_ef; SIGM uses merge_ef_construction.
    jump_ef: int = 40
    local_ef: int = 10
    next_step_k: int = 6
    next_step_ef: int = 6
    search_M: int = 40
    search_ef: int = 40
    merge_ef_construction: int = -1  # -1 = inherit from the loaded index
    merge_lambda: int = 4            # HNSWMerger initial/fixed forward-search width
    merge_lambda_mode: str = "fixed" # fixed | adaptive (paper multi-index rule)
    cleanup_merged: bool = False       # preserve old behavior unless a config opts in

    def merge_kv(self) -> dict:
        """cfg keys for the merge/insert phase. Harmless for algos that ignore them."""
        return {"jump_ef": self.jump_ef, "local_ef": self.local_ef,
                "next_step_k": self.next_step_k, "next_step_ef": self.next_step_ef,
                "search_M": self.search_M, "search_ef": self.search_ef,
                "merge_ef_construction": self.merge_ef_construction,
                "lambda": self.merge_lambda,
                "thread": self.thread}

    def merge_id(self) -> dict:
        """Identity of this parameter point — goes into run_key and the row."""
        return {"jump_ef": self.jump_ef, "local_ef": self.local_ef,
                "next_step_k": self.next_step_k, "next_step_ef": self.next_step_ef,
                "search_M": self.search_M, "search_ef": self.search_ef,
                "merge_ef_construction": self.merge_ef_construction,
                "merge_lambda": self.merge_lambda,
                "merge_lambda_mode": self.merge_lambda_mode}


def adaptive_merge_lambda(
    lambda0: int,
    max_lambda: int,
    current_larger_n: int,
    initial_larger_n: int,
) -> int:
    """Paper-inspired adaptive lambda for a sequence of HNSW-Merger operations.

    Section 7.2 of Jin et al. interpolates linearly in log(index size) between
    (N0, lambda0) and (max_lambda * N0, max_lambda).  For the P<=16, M=16
    experiment the reset-after-reaching-M rule is never triggered, so a stateless
    capped interpolation is sufficient.

    The returned value is rounded to the nearest integer and clamped to
    [lambda0, max_lambda].
    """
    lambda0 = int(lambda0)
    max_lambda = int(max_lambda)
    current_larger_n = int(current_larger_n)
    initial_larger_n = int(initial_larger_n)
    if lambda0 <= 0:
        raise ValueError("lambda0 must be positive")
    if max_lambda <= 0:
        raise ValueError("max_lambda must be positive")
    if initial_larger_n <= 0 or current_larger_n <= 0:
        raise ValueError("index sizes must be positive")
    if max_lambda <= lambda0 or current_larger_n <= initial_larger_n:
        return min(lambda0, max_lambda)

    ratio = current_larger_n / initial_larger_n
    fraction = min(1.0, max(0.0, math.log(ratio) / math.log(max_lambda)))
    value = lambda0 + (max_lambda - lambda0) * fraction
    return max(lambda0, min(max_lambda, int(round(value))))


# --------------------------------------------------------------------------- #
# runner                                                                       #
# --------------------------------------------------------------------------- #
class HNSWMergerRunner:
    def __init__(self, paths: Paths, params: CppParams, env_threads: Optional[int] = None):
        self.p = paths
        self.cp = params
        self.workload = WORKLOAD.get(params.dim, "SIFT1M")
        os.makedirs(paths.workdir, exist_ok=True)
        paths.workdir = os.path.abspath(paths.workdir)   # binaries run with cwd=binary dir
        self.env = dict(os.environ)
        threads = env_threads or getattr(params, "thread", None)
        if threads:
            self.env["OMP_NUM_THREADS"] = str(threads)

    def _run(self, binary: str, cfg_path: str) -> str:
        r = subprocess.run([binary, os.path.abspath(cfg_path)],
                           cwd=os.path.dirname(binary) or ".",
                           capture_output=True, text=True, env=self.env)
        if r.returncode != 0:
            raise RuntimeError(f"{binary} failed:\n{r.stdout}\n{r.stderr}")
        return r.stdout

    def build_leaf(self, lrange: int, rrange: int) -> Tuple[str, Dict]:
        # M, ef_construction, and construction thread count are part of the leaf
        # cache key: distinct thread-matrix points must each build from empty, and
        # each artifact carries its own build_calc in the sidecar .meta.json.
        tag = f"_M{self.cp.M}_efc{self.cp.ef_construction}_t{self.cp.thread}"
        idx = os.path.join(self.p.workdir, f"leaf_{lrange}_{rrange}{tag}.hnsw")
        meta = idx + ".meta.json"
        if os.path.exists(idx) and os.path.exists(meta):
            with open(meta) as f:                      # carry forward the real counts
                return idx, {**json.load(f), "cached": True}
        cfg = os.path.join(self.p.workdir, f"build_{lrange}_{rrange}{tag}.cfg")
        _write_kv(cfg, {
            "dim": self.cp.dim, "max_elements": rrange - lrange, "nb": self.cp.nb,
            "M": self.cp.M, "ef_construction": self.cp.ef_construction,
            "lrange": lrange, "rrange": rrange,
            "base_filepath": self.p.base, "index_path": idx,
        })
        b = parse_builds(self._run(self.p.builds_bin, cfg))
        with open(meta, "w") as f:                     # so a later algo reusing this leaf gets real counts
            json.dump({"build_calc": b["build_calc"], "build_seconds": b["build_seconds"]}, f)
        return idx, b

    def query_only(self, idx: str, method: str, total_n: int) -> Dict:
        """Run ./exps on an existing single index purely to get its recall curve.

        Upstream's INSERT branch returns before the common query loop. REBUILD
        with rerun=false is the generic load-existing-index path and falls
        through to that loop, so use it as an implementation detail regardless
        of the caller's logical construction method.
        """
        cfg = os.path.join(self.p.workdir, f"query_{uuid.uuid4().hex[:8]}.cfg")
        _write_kv(cfg, {
            "workload_type": self.workload, "merge_method": "REBUILD",
            "dim": self.cp.dim, "max_elements": total_n, "nb": total_n,
            "M": self.cp.M, "ef_construction": self.cp.ef_construction,
            "k": self.cp.k, "kk": self.cp.kk, "nq": self.cp.nq,
            "iterations": 1, "rerun": "false", "save_index": "false",
            "base_filepath": self.p.base, "query_filepath": self.p.query,
            "groundtruth_filepath": self.p.groundtruth,
            "index_path": idx, "save_path": self.p.workdir,
            "efs_array": ", ".join(str(e) for e in self.cp.efs_array),
            **self.cp.merge_kv(),
        })
        out = parse_exps(
            self._run(self.p.exps_bin, cfg),
            expect_method="REBUILD",
        )
        out["merge_method"] = method
        return out

    def merge_pair(self, idx_a: str, idx_b: str, method: str,
                   efs: List[int], total_n: int,
                   merge_lambda: Optional[int] = None) -> Tuple[str, Dict]:
        cfg = os.path.join(self.p.workdir, f"merge_{uuid.uuid4().hex[:8]}.cfg")
        save_dir = self.p.workdir
        merge_kv = self.cp.merge_kv()
        if merge_lambda is not None:
            merge_kv["lambda"] = int(merge_lambda)
        _write_kv(cfg, {
            "workload_type": self.workload, "merge_method": method,
            "dim": self.cp.dim, "max_elements": total_n, "nb": total_n,
            "M": self.cp.M, "ef_construction": self.cp.ef_construction,
            "k": self.cp.k, "kk": self.cp.kk, "nq": self.cp.nq,
            "iterations": 1, "rerun": "true", "save_index": "true",
            "base_filepath": self.p.base, "query_filepath": self.p.query,
            "groundtruth_filepath": self.p.groundtruth,
            "index_path": f"{idx_a},{idx_b}", "save_path": save_dir,
            "efs_array": ", ".join(str(e) for e in efs),
            **merge_kv,
        })
        out = parse_exps(self._run(self.p.exps_bin, cfg), expect_method=method)
        # ./exps saves to save_dir/<method>_<workload>.hnsw (fixed name) -> rename unique
        produced = self._saved_index_name(method)
        merged = os.path.join(self.p.workdir, f"merged_{uuid.uuid4().hex[:8]}.hnsw")
        if os.path.exists(produced):
            shutil.move(produced, merged)
            # opt-in: also keep a self-describing copy for graph_structure analysis.
            # Set NGMBENCH_DUMP_DIR to a directory; filename encodes method + params,
            # so no uuid reverse-engineering is needed later.
            dump_dir = os.environ.get("NGMBENCH_DUMP_DIR")
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                pid = self.cp.merge_id()
                tag = ",".join(f"{k}{v}" for k, v in sorted(pid.items()) if v != -1) or "default"
                safe = tag.replace(" ", "")
                name = f"{method}_{self.workload}_M{self.cp.M}_efc{self.cp.ef_construction}_{safe}.hnsw"
                shutil.copy2(merged, os.path.join(dump_dir, name))
        return merged, out

    def _saved_index_name(self, method: str) -> str:
        # C++ save names per branch (experiment.cpp): ngm_/igtm_/cgtm_/es_ match
        # the stem, but TWO_MERGE (HNSWMerger) writes the generic "merged-index_".
        if method == "TWO_MERGE":
            return os.path.join(self.p.workdir, f"merged-index_{self.workload}.hnsw")
        stem = {"NGM": "ngm", "IGTM": "igtm", "CGTM": "cgtm", "ES": "es"}.get(
            method, method.lower())
        return os.path.join(self.p.workdir, f"{stem}_{self.workload}.hnsw")

    def divide_and_conquer(
        self,
        leaves: List[str],
        method: str,
        order: str,
        total_n: int,
        leaf_sizes: Optional[List[int]] = None,
    ) -> Dict:
        """Merge all leaves and retain an auditable record of every pairwise step.

        ``balanced`` is the natural divide-and-conquer tree.  For equal leaves,
        ``sequential`` is the large-first chain recommended by HNSW-Merger for
        minimizing total merge work: after the first pair, the growing accumulator
        is repeatedly merged with one remaining leaf.

        Intermediate merged files are deleted as soon as both have been consumed.
        Cached leaf indexes are never deleted.
        """
        if leaf_sizes is None:
            leaf_sizes = [total_n // len(leaves)] * len(leaves)
            leaf_sizes[-1] += total_n - sum(leaf_sizes)
        if len(leaves) != len(leaf_sizes):
            raise ValueError("leaves and leaf_sizes must have the same length")
        if not leaves:
            raise ValueError("at least one leaf is required")

        merge_calc = 0
        merge_seconds = 0.0
        final_curve = None
        merge_steps: List[Dict] = []
        initial_larger_n = max(int(n) for n in leaf_sizes)

        nodes = [
            {"path": path, "n": int(size), "temporary": False}
            for path, size in zip(leaves, leaf_sizes)
        ]

        def remove_consumed(node: Dict) -> None:
            if (
                self.cp.cleanup_merged
                and node["temporary"]
                and os.path.exists(node["path"])
            ):
                os.remove(node["path"])

        def lambda_for(a: Dict, b: Dict) -> Optional[int]:
            if method != "TWO_MERGE":
                return None
            mode = (self.cp.merge_lambda_mode or "fixed").lower()
            if mode == "fixed":
                return int(self.cp.merge_lambda)
            if mode == "adaptive":
                return adaptive_merge_lambda(
                    lambda0=self.cp.merge_lambda,
                    max_lambda=self.cp.M,
                    current_larger_n=max(a["n"], b["n"]),
                    initial_larger_n=initial_larger_n,
                )
            raise ValueError(
                f"unknown merge_lambda_mode={self.cp.merge_lambda_mode!r}; "
                "expected 'fixed' or 'adaptive'"
            )

        def do(a: Dict, b: Dict, is_final: bool, level: int, step_index: int) -> Dict:
            nonlocal merge_calc, merge_seconds, final_curve
            efs = self.cp.efs_array if is_final else [self.cp.efs_array[0]]
            lam = lambda_for(a, b)
            merged, out = self.merge_pair(
                a["path"], b["path"], method, efs, total_n,
                merge_lambda=lam,
            )
            step_calc = int(out["merge_calc"] or 0)
            step_seconds = float(out["merge_seconds"] or 0.0)
            merge_calc += step_calc
            merge_seconds += step_seconds
            if is_final:
                final_curve = out["recall_curve"]
            merge_steps.append({
                "step": step_index,
                "level": level,
                "left_n": a["n"],
                "right_n": b["n"],
                "output_n": a["n"] + b["n"],
                "merge_calc": step_calc,
                "merge_seconds": step_seconds,
                "lambda_used": lam,
                "is_final": bool(is_final),
            })
            remove_consumed(a)
            remove_consumed(b)
            return {
                "path": merged,
                "n": a["n"] + b["n"],
                "temporary": True,
            }

        step_index = 0
        if order == "sequential":
            acc = nodes[0]
            for i, nxt in enumerate(nodes[1:], 1):
                step_index += 1
                acc = do(
                    acc, nxt,
                    is_final=(i == len(nodes) - 1),
                    level=i - 1,
                    step_index=step_index,
                )
            root = acc
        elif order == "balanced":
            current = list(nodes)
            level_no = 0
            while len(current) > 1:
                next_level = []
                last_round = len(current) <= 2
                for i in range(0, len(current), 2):
                    if i + 1 < len(current):
                        step_index += 1
                        next_level.append(do(
                            current[i], current[i + 1],
                            is_final=(last_round and i == 0),
                            level=level_no,
                            step_index=step_index,
                        ))
                    else:
                        next_level.append(current[i])
                current = next_level
                level_no += 1
            root = current[0]
        else:
            raise ValueError(f"unknown order: {order!r}")

        return {
            "root": root["path"],
            "merge_calc": merge_calc,
            "merge_seconds": merge_seconds,
            "recall_curve": final_curve,
            "merge_steps": merge_steps,
        }

    def sigm_insert(self, leaf0, lrange, rrange, total_n):
        """SIGM merge step: load leaf0 (resized to total_n) and insert base
        points [lrange, rrange) into it. Requires the experiment.cpp patch."""
        cfg = os.path.join(self.p.workdir, f"sigm_{uuid.uuid4().hex[:8]}.cfg")
        _write_kv(cfg, {
            "workload_type": self.workload, "merge_method": "INSERT",
            "dim": self.cp.dim, "max_elements": total_n, "nb": total_n,
            "M": self.cp.M, "ef_construction": self.cp.ef_construction,
            "k": self.cp.k, "kk": self.cp.kk, "nq": self.cp.nq,
            "iterations": 1, "rerun": "true", "save_index": "false",
            "base_filepath": self.p.base, "query_filepath": self.p.query,
            "groundtruth_filepath": self.p.groundtruth,
            "index_path": leaf0,               # single index, not "a,b"
            "lrange": lrange, "rrange": rrange,
            "save_path": self.p.workdir,
            "efs_array": ", ".join(str(e) for e in self.cp.efs_array),
            **self.cp.merge_kv(),
        })
        return parse_exps(self._run(self.p.exps_bin, cfg), expect_method="INSERT")


def contiguous_partitions(n: int, n_parts: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, n, n_parts + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_parts)]


def run_hnswmerger(algo: str, n_parts: int, order: str, paths: Paths,
                   params: CppParams) -> Dict:
    """Top-level: build leaves over contiguous partitions, D&C merge, return a
    record in the harness schema (build_calc + merge_calc both populated)."""
    method = ALGO_TO_METHOD[algo]
    runner = HNSWMergerRunner(paths, params)

    build_calc = 0
    build_seconds = 0.0
    leaves: List[str] = []
    leaf_sizes: List[int] = []
    for (lo, hi) in contiguous_partitions(params.nb, n_parts):
        idx, b = runner.build_leaf(lo, hi)
        leaves.append(idx)
        leaf_sizes.append(hi - lo)
        build_calc += b["build_calc"] or 0
        build_seconds += b["build_seconds"] or 0.0

    if n_parts == 1:
        # nothing to merge: the single leaf IS the index. Run a query-only ./exps
        # so the from-scratch baseline (INSERT/REBUILD) still gets a recall curve.
        dc = {"root": leaves[0], "merge_calc": 0, "merge_seconds": 0.0, "recall_curve": None}
        try:
            q = runner.query_only(leaves[0], method, total_n=params.nb)
            dc["recall_curve"] = q.get("recall_curve")
        except Exception as e:
            dc["recall_curve"] = None   # binary may not support single-index query for this method
            print(f"  (query-only recall for {algo}/n_parts=1 unavailable: {e})")
    else:
        if algo == "SIGM":
            first_hi = contiguous_partitions(params.nb, n_parts)[0][1]   # reuse the loop's boundary
            out = runner.sigm_insert(leaves[0], lrange=first_hi,
                                     rrange=params.nb, total_n=params.nb)
            dc = {"merge_calc": out["merge_calc"] or 0,
                  "merge_seconds": out["merge_seconds"] or 0.0,
                  "recall_curve": None}
        else:
            dc = runner.divide_and_conquer(
                leaves, method, order, total_n=params.nb, leaf_sizes=leaf_sizes
            )

    headline = None
    if dc["recall_curve"]:
        headline = max((c["recall"] for c in dc["recall_curve"]
                        if c["recall"] is not None), default=None)
    record = {
        "builder": "hnswmerger", "algo": algo, "n_parts": n_parts,
        "partition_method": "range", "order": order,
        "dim": params.dim, "n": params.nb,
        "m": params.M, "ef_construction": params.ef_construction,
        "build_calc": build_calc, "merge_calc": dc["merge_calc"],
        "total_calc": build_calc + dc["merge_calc"],
        "build_seconds": build_seconds, "merge_seconds": dc["merge_seconds"],
        f"recall@{params.k}": headline,
        "recall_curve": dc["recall_curve"],
        "merge_steps": dc.get("merge_steps", []),
    }
    root = dc.get("root")
    keep_merged = os.environ.get("NGMBENCH_KEEP_MERGED", "").lower() in {
        "1", "true", "yes", "on"
    }
    if (
        params.cleanup_merged
        and n_parts > 1
        and root
        and not keep_merged
        and os.path.exists(root)
    ):
        os.remove(root)
    return record