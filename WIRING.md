# Wiring the merge-parameter sweep + iso-quality metric into ngmbench

Apply after `sigm_insert_min.patch` and `merge_params_and_ds.patch`, and after
`make exp`. Verify the binary first:

```bash
strings exps | grep -c "distance calls"                    # >= 5 (4 merges + INSERT)
strings exps | grep -c "search distance calls per query"   # 1
```

---

## 1. `ngmbench/index/hnswmerger.py` — CppParams

Add the merge knobs and the thread control. Defaults match `baseline2.h`
signatures, so omitting them reproduces your existing rows.

```python
@dataclass
class CppParams:
    # ... existing fields ...
    thread: int = 1                  # drives OMP_NUM_THREADS *and* the cfg thread key
    # merge knobs (Ponomarenko). NGM uses search_ef; IGTM uses all five;
    # CGTM's signature has NO next_step_ef; SIGM uses merge_ef_construction.
    jump_ef: int = 40
    local_ef: int = 10
    next_step_k: int = 6
    next_step_ef: int = 6
    search_M: int = 40
    search_ef: int = 40
    merge_ef_construction: int = -1  # -1 = inherit from the loaded index

    def merge_kv(self) -> dict:
        """cfg keys for the merge/insert phase. Harmless for algos that ignore them."""
        return {"jump_ef": self.jump_ef, "local_ef": self.local_ef,
                "next_step_k": self.next_step_k, "next_step_ef": self.next_step_ef,
                "search_M": self.search_M, "search_ef": self.search_ef,
                "merge_ef_construction": self.merge_ef_construction,
                "thread": self.thread}

    def merge_id(self) -> dict:
        """Identity of this parameter point — goes into run_key and the row."""
        return {"jump_ef": self.jump_ef, "local_ef": self.local_ef,
                "next_step_k": self.next_step_k, "next_step_ef": self.next_step_ef,
                "search_M": self.search_M, "search_ef": self.search_ef,
                "merge_ef_construction": self.merge_ef_construction}
```

In `HNSWMergerRunner.__init__`, drive OMP from it (this is what `./builds` reads —
`build_index.cpp` hardcodes `omp_get_max_threads()`):

```python
        self.env = dict(os.environ)
        threads = env_threads or getattr(params, "thread", None)
        if threads:
            self.env["OMP_NUM_THREADS"] = str(threads)
```

Then in **every** `_write_kv(cfg, {...})` — `merge_pair`, `sigm_insert`,
`query_only`, `build_leaf` — splat the knobs in:

```python
    _write_kv(cfg, {..., **self.cp.merge_kv()})
```

## 2. `parse_exps` — pick up `d_s`

The eval loop prints, per ef in `efs_array`, three repeats of
`search distance calls per query = <float>` followed by the recall line. Group by
the `set ef =` marker and take the median repeat (they are identical at
`thread=1`, so this is just robustness):

```python
DS_RE  = re.compile(r"search distance calls per query = ([\d.]+)")
EF_RE  = re.compile(r"set ef = (\d+)")
REC_RE = re.compile(r"R@\d+ = ([\d.]+)")

def parse_recall_curve(stdout: str) -> list[dict]:
    curve, ef, ds_buf, rec_buf = [], None, [], []
    def flush():
        if ef is not None and ds_buf:
            curve.append({"ef": ef,
                          "d_s": sorted(ds_buf)[len(ds_buf)//2],
                          "recall": sorted(rec_buf)[len(rec_buf)//2] if rec_buf else None})
    for line in stdout.splitlines():
        m = EF_RE.search(line)
        if m:
            flush(); ef, ds_buf, rec_buf = int(m.group(1)), [], []
            continue
        m = DS_RE.search(line)
        if m: ds_buf.append(float(m.group(1))); continue
        m = REC_RE.search(line)
        if m: rec_buf.append(float(m.group(1)))
    flush()
    return curve
```

Merge that into whatever `parse_exps` already returns, so each `recall_curve`
entry gains a `d_s` field alongside `recall` and `query_seconds`.

## 3. `run_key` and the row — or the sweep silently collapses

`run_key` currently keys on `(dataset, algo, n_parts, order)`. Sweeping params
produces several rows per that tuple; the skip-cache will treat variant #2 as
done, and `make_figures.load()`'s last-wins dedup will discard the rest.

```python
import hashlib, json
def run_key(dataset, algo, n_parts, order, params) -> str:
    pid = hashlib.md5(json.dumps(params.merge_id(), sort_keys=True).encode()).hexdigest()[:8]
    return f"{dataset}|{algo}|{n_parts}|{order}|{pid}"
```

And in the returned record add `"params": params.merge_id()` plus
`"threads": params.thread`. **Do this before running anything.**

## 4. Sweep JSON

Per-entry `params` lists; `cli_cpp` takes the cartesian product with
`algo` x `n_parts` and builds a `CppParams` per point.

```json
{
  "dataset": {"name": "sift1m", "dim": 128, "nb": 1000000,
              "base": "/data/sift_scales/sift1m_base.fvecs",
              "query": "/data/sift_scales/sift_query.fvecs",
              "groundtruth": "/data/sift_scales/sift1m_gt.ivecs"},
  "hnsw": {"M": 16, "ef_construction": 200},
  "eval": {"k": 10, "kk": 100, "nq": 10000, "efs_array": [10, 50, 100, 200, 400]},
  "threads": 1,
  "workdir": ".work_sift1m",
  "results_path": "results_sift1m.jsonl",
  "sweep": [
    {"algo": ["IGTM"], "n_parts": [2], "order": ["balanced"], "params": [
      {"jump_ef": 40, "local_ef": 10, "next_step_k": 6, "next_step_ef": 6, "search_M": 40},
      {"jump_ef": 20, "local_ef": 5,  "next_step_k": 3, "next_step_ef": 3, "search_M": 3},
      {"jump_ef": 64, "local_ef": 5,  "next_step_k": 3, "next_step_ef": 3, "search_M": 5},
      {"jump_ef": 64, "local_ef": 10, "next_step_k": 3, "next_step_ef": 10, "search_M": 5},
      {"jump_ef": 5,  "local_ef": 7,  "next_step_k": 3, "next_step_ef": 3, "search_M": 5}
    ]},
    {"algo": ["CGTM"], "n_parts": [2], "order": ["balanced"], "params": [
      {"jump_ef": 40, "local_ef": 10, "next_step_k": 6, "search_M": 40},
      {"jump_ef": 20, "local_ef": 5,  "next_step_k": 3, "search_M": 3},
      {"jump_ef": 15, "local_ef": 5,  "next_step_k": 3, "search_M": 5}
    ]},
    {"algo": ["NGM"], "n_parts": [2], "order": ["balanced"], "params": [
      {"search_ef": 40}, {"search_ef": 20}, {"search_ef": 10}
    ]},
    {"algo": ["SIGM"], "n_parts": [2], "order": ["balanced"], "params": [
      {"merge_ef_construction": -1}, {"merge_ef_construction": 48},
      {"merge_ef_construction": 24}, {"merge_ef_construction": 16}
    ]},
    {"algo": ["INSERT"], "n_parts": [1], "order": ["balanced"], "params": [{}]}
  ]
}
```

The IGTM grid above is a thinned version of `IGTM_bench.py`'s 28-point sweep
plus the `j5,l7` point from the paper table. **`10m` should reuse this file with
the `params` lists cut to a single entry each** (the 1M winner) — plus, cheaply,
the header defaults, to bound how far the optimum drifts with N.

## 5. Figures

`fig_merge_cost` / `fig_total_cost` assume one row per `(algo, n_parts)`; with a
sweep they need a param dimension. Two additions worth making:

- **Iso-quality scatter (the important one).** x = `d_s` at a fixed target recall,
  y = `merge_calc`, one point per (algo, param-point), colour by algo. This is the
  comparison the paper's table makes — merge cost read off at matched search cost.
  Without it a cheap config "wins" merge cost by producing a worse graph.
- **Pareto front per algo**: `merge_calc` vs best-recall, so the sweep collapses to
  a frontier rather than a cloud.

For the existing charts, select one canonical param point per algo (the
iso-quality winner) and note the choice in the caption.
