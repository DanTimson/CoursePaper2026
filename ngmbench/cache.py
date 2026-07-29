"""Content-addressed stage cache and a tidy results log.

Artifacts live under ``cache_dir`` keyed by stage + digest. A present file means
the stage is done; reruns skip it. Results are appended to the log named by the config's results_path.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Any, Callable, Optional


class Cache:
    def __init__(self, root: str = ".ngmbench_cache"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, stage: str, key: str, ext: str = "pkl") -> str:
        d = os.path.join(self.root, stage)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{key}.{ext}")

    def has(self, stage: str, key: str, ext: str = "pkl") -> bool:
        return os.path.exists(self.path(stage, key, ext))

    def get_or_compute(
        self,
        stage: str,
        key: str,
        compute: Callable[[], Any],
        *,
        save: Optional[Callable[[Any, str], None]] = None,
        load: Optional[Callable[[str], Any]] = None,
        ext: str = "pkl",
    ) -> Any:
        p = self.path(stage, key, ext)
        if os.path.exists(p):
            if load:
                return load(p)
            with open(p, "rb") as f:
                return pickle.load(f)
        val = compute()
        if save:
            save(val, p)
        else:
            with open(p, "wb") as f:
                pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)
        return val


class ResultsLog:
    def __init__(self, path: str = "results.jsonl"):
        self.path = path

    def append(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def load_all(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]