"""ngmbench — experiment harness for navigable-graph construction by merge.

Compares divide-and-conquer HNSW construction by graph merge (NGM / IGTM / CGTM
from Ponomarenko, arXiv:2505.16064; and the SIGMOD'26 HNSW-Merger of Jin et al.)
against sequential insertion (SIGM) and full rebuild, across BIGANN SIFT slices
from 10k to 10M. The primary within-family metric is distance-computation count,
which is language- and threading-independent; wall-clock QPS and recall@k are
reported alongside for search quality.

Experiments are driven through the C++ backend: `python -m ngmbench.cli_cpp`
shells out to a patched HNSWMerger build and records metrics to a JSONL log.
Figures and trend analyses live in `scripts/`.
"""
__version__ = "1.0.0"