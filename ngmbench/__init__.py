"""ngmbench — experiment harness for navigable-graph construction by merge.

Compares divide-and-conquer HNSW merge construction (NGM / IGTM / CGTM,
from Ponomarenko arXiv:2505.16064) against sequential insertion (SIGM) and
NN-Descent, on SIFT1M / GIST1M-style data.

Phase 1 is pure-Python (the vendored reference implementation); the primary
within-merge-family metric is distance-computation count, plus wall-clock and
recall@k. C++ ports are a later phase.
"""
__version__ = "0.1.0"
