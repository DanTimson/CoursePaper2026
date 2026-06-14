"""Streamlit playground for browsing merge-vs-NN-Descent results.

    streamlit run app/playground.py -- --results results.jsonl

Reads the cached results log (no recomputation) and renders Pareto views and a
results table. Distance-count axes apply to the merge family (and SIGM); the
build-time axis applies to everything including NN-Descent, which has null
distance counts by design.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st


def _results_path() -> str:
    # support `streamlit run app/playground.py -- --results foo.jsonl`
    if "--results" in sys.argv:
        return sys.argv[sys.argv.index("--results") + 1]
    return os.environ.get("NGMBENCH_RESULTS", "results.jsonl")


@st.cache_data
def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    # de-duplicate on run_key, keeping the last write
    if "run_key" in df.columns:
        df = df.drop_duplicates(subset="run_key", keep="last")
    return df


def main():
    st.set_page_config(page_title="NGM merge bench", layout="wide")
    st.title("Navigable-graph merge construction — results")

    path = _results_path()
    df = load(path)
    if df.empty:
        st.warning(f"No results at `{path}`. Run a sweep first, e.g. "
                   "`python -m ngmbench.cli --config config/synthetic_demo.json`.")
        return

    k_cols = [c for c in df.columns if c.startswith("recall@")]
    kcol = k_cols[0] if k_cols else None

    with st.sidebar:
        st.header("Filters")
        datasets = sorted(df["dataset"].unique())
        dsel = st.multiselect("Dataset", datasets, default=datasets)
        algos = sorted(df["algo"].dropna().unique()) if "algo" in df else []
        asel = st.multiselect("Algorithm", algos, default=algos)
        st.caption(f"Source: `{path}` · {len(df)} runs")

    f = df[df["dataset"].isin(dsel)]
    if asel:
        f = f[f["algo"].isin(asel) | f["algo"].isna()]

    # ---- Pareto: recall vs construction distance computations --------------- #
    st.subheader("Construction cost (distance computations) vs recall")
    st.caption("Merge family + SIGM. Lower-right is better. "
               "NN-Descent is omitted here (no comparable distance count).")
    mf = f[f["total_calc"].notna()].copy()
    if not mf.empty and kcol:
        mf["config"] = mf["algo"].astype(str) + " · " + mf.get("order", "-").astype(str) \
            + " · parts=" + mf["n_parts"].astype(str) + " · " + mf.get("partition_method", "-").astype(str)
        chart = (
            alt.Chart(mf)
            .mark_circle(size=130, opacity=0.85)
            .encode(
                x=alt.X("total_calc:Q", title="construction distance comps",
                        scale=alt.Scale(zero=False)),
                y=alt.Y(f"{kcol}:Q", title=kcol, scale=alt.Scale(zero=False)),
                color=alt.Color("algo:N", title="algorithm"),
                shape=alt.Shape("order:N", title="merge order"),
                tooltip=["config", kcol, "build_calc", "merge_calc", "total_calc",
                         "build_seconds", "merge_seconds", "frac_reachable_L0"],
            )
            .interactive()
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)

    # ---- Pareto: recall vs build wall-clock (all builders) ------------------ #
    st.subheader("Construction wall-clock vs recall (all builders)")
    st.caption("Includes NN-Descent. Merge wall-clock = leaf build + merge.")
    wf = f.copy()
    if "merge_seconds" in wf:
        wf["wall_seconds"] = wf["build_seconds"].fillna(0) + wf["merge_seconds"].fillna(0)
    else:
        wf["wall_seconds"] = wf["build_seconds"]
    if not wf.empty and kcol:
        wf["label"] = wf["algo"].astype(str) + " parts=" + wf["n_parts"].astype(str)
        chart2 = (
            alt.Chart(wf)
            .mark_circle(size=130, opacity=0.85)
            .encode(
                x=alt.X("wall_seconds:Q", title="construction seconds", scale=alt.Scale(zero=False)),
                y=alt.Y(f"{kcol}:Q", title=kcol, scale=alt.Scale(zero=False)),
                color=alt.Color("algo:N", title="builder/algorithm"),
                tooltip=["label", kcol, "wall_seconds", "build_seconds", "merge_seconds"],
            )
            .interactive()
            .properties(height=420)
        )
        st.altair_chart(chart2, use_container_width=True)

    # ---- Graph quality ------------------------------------------------------ #
    st.subheader("Merged-graph quality")
    qcols = [c for c in ["deg_mean", "deg_p95", "deg_max", "n_layers",
                         "frac_reachable_L0"] if c in f.columns]
    if qcols:
        st.dataframe(
            f[["dataset", "algo", "order", "n_parts", "partition_method", *qcols]]
            .sort_values(["dataset", "algo"]).reset_index(drop=True),
            use_container_width=True,
        )

    # ---- Raw table ---------------------------------------------------------- #
    with st.expander("All runs (raw)"):
        st.dataframe(f.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
