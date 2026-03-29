"""
app.py
======
Streamlit dashboard for Quantum Graph Coloring.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

from pipeline import run_pipeline
from qgc.visualization.graph_plots import (
    draw_colored_graph,
    draw_comparison,
    draw_qubo_heatmap,
    draw_summary_bars,
)
from qgc.visualization.convergence import (
    draw_convergence,
    draw_probability_distribution,
    draw_multi_convergence,
)

st.set_page_config(
    page_title="Quantum Graph Coloring",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Quantum Graph Coloring")
st.caption(
    "Hybrid quantum–classical graph coloring via QAOA and qudit-inspired optimisation. "
    "Configure the graph and solver in the sidebar, then hit **Run Optimization**."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Solver")
    mode = st.selectbox(
        "Mode",
        ["comparative", "hybrid"],
        help=(
            "**comparative** — runs Greedy, QAOA, and Qudit side-by-side.\n\n"
            "**hybrid** — runs Greedy first; escalates to QAOA only if conflicts remain."
        ),
    )

    st.subheader("Graph")
    graph_type = st.selectbox("Graph Type", ["Random", "Cycle", "Complete", "Petersen"])
    k = st.slider("Colors (k)", 2, 6, 3)
    n = st.slider("Nodes", 3, 20, 10)
    edges = st.slider("Edges (Random only)", 1, 50, 15)

    st.subheader("Display")
    show_qubo     = st.checkbox("Show QUBO heatmap",         value=False)
    show_probs    = st.checkbox("Show probability distribution (QAOA)", value=True)
    show_summary  = st.checkbox("Show summary bar charts",   value=True)

    run = st.button("▶ Run Optimization", use_container_width=True, type="primary")


# ── Graph factory (always returns nx.Graph — fixes Pylance errors) ─────────
def create_graph(graph_type: str, n: int, edges: int) -> nx.Graph:
    if graph_type == "Random":
        return nx.gnm_random_graph(n, edges, seed=42)
    elif graph_type == "Cycle":
        return nx.cycle_graph(n)
    elif graph_type == "Complete":
        return nx.complete_graph(n)
    elif graph_type == "Petersen":
        return nx.petersen_graph()
    else:
        return nx.gnm_random_graph(n, edges, seed=42)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fig_to_st(fig: matplotlib.figure.Figure) -> None:
    """Render a matplotlib Figure in Streamlit then close it."""
    st.pyplot(fig)
    plt.close(fig)


def _valid_badge(n_conflicts: int) -> str:
    return "✅ Valid" if n_conflicts == 0 else f"❌ {n_conflicts} conflict(s)"


# ── Main ──────────────────────────────────────────────────────────────────────
if not run:
    # Landing state — show a preview of the selected graph
    st.info("Configure the graph and solver in the sidebar, then press **▶ Run Optimization**.")
    G_preview: nx.Graph = create_graph(graph_type, n, edges)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    nx.draw_spring(
        G_preview, ax=ax,
        node_color="#3498DB", node_size=400,
        edge_color="#AAAAAA", font_color="white",
        with_labels=True, font_weight="bold",
    )
    ax.set_title(
        f"{graph_type} graph — {G_preview.number_of_nodes()} nodes, "
        f"{G_preview.number_of_edges()} edges",
        fontsize=10, fontweight="bold",
    )
    st.pyplot(fig)
    plt.close(fig)

else:
    G: nx.Graph = create_graph(graph_type, n, edges)

    # ── Header metrics ────────────────────────────────────────────────
    st.subheader(f"Graph: {graph_type}  ·  Mode: {mode.upper()}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Nodes", G.number_of_nodes())
    m2.metric("Edges", G.number_of_edges())
    m3.metric("Colors (k)", k)

    st.divider()

    with st.spinner("Running optimisation — this may take a few seconds…"):
        results = run_pipeline(G, k, mode=mode)

    # ── Results table ─────────────────────────────────────────────────
    st.subheader("📊 Results")
    rows = []
    for solver_name, res in results.items():
        if res is None:
            rows.append({"Solver": solver_name.upper(), "Colors Used": "—",
                          "Conflicts": "—", "Valid": "⏭ skipped"})
        else:
            rows.append({
                "Solver":      solver_name.upper(),
                "Colors Used": str(len(set(res["coloring"]))),
                "Conflicts":   str(res["conflicts"]),
                "Valid":       _valid_badge(res["conflicts"]),
            })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()

    # ── Side-by-side colored graph comparison ─────────────────────────
    st.subheader("🎨 Graph Colorings")
    valid_results = {k_: v for k_, v in results.items() if v is not None}
    comparison_colorings = {
        label.upper(): res["coloring"] for label, res in valid_results.items()
    }
    if comparison_colorings:
        fig = draw_comparison(G, comparison_colorings,
                              suptitle=f"{graph_type} — {mode} mode")
        _fig_to_st(fig)

    st.divider()

    # ── QAOA-specific panels ──────────────────────────────────────────
    qaoa_key  = "qaoa"  if mode == "comparative" else "hybrid"
    qaoa_res  = results.get(qaoa_key)

    # Normalise: qudit results don't have probs/convergence
    has_convergence = qaoa_res is not None and qaoa_res.get("convergence")
    has_probs       = qaoa_res is not None and qaoa_res.get("probs") is not None

    if has_convergence or has_probs:
        st.subheader("📉 QAOA Analysis")
        left, right = st.columns(2)

        if has_convergence:
            assert qaoa_res is not None
            with left:
                st.markdown("**Convergence curve**")
                fig, ax = plt.subplots(figsize=(6, 3))
                draw_convergence(qaoa_res["convergence"],
                                 title=f"{qaoa_key.upper()} convergence", ax=ax)
                fig.tight_layout()
                _fig_to_st(fig)

        if has_probs and show_probs:
            assert qaoa_res is not None
            with right:
                st.markdown("**Measurement probability distribution**")
                n_qubits = G.number_of_nodes() * k
                fig, ax = plt.subplots(figsize=(6, 3))
                draw_probability_distribution(
                    qaoa_res["probs"], n_qubits,
                    title="Top-20 measurement outcomes",
                    ax=ax, graph=G, k=k,
                )
                fig.tight_layout()
                _fig_to_st(fig)

        st.divider()

    # ── Multi-optimizer convergence (comparative mode) ─────────────────
    if mode == "comparative":
        traces = {}
        for solver_label, res in results.items():
            if res and res.get("convergence"):
                traces[solver_label.upper()] = res["convergence"]
        if len(traces) > 1:
            st.subheader("📈 Convergence Comparison")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            draw_multi_convergence(traces, title="Convergence — all solvers", ax=ax)
            fig.tight_layout()
            _fig_to_st(fig)
            st.divider()

    # ── QUBO heatmap ──────────────────────────────────────────────────
    if show_qubo:
        st.subheader("🔲 QUBO Matrix")
        try:
            from qgc.core.hamiltonian import build_qubo
            Q = build_qubo(G, k=k, penalty=4.0)
            fig, ax = plt.subplots(figsize=(6, 5))
            draw_qubo_heatmap(Q, ax=ax, title="QUBO matrix Q", k=k, n=G.number_of_nodes())
            fig.tight_layout()
            _fig_to_st(fig)
            st.divider()
        except Exception as e:
            st.warning(f"Could not render QUBO heatmap: {e}")

    # ── Summary bar charts ────────────────────────────────────────────
    if show_summary and len(valid_results) >= 2:
        st.subheader("📋 Summary")
        # Repackage into the format draw_summary_bars expects
        case = {
            "name": f"{graph_type}({G.number_of_nodes()},{G.number_of_edges()})",
            "greedy_result": {
                "n_colors":    len(set(results["greedy"]["coloring"])),
                "n_conflicts": results["greedy"]["conflicts"],
            },
        }
        quantum_key = "qaoa" if "qaoa" in results and results["qaoa"] else \
                      "qudit" if "qudit" in results else \
                      "hybrid" if "hybrid" in results else None
        if quantum_key and results.get(quantum_key):
            case["quantum_result"] = {
                "n_colors":    len(set(results[quantum_key]["coloring"])),
                "n_conflicts": results[quantum_key]["conflicts"],
            }
            fig, (ax_c, ax_f) = plt.subplots(1, 2, figsize=(10, 3.5))
            draw_summary_bars([case], mode=mode if mode == "hybrid" else "comparative",
                              ax_colors=ax_c, ax_conf=ax_f)
            ax_c.set_title("Colors used", fontsize=9, fontweight="bold")
            ax_f.set_title("Conflicts",   fontsize=9, fontweight="bold")
            fig.tight_layout()
            _fig_to_st(fig)
        st.divider()

    # ── Raw detail expander ───────────────────────────────────────────
    with st.expander("🔍 Raw solver output"):
        for solver_label, res in results.items():
            if res is None:
                st.write(f"**{solver_label.upper()}** — skipped")
            else:
                st.write(f"**{solver_label.upper()}**")
                st.json({
                    "coloring":  res["coloring"],
                    "conflicts": res["conflicts"],
                    "n_colors":  len(set(res["coloring"])),
                    "is_valid":  res["conflicts"] == 0,
                })