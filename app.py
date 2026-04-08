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
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
import itertools

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

# ── Sidebar ────────────────────────────────────────────────────────────────────
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

    st.subheader("QAOA Demo")
    force_qaoa = st.checkbox("⚡ Force QAOA (demo mode)", value=False)
    MAX_QAOA_QUBITS = 12
    total_qubits = n * k
    if force_qaoa:
        if total_qubits <= MAX_QAOA_QUBITS:
            st.success("QAOA forced (demo mode)")
        else:
            st.warning(f"QAOA not executed due to exponential state size (n×k={total_qubits} > {MAX_QAOA_QUBITS})")
    
    st.subheader("Display")
    show_qubo    = st.checkbox("Show QUBO heatmap",                   value=False)
    show_probs   = st.checkbox("Show probability distribution (QAOA)", value=True)
    show_summary = st.checkbox("Show summary bar charts",              value=True)

    run = st.button("▶ Run Optimization", use_container_width=True, type="primary")


# ── Graph factory ──────────────────────────────────────────────────────────────
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fig_to_st(fig: matplotlib.figure.Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


def _valid_badge(n_conflicts: int) -> str:
    return "✅ Valid" if n_conflicts == 0 else f"❌ {n_conflicts} conflict(s)"


def _winner_of(results: dict, metric: str, lower_is_better: bool = True) -> str:
    """Return the solver name that wins on a given metric (ignoring None values)."""
    best_solver = None
    best_val    = None
    for name, res in results.items():
        if res is None:
            continue
        val = res["metrics"].get(metric)
        if val is None:
            continue
        if best_val is None:
            best_val, best_solver = val, name
        elif lower_is_better and val < best_val:
            best_val, best_solver = val, name
        elif not lower_is_better and val > best_val:
            best_val, best_solver = val, name
    return best_solver or "—"


# ── Radar / spider chart ───────────────────────────────────────────────────────
def draw_radar(results: dict, ax: Axes) -> None:
    """
    Spider chart comparing all solvers across 5 normalised metrics.
    All axes run 0 (worst) → 1 (best).
    """
    metrics_cfg = [
        ("valid",             False, "Valid"),
        ("color_efficiency",  False, "Color\nEfficiency"),
        ("conflict_rate",     True,  "Conflict\nRate"),
        ("elapsed_ms",        True,  "Speed"),
        ("score",             False, "Score"),
    ]
    labels = [m[2] for m in metrics_cfg]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    COLORS = {
        "greedy": "#95A5A6",
        "qaoa":   "#E74C3C",
        "qudit":  "#3498DB",
        "hybrid": "#9B59B6",
    }

    ax.set_theta_offset(np.pi / 2)    # type: ignore[attr-defined]
    ax.set_theta_direction(-1)         # type: ignore[attr-defined]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6, color="#888")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["polar"].set_alpha(0.2)

    # Collect raw values for normalisation
    raw: dict[str, list] = {}
    for solver_name, res in results.items():
        if res is None:
            continue
        row = []
        for key, lower_better, _ in metrics_cfg:
            v = res["metrics"].get(key)
            if v is None:
                row.append(0.5)
            elif isinstance(v, bool):
                row.append(1.0 if v else 0.0)
            else:
                row.append(float(v))
        raw[solver_name] = row

    # Normalise per metric across solvers
    all_vals = list(raw.values())
    norm: dict[str, list] = {}
    for idx, (key, lower_better, _) in enumerate(metrics_cfg):
        col = [r[idx] for r in all_vals if r]
        mn, mx = min(col), max(col)
        rng = mx - mn if mx != mn else 1.0
        for solver_name in raw:
            v_norm = (raw[solver_name][idx] - mn) / rng
            if lower_better:
                v_norm = 1.0 - v_norm
            norm.setdefault(solver_name, []).append(round(v_norm, 3))

    # Plot each solver
    for solver_name, vals in norm.items():
        vals_closed = vals + vals[:1]
        color = COLORS.get(solver_name, "#555")
        ax.plot(angles, vals_closed, color=color, linewidth=2.2, alpha=0.9,
                label=solver_name.upper())
        ax.fill(angles, vals_closed, color=color, alpha=0.10)
        # Mark each vertex
        ax.scatter(angles[:-1], vals, color=color, s=30, zorder=5,
                   edgecolors="white", linewidths=0.8)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              fontsize=8, framealpha=0.85)
    ax.set_title("Algorithm Comparison — Radar", fontsize=10,
                 fontweight="bold", pad=18)


# ── Per-metric winner badge ────────────────────────────────────────────────────
def _render_winner_badges(results: dict) -> None:
    """Show a horizontal row of metric winner pills."""
    metrics = [
        ("score",          False, "🏆 Best Score"),
        ("elapsed_ms",     True,  "⚡ Fastest"),
        ("colors_used",    True,  "🎨 Fewest Colors"),
        ("conflict_rate",  True,  "✅ Fewest Conflicts"),
        ("color_efficiency", False, "🔬 Most Efficient"),
    ]
    cols = st.columns(len(metrics))
    COLORS = {
        "greedy": "#95A5A6",
        "qaoa":   "#E74C3C",
        "qudit":  "#3498DB",
        "hybrid": "#9B59B6",
    }
    for col, (metric, lower_better, label) in zip(cols, metrics):
        winner = _winner_of(results, metric, lower_is_better=lower_better)
        color  = COLORS.get(winner, "#555")
        val    = results.get(winner, {})
        if val:
            raw_val = val.get("metrics", {}).get(metric)
            if metric == "elapsed_ms":
                val_str = f"{raw_val:.1f} ms" if raw_val is not None else "—"
            elif metric == "conflict_rate":
                val_str = f"{raw_val*100:.1f}%" if raw_val is not None else "—"
            elif metric == "color_efficiency":
                val_str = f"{raw_val:.3f}" if raw_val is not None else "—"
            else:
                val_str = str(raw_val) if raw_val is not None else "—"
        else:
            val_str = "—"
        col.markdown(
            f"""<div style="border:1.5px solid {color};border-radius:8px;
                padding:10px 8px;text-align:center;margin:2px">
                <div style="font-size:11px;color:#888;margin-bottom:3px">{label}</div>
                <div style="font-weight:700;color:{color};font-size:15px">{winner.upper()}</div>
                <div style="font-size:11px;color:#888;margin-top:2px">{val_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Main ───────────────────────────────────────────────────────────────────────
if not run:
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

    # ── Header metrics ─────────────────────────────────────────────────
    st.subheader(f"Graph: {graph_type}  ·  Mode: {mode.upper()}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Nodes", G.number_of_nodes())
    m2.metric("Edges", G.number_of_edges())
    m3.metric("Colors (k)", k)

    st.divider()

    with st.spinner("Running optimisation — this may take a few seconds…"):
        results = run_pipeline(G, k, mode=mode, force_qaoa=force_qaoa)

    # ── Basic results table ────────────────────────────────────────────
    st.subheader("📊 Results")
    rows = []
    for solver_name, res in results.items():
        if res is None:
            rows.append({"Solver": solver_name.upper(), "Colors Used": "—",
                          "Conflicts": "—", "Valid": "⏭ skipped"})
        else:
            if res.get("status") == "skipped":
                rows.append({
                    "Solver":      solver_name.upper(),
                    "Colors Used": "—",
                    "Conflicts":   "—",
                    "Valid":       "⛔ skipped (too large)",
                })
            else:
                rows.append({
                    "Solver":      solver_name.upper(),
                    "Colors Used": str(len(set(res["coloring"]))),
                    "Conflicts":   str(res["conflicts"]),
                    "Valid":       _valid_badge(res["conflicts"]),
                })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── QAOA demo mode feedback ────────────────────────────────────────
    if force_qaoa:
        qaoa_result = results.get("qaoa") or results.get("hybrid")
        if qaoa_result and qaoa_result.get("demo_status") == "executed":
            st.success("✅ QAOA executed successfully")
        elif qaoa_result and qaoa_result.get("demo_status") == "skipped_too_large":
            st.error("⛔ QAOA skipped: problem too large for safe simulation (n×k > 12)")
        else:
            st.info("ℹ️ QAOA was not reached in this run.")
            
    # ══════════════════════════════════════════════════════════════════
    # ── COMPARISON METRICS ────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════
    st.subheader("📐 Algorithm Comparison")

    # ── Winner badges row ──────────────────────────────────────────────
    valid_results = {k_: v for k_, v in results.items() if v is not None}
    _render_winner_badges(valid_results)

    st.markdown("")   # spacing

    # ── Detailed metrics table ─────────────────────────────────────────
    metric_rows = []
    for solver_name, res in results.items():
        if res is None:
            continue
        m = res["metrics"]
        metric_rows.append({
            "Solver":           solver_name.upper(),
            "Colors Used":      m["colors_used"],
            "Conflicts":        m["conflicts"],
            "Conflict Rate":    f"{m['conflict_rate']*100:.1f}%",
            "Color Efficiency": f"{m['color_efficiency']:.3f}",
            "χ Lower Bound":    m["k_lower_bound"],
            "Time (ms)":        f"{m['elapsed_ms']:.1f}",
            "Final Energy":     f"{m['energy']:.3f}" if m["energy"] is not None else "—",
            "Energy Δ":         f"{m['energy_improvement']:.3f}" if m["energy_improvement"] is not None else "—",
            "# Evaluations":    str(m["n_evaluations"]) if m["n_evaluations"] is not None else "—",
            "Score /100":       m["score"],
            "Valid":            "✅" if m["valid"] else "❌",
        })

    metric_df = pd.DataFrame(metric_rows)
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Metric definitions"):
        st.markdown("""
| Metric | Meaning |
|---|---|
| **Colors Used** | Number of distinct colors assigned across all nodes |
| **Conflicts** | Edges where both endpoints share the same color (0 = valid proper coloring) |
| **Conflict Rate** | Conflicts ÷ total edges — normalized conflict density |
| **Color Efficiency** | Clique number (χ lower bound) ÷ colors used. 1.0 = optimal, lower = wasteful |
| **χ Lower Bound** | Maximum clique size — a proven lower bound on the chromatic number |
| **Time (ms)** | Wall-clock milliseconds for the solver to return |
| **Final Energy** | Minimum Ising Hamiltonian energy found (QAOA only) |
| **Energy Δ** | Energy improvement from first to last optimizer evaluation (QAOA only) |
| **# Evaluations** | Number of cost-function calls made by the classical optimizer (QAOA only) |
| **Score /100** | Composite: starts at 100, penalizes conflicts (×20 each) and extra colors (×5 each), small speed bonus |
""")

    st.markdown("")

    # ── Radar chart + bar charts side by side ─────────────────────────
    col_radar, col_bar = st.columns([1, 1])

    with col_radar:
        st.markdown("**Performance radar**")
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
        draw_radar(valid_results, ax)
        fig.tight_layout()
        _fig_to_st(fig)

    with col_bar:
        st.markdown("**Score comparison**")
        score_data = {
            s: r["metrics"]["score"]
            for s, r in valid_results.items()
        }
        COLORS = {
            "greedy": "#95A5A6", "qaoa": "#E74C3C",
            "qudit": "#3498DB",  "hybrid": "#9B59B6",
        }
        fig, ax = plt.subplots(figsize=(5, 3.5))
        names  = list(score_data.keys())
        scores = list(score_data.values())
        bars   = ax.barh(names, scores,
                         color=[COLORS.get(n, "#888") for n in names],
                         edgecolor="white", linewidth=0.8, height=0.55)
        for bar, val in zip(bars, scores):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 115)
        ax.set_xlabel("Score /100", fontsize=9)
        ax.set_yticklabels([n.upper() for n in names], fontsize=9)
        ax.axvline(100, color="#27AE60", linestyle="--", linewidth=1.2,
                   alpha=0.6, label="Perfect = 100")
        ax.legend(fontsize=8, framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.2, linestyle=":")
        fig.tight_layout()
        _fig_to_st(fig)

    # ── Time breakdown ─────────────────────────────────────────────────
    st.markdown("**Runtime breakdown (ms)**")
    time_data = {
        s: r["metrics"]["elapsed_ms"]
        for s, r in valid_results.items()
    }
    fig, ax = plt.subplots(figsize=(9, 2))
    names_t = list(time_data.keys())
    times   = list(time_data.values())
    total   = sum(times) or 1
    left    = 0.0
    for nm, t in zip(names_t, times):
        pct  = t / total
        color = COLORS.get(nm, "#888")
        ax.barh(0, pct, left=left, color=color, edgecolor="white",
                linewidth=0.8, height=0.55)
        if pct > 0.06:
            ax.text(left + pct / 2, 0, f"{nm.upper()}\n{t:.0f}ms",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")
        left += pct
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_xlabel("Share of total runtime", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.2, linestyle=":")
    patches = [mpatches.Patch(color=COLORS.get(n, "#888"), label=n.upper())
               for n in names_t]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              bbox_to_anchor=(1, 1.8), framealpha=0.85)
    fig.tight_layout()
    _fig_to_st(fig)

    st.divider()

    # ── Runtime comparison bar chart ──────────────────────────────────
    st.subheader("⏱️ Runtime Comparison")
    runtime_rows = [
        {"Solver": s.upper(), "Time (ms)": r["metrics"]["elapsed_ms"]}
        for s, r in valid_results.items()
    ]
    if runtime_rows:
        rt_df = pd.DataFrame(runtime_rows).sort_values("Time (ms)")
        fig, ax = plt.subplots(figsize=(7, 2.5))
        rt_colors = [COLORS.get(r["Solver"].lower(), "#888") for _, r in rt_df.iterrows()]
        bars = ax.bar(rt_df["Solver"], rt_df["Time (ms)"],
                      color=rt_colors, edgecolor="white", linewidth=0.8, width=0.5)
        for bar, val in zip(bars, rt_df["Time (ms)"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(rt_df["Time (ms)"]) * 0.02,
                    f"{val:.1f} ms", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel("Time (ms)", fontsize=9)
        ax.set_title("Solver Runtime Comparison", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.2, linestyle=":")
        fig.tight_layout()
        _fig_to_st(fig)

    st.divider()
    # ── Graph colorings ────────────────────────────────────────────────
    st.subheader("🎨 Graph Colorings")
    comparison_colorings = {
        label.upper(): res["coloring"] for label, res in valid_results.items()
    }
    if comparison_colorings:
        fig = draw_comparison(G, comparison_colorings,
                              suptitle=f"{graph_type} — {mode} mode")
        _fig_to_st(fig)

    st.divider()

    # ── QAOA-specific panels ───────────────────────────────────────────
    qaoa_key = "qaoa" if mode == "comparative" else "hybrid"
    qaoa_res = results.get(qaoa_key)

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

    # ── Multi-optimizer convergence ────────────────────────────────────
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

    # ── QUBO heatmap ───────────────────────────────────────────────────
    #yehaw
    '''if show_qubo:
        st.subheader("🔲 QUBO Matrix")
        try:
            from qgc.core.hamiltonian import build_qubo
            Q = build_qubo(G, k=k, penalty=4.0)
            fig, ax = plt.subplots(figsize=(6, 5))
            draw_qubo_heatmap(Q, ax=ax, title="QUBO matrix Q",
                              k=k, n=G.number_of_nodes())
            fig.tight_layout()
            _fig_to_st(fig)
            st.divider()
        except Exception as e:
            st.warning(f"Could not render QUBO heatmap: {e}")'''
    # ── QUBO heatmap + validation ──────────────────────────────────────
    if show_qubo:
        st.subheader("🔲 QUBO Matrix")
        try:
            from qgc.core.hamiltonian import build_qubo
            Q = build_qubo(G, k=k, penalty=4.0)
            fig, ax = plt.subplots(figsize=(6, 5))
            draw_qubo_heatmap(Q, ax=ax, title="QUBO matrix Q",
                              k=k, n=G.number_of_nodes())
            fig.tight_layout()
            _fig_to_st(fig)
        except Exception as e:
            st.warning(f"Could not render QUBO heatmap: {e}")

        # ── QUBO Validation table (small graph only) ───────────────────
        st.markdown("**QUBO Bitstring Validation** *(2 nodes, k=2 fixed)*")
        try:
            from qgc.core.hamiltonian import build_qubo
            G_val = nx.Graph()
            G_val.add_edge(0, 1)
            k_val = 2
            Q_val = build_qubo(G_val, k=k_val, penalty=4.0)
            n_bits = G_val.number_of_nodes() * k_val  # = 4

            rows_val = []
            for bits in itertools.product([0, 1], repeat=n_bits):
                x = np.array(bits, dtype=float)
                energy = float(x @ Q_val @ x)
                # Valid = each node has exactly one color assigned
                node_assignments = [
                    bits[i * k_val : (i + 1) * k_val]
                    for i in range(G_val.number_of_nodes())
                ]
                valid_assign = all(sum(a) == 1 for a in node_assignments)
                # Valid coloring = valid assignment AND no conflict
                coloring = [a.index(1) if sum(a) == 1 else -1
                            for a in node_assignments]
                conflict = (
                    valid_assign and
                    len(coloring) == 2 and
                    coloring[0] == coloring[1]
                )
                status = (
                    "✅ Valid coloring" if valid_assign and not conflict else
                    "❌ Color conflict" if valid_assign and conflict     else
                    "⚠️ Invalid assignment"
                )
                rows_val.append({
                    "Bitstring": "".join(map(str, bits)),
                    "x₀c₀ x₀c₁ x₁c₀ x₁c₁": " ".join(map(str, bits)),
                    "Energy": round(energy, 3),
                    "Status": status,
                })

            df_val = pd.DataFrame(rows_val).sort_values("Energy").reset_index(drop=True)
            st.dataframe(
                df_val.style.apply(
                    lambda col: [
                        "background-color: #1a3a1a" if "Valid coloring" in v else
                        "background-color: #3a1a1a" if "conflict" in v else
                        "background-color: #2a2a1a"
                        for v in col
                    ] if col.name == "Status" else [""] * len(col),
                    axis=0,
                ),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("ℹ️ How to read this table"):
                st.markdown("""
- **Bitstring**: 4 bits for 2 nodes × 2 colors. Bits are `x₀c₀ x₀c₁ x₁c₀ x₁c₁`
- **Energy**: QUBO objective value `xᵀQx`. Lower = better.
- **✅ Valid coloring**: each node has exactly one color, and adjacent nodes differ
- **❌ Color conflict**: valid assignment but adjacent nodes share a color (penalized)
- **⚠️ Invalid assignment**: a node has 0 or 2+ colors active (heavily penalized)
""")
        except Exception as e:
            st.warning(f"Could not render QUBO validation: {e}")

        st.divider()

    # ── Summary bar charts ─────────────────────────────────────────────
    if show_summary and len(valid_results) >= 2:
        st.subheader("📋 Summary")
        case = {
            "name": f"{graph_type}({G.number_of_nodes()},{G.number_of_edges()})",
            "greedy_result": {
                "n_colors":    len(set(results["greedy"]["coloring"])),
                "n_conflicts": results["greedy"]["conflicts"],
            },
        }
        quantum_key = (
            "qaoa"   if "qaoa"   in results and results["qaoa"]   else
            "qudit"  if "qudit"  in results                        else
            "hybrid" if "hybrid" in results                        else None
        )
        if quantum_key and results.get(quantum_key):
            case["quantum_result"] = {
                "n_colors":    len(set(results[quantum_key]["coloring"])),
                "n_conflicts": results[quantum_key]["conflicts"],
            }
            fig, (ax_c, ax_f) = plt.subplots(1, 2, figsize=(10, 3.5))
            draw_summary_bars(
                [case],
                mode=mode if mode == "hybrid" else "comparative",
                ax_colors=ax_c, ax_conf=ax_f,
            )
            ax_c.set_title("Colors used", fontsize=9, fontweight="bold")
            ax_f.set_title("Conflicts",   fontsize=9, fontweight="bold")
            fig.tight_layout()
            _fig_to_st(fig)
        st.divider()

    # ── Scalability Analysis ───────────────────────────────────────────
    st.subheader("📏 Scalability Analysis")
    st.caption("Runtime vs. graph size across solvers. Uses random graphs with ~2× edges as nodes.")

    with st.spinner("Running scalability experiments…"):
        scale_sizes  = [5, 10, 15, 20]
        scale_data   = {s: [] for s in (
            ["greedy", "qaoa", "qudit"] if mode == "comparative"
            else ["greedy", "hybrid"]
        )}

        for sz in scale_sizes:
            n_edges_sc = min(sz * 2, sz * (sz - 1) // 2)
            G_sc = nx.gnm_random_graph(sz, n_edges_sc, seed=42)
            try:
                sc_res = run_pipeline(G_sc, k, name=f"scale-{sz}", mode=mode)
                for solver_name in scale_data:
                    r = sc_res.get(solver_name)
                    if r is not None:
                        scale_data[solver_name].append(
                            (sz, r["metrics"]["elapsed_ms"])
                        )
                    else:
                        scale_data[solver_name].append((sz, None))
            except Exception:
                for solver_name in scale_data:
                    scale_data[solver_name].append((sz, None))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    COLORS_SC = {
        "greedy": "#95A5A6", "qaoa": "#E74C3C",
        "qudit": "#3498DB",  "hybrid": "#9B59B6",
    }
    markers = {"greedy": "o", "qaoa": "s", "qudit": "^", "hybrid": "D"}
    for solver_name, points in scale_data.items():
        xs = [p[0] for p in points if p[1] is not None]
        ys = [p[1] for p in points if p[1] is not None]
        if xs:
            ax.plot(xs, ys,
                    color=COLORS_SC.get(solver_name, "#888"),
                    marker=markers.get(solver_name, "o"),
                    linewidth=2, markersize=6,
                    label=solver_name.upper())
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}ms", (x, y),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=COLORS_SC.get(solver_name, "#888"))

    ax.set_xlabel("Number of Nodes", fontsize=10)
    ax.set_ylabel("Runtime (ms)", fontsize=10)
    ax.set_title(f"Scalability — Runtime vs Graph Size  (k={k}, ~2n edges)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(scale_sizes)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle=":")
    fig.tight_layout()
    _fig_to_st(fig)

    # Scalability data table
    sc_rows = []
    for sz_idx, sz in enumerate(scale_sizes):
        row: dict[str, object] = {"Nodes": sz, "Edges (~2n)": min(sz * 2, sz * (sz - 1) // 2)}
        for solver_name, points in scale_data.items():
            val = points[sz_idx][1]
            row[f"{solver_name.upper()} (ms)"] = (
                f"{val:.1f}" if val is not None else "skipped"
            )
        sc_rows.append(row)
    st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)
    st.divider()
    
    # ── Raw detail expander ────────────────────────────────────────────
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
                    "metrics":   res["metrics"],
                })