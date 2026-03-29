"""
qgc/visualization/graph_plots.py
==================================
Graph drawing utilities for colored graph visualizations.

All drawing functions accept a Matplotlib Axes object so they compose
cleanly into any figure layout.
"""

from typing import Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


# ── Color palette ─────────────────────────────────────────────────────
PALETTE = [
    "#E74C3C",  # red
    "#3498DB",  # blue
    "#2ECC71",  # green
    "#F39C12",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#E67E22",  # dark orange
    "#34495E",  # dark grey-blue
]

CONFLICT_COLOR  = "#FF2D2D"
VALID_BADGE_COL = "#27AE60"
BAD_BADGE_COL   = "#E74C3C"


def get_color(color_index: int) -> str:
    """Map a color index to a hex color string."""
    return PALETTE[color_index % len(PALETTE)]


# ── Single graph ──────────────────────────────────────────────────────

def draw_colored_graph(
    graph:    nx.Graph,
    coloring: list,
    title:    str                          = "",
    ax:       Optional[matplotlib.axes.Axes] = None,
    seed:     int                          = 42,
) -> None:
    """
    Draw a graph with nodes painted by coloring.

    Improvements over original:
      - Node size scales with graph size for readability.
      - Conflict edges drawn thick dashed red with ✗ edge label.
      - Validity badge (✓ valid / ✗ conflicts) appended to title.
      - Legend sorted and compact.

    Args:
        graph:    NetworkX graph.
        coloring: List of color indices (one per node).
        title:    Axes title string.
        ax:       Matplotlib Axes to draw on (created if None).
        seed:     Spring-layout random seed for reproducibility.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    n            = graph.number_of_nodes()
    node_size    = max(250, min(800, 2400 // max(n, 1)))
    font_size    = max(7, min(11, 22 // max(n, 1)))
    pos          = nx.spring_layout(graph, seed=seed)
    node_colors  = [get_color(coloring[nd]) for nd in graph.nodes()]

    conflict_edges = [(u, v) for u, v in graph.edges() if coloring[u] == coloring[v]]
    normal_edges   = [(u, v) for u, v in graph.edges() if (u, v) not in conflict_edges]

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                           node_size=node_size, ax=ax, linewidths=1.2,
                           edgecolors="white")
    nx.draw_networkx_labels(graph, pos, font_color="white",
                            font_weight="bold", font_size=font_size, ax=ax)

    # Normal edges
    if normal_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=normal_edges,
                               edge_color="#AAAAAA", width=1.6, ax=ax,
                               alpha=0.85)

    # Conflict edges — thick, dashed, red
    if conflict_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=conflict_edges,
                               edge_color=CONFLICT_COLOR, width=2.8,
                               style="dashed", ax=ax)
        # Label each conflict edge with ✗
        elabels = {e: "✗" for e in conflict_edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=elabels,
                                     font_color=CONFLICT_COLOR, font_size=8,
                                     ax=ax, bbox=dict(alpha=0))

    # Legend
    used_colors = sorted(set(coloring))
    patches = [
        mpatches.Patch(facecolor=get_color(c), edgecolor="white",
                       linewidth=0.6, label=f"Color {c}")
        for c in used_colors
    ]
    if conflict_edges:
        patches.append(mpatches.Patch(facecolor=CONFLICT_COLOR, edgecolor="white",
                                      linewidth=0.6, label=f"Conflict ×{len(conflict_edges)}"))
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              framealpha=0.88, frameon=True, edgecolor="#CCC")

    # Title with validity badge
    n_conf      = len(conflict_edges)
    badge       = "✓ valid" if n_conf == 0 else f"✗ {n_conf} conflict(s)"
    badge_color = VALID_BADGE_COL if n_conf == 0 else BAD_BADGE_COL
    ax.set_title(f"{title}\n{badge}", fontsize=9, fontweight="bold", pad=5,
                 color=badge_color if n_conf > 0 else "black")
    ax.axis("off")


# ── Side-by-side comparison ───────────────────────────────────────────

def draw_comparison(
    graph:    nx.Graph,
    results:  dict,
    suptitle: str                    = "",
    figsize:  Optional[tuple]        = None,
) -> matplotlib.figure.Figure:
    """
    Draw multiple colorings of the same graph side-by-side.

    Args:
        graph:    NetworkX graph.
        results:  {method_label: coloring_list}.
        suptitle: Figure super-title.
        figsize:  (width, height). Auto-computed if None.

    Returns:
        Matplotlib Figure.
    """
    from qgc.core.coloring import count_conflicts
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (label, coloring) in zip(axes, results.items()):
        nc    = len(set(coloring))
        nconf = count_conflicts(coloring, graph)
        draw_colored_graph(
            graph, coloring,
            title=f"{label}\n{nc} colors · {nconf} conflict(s)",
            ax=ax,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ── QUBO landscape heatmap ────────────────────────────────────────────

def draw_qubo_heatmap(
    Q:     np.ndarray,
    ax:    Optional[matplotlib.axes.Axes] = None,
    title: str                          = "QUBO Matrix Q",
    k:     Optional[int]               = None,
    n:     Optional[int]               = None,
) -> None:
    """
    Draw an annotated heatmap of the QUBO matrix Q.

    Improvements over original:
      - Cell values annotated (omitted for large matrices).
      - Tick labels show variable names x_{i,c} when k and n supplied.
      - Diverging colormap centred on zero.
      - Colorbar with cleaner label.

    Args:
        Q:     QUBO matrix (dim × dim).
        ax:    Matplotlib Axes.
        title: Axes title.
        k:     Number of colors (for tick labels).
        n:     Number of nodes  (for tick labels).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    dim    = Q.shape[0]
    vmax   = np.abs(Q).max() or 1.0
    im     = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar   = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Coefficient", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Annotate cells if matrix is small enough
    if dim <= 12:
        for i in range(dim):
            for j in range(dim):
                val = Q[i, j]
                if val != 0:
                    text_col = "white" if abs(val) > vmax * 0.55 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=7, color=text_col, fontweight="bold")

    # Tick labels
    if k and n and dim == n * k:
        labels = [f"x{{{i},{c}}}" for i in range(n) for c in range(k)]
        ax.set_xticks(range(dim))
        ax.set_yticks(range(dim))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xlabel("Variable index", fontsize=8)
        ax.set_ylabel("Variable index", fontsize=8)
        ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)


# ── Summary bar chart ─────────────────────────────────────────────────

def draw_summary_bars(
    cases_data: list,
    mode:       str                          = "comparative",
    ax_colors:  Optional[matplotlib.axes.Axes] = None,
    ax_conf:    Optional[matplotlib.axes.Axes] = None,
) -> None:
    """
    Two grouped bar charts summarising all results:
      Left  — number of colors used per solver per graph.
      Right — number of conflicts per solver per graph.

    Args:
        cases_data: List of solve_case() result dicts.
        mode:       "comparative" or "hybrid".
        ax_colors:  Axes for colors-used bars (created internally if None).
        ax_conf:    Axes for conflicts bars.
    """
    names      = [c["name"] for c in cases_data]
    n_graphs   = len(names)
    x          = np.arange(n_graphs)

    if mode == "comparative":
        solvers = [
            ("Greedy",  "#95A5A6", [c["greedy_result"]["n_colors"]  for c in cases_data],
                                   [c["greedy_result"]["n_conflicts"] for c in cases_data]),
            ("Quantum", "#3498DB", [c["quantum_result"]["n_colors"]  for c in cases_data],
                                   [c["quantum_result"]["n_conflicts"] for c in cases_data]),
        ]
    else:
        solvers = [
            ("Greedy",  "#95A5A6", [c["greedy_result"]["n_colors"]  for c in cases_data],
                                   [c["greedy_result"]["n_conflicts"] for c in cases_data]),
            ("Hybrid",  "#9B59B6", [c["quantum_result"]["n_colors"]  for c in cases_data],
                                   [c["quantum_result"]["n_conflicts"] for c in cases_data]),
        ]

    n_sol  = len(solvers)
    width  = 0.35 if n_sol == 2 else 0.25
    offset = np.linspace(-(n_sol - 1) * width / 2, (n_sol - 1) * width / 2, n_sol)

    for ax, value_idx, ylabel, ylim_pad in [
        (ax_colors, 2, "Colors used",    1),
        (ax_conf,   3, "Conflicts",       0.5),
    ]:
        if ax is None:
            continue
        for (label, color, colors_vals, conf_vals), off in zip(solvers, offset):
            vals = colors_vals if value_idx == 2 else conf_vals
            bars = ax.bar(x + off, vals, width, label=label,
                          color=color, edgecolor="white", linewidth=0.7,
                          alpha=0.88)
            # Value labels on top of bars
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(val), ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] + ylim_pad)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)