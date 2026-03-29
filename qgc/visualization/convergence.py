"""
qgc/visualization/convergence.py
==================================
Plots for QAOA convergence curves and measurement probability distributions.

Mirrors the convergence figures in Oh et al. 2019 (Figs. 6, 9, 12).
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d


# ── Convergence curve ─────────────────────────────────────────────────

def draw_convergence(
    energy_trace: list,
    title:        str                          = "QAOA Convergence",
    ax:           Optional[matplotlib.axes.Axes] = None,
    color:        str                          = "#E74C3C",
    reference:    Optional[float]             = None,
) -> None:
    """
    Draw the QAOA cost-function convergence curve.

    Improvements over original:
      - Smoothed trend line overlaid on raw trace.
      - Minimum energy annotated with a marker and label.
      - Cleaner fill_between shading.
      - Axis spines cleaned up.

    Args:
        energy_trace: List of energy values recorded during optimisation.
        title:        Axes title.
        ax:           Matplotlib Axes (created if None).
        color:        Line color hex string.
        reference:    Optional horizontal dashed reference line.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    if not energy_trace:
        ax.text(0.5, 0.5, "No convergence data\n(qudit-inspired method)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888")
        ax.axis("off")
        ax.set_title(title, fontsize=9, fontweight="bold")
        return

    xs = np.arange(len(energy_trace))
    ys = np.array(energy_trace)

    # Raw trace (light)
    ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.35)

    # Smoothed trend (prominent)
    if len(ys) >= 7:
        smooth = uniform_filter1d(ys, size=max(3, len(ys) // 15))
        ax.plot(xs, smooth, color=color, linewidth=2.2, alpha=0.95,
                label="Smoothed")

    # Fill under smoothed curve
    base_ys = smooth if len(ys) >= 7 else ys
    ax.fill_between(xs, base_ys, base_ys.min(),
                    alpha=0.10, color=color)

    # Annotate global minimum
    min_idx = int(np.argmin(ys))
    min_val = ys[min_idx]
    ax.scatter([min_idx], [min_val], color=color, s=55, zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.annotate(f"min={min_val:.2f}",
                xy=(min_idx, min_val),
                xytext=(min(min_idx + len(xs) * 0.05, len(xs) * 0.85), min_val + (ys.max() - min_val) * 0.15),
                fontsize=7, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    # Optional reference line
    if reference is not None:
        ax.axhline(reference, linestyle="--", color="#2C3E50",
                   linewidth=1.2, label=f"Reference: {reference:.2f}")
        ax.legend(fontsize=7, framealpha=0.85)

    ax.set_xlabel("Function evaluations", fontsize=8)
    ax.set_ylabel("Ising energy ⟨H⟩", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.grid(True, alpha=0.20, linestyle=":")
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


# ── Probability distribution ──────────────────────────────────────────

def draw_probability_distribution(
    probs:    np.ndarray,
    n_qubits: int,
    top_n:    int                          = 20,
    title:    str                          = "Measurement outcomes",
    ax:       Optional[matplotlib.axes.Axes] = None,
    graph              = None,
    k:        Optional[int]               = None,
) -> None:
    """
    Bar chart of the top_n most probable QAOA measurement outcomes.

    Improvements over original:
      - Bars colored green (valid colorings) vs blue (invalid) vs red (top-1).
      - Probability values shown above tall bars.
      - Cleaner axis styling.

    Args:
        probs:    Probability vector of length 2^n_qubits.
        n_qubits: Total qubit count.
        top_n:    How many top states to show.
        title:    Axes title.
        ax:       Matplotlib Axes (created if None).
        graph:    NetworkX graph (used to check validity of each state).
        k:        Number of colors (needed with graph to check validity).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    top_idx = np.argsort(probs)[-top_n:][::-1]
    top_p   = probs[top_idx]
    labels  = [format(int(i), f"0{min(n_qubits, 8)}b") for i in top_idx]

    # Determine bar colors based on validity
    bar_colors = []
    if graph is not None and k is not None:
        from qgc.core.coloring import decode_one_hot, count_conflicts
        n_nodes = graph.number_of_nodes()
        for rank, idx in enumerate(top_idx):
            col   = decode_one_hot(int(idx), n_nodes, k)
            valid = count_conflicts(col, graph) == 0
            if rank == 0:
                bar_colors.append("#E74C3C")    # top-1: red
            elif valid:
                bar_colors.append("#27AE60")    # valid: green
            else:
                bar_colors.append("#3498DB")    # invalid: blue
    else:
        bar_colors = ["#E74C3C"] + ["#3498DB"] * (len(top_p) - 1)

    bars = ax.bar(range(len(top_p)), top_p,
                  color=bar_colors, edgecolor="white", linewidth=0.7)

    # Label bars above threshold
    threshold = top_p.max() * 0.3
    for bar, val in zip(bars, top_p):
        if val >= threshold:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top_p.max() * 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color="#333")

    ax.set_xticks(range(len(top_p)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Probability", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", alpha=0.18, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Rectangle((0, 0), 1, 1, fc="#E74C3C", label="Top-1 state"),
        mpatches.Rectangle((0, 0), 1, 1, fc="#27AE60", label="Valid coloring"),
        mpatches.Rectangle((0, 0), 1, 1, fc="#3498DB", label="Invalid state"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
              framealpha=0.85)


# ── Multi-optimizer convergence comparison ────────────────────────────

def draw_multi_convergence(
    traces: dict,
    title:  str                          = "Convergence comparison",
    ax:     Optional[matplotlib.axes.Axes] = None,
) -> None:
    """
    Overlay multiple convergence curves on one axes.
    Useful for comparing COBYLA / BFGS / SLSQP (Oh et al. Figs 6, 9, 12).

    Improvements over original:
      - Each curve has a smoothed overlay.
      - Final energy value annotated at right end of each curve.
      - Cleaner axis styling.

    Args:
        traces: {label: energy_list} dict.
        title:  Axes title.
        ax:     Matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    for (label, trace), color in zip(traces.items(), colors):
        if not trace:
            continue
        xs = np.arange(len(trace))
        ys = np.array(trace)

        # Raw (faint)
        ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.3)

        # Smoothed
        if len(ys) >= 7:
            smooth = uniform_filter1d(ys, size=max(3, len(ys) // 12))
            ax.plot(xs, smooth, color=color, linewidth=2.0, alpha=0.95,
                    label=label)
            final_y = smooth[-1]
        else:
            ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.95, label=label)
            final_y = ys[-1]

        # Annotate final value
        ax.annotate(f"{final_y:.1f}", xy=(xs[-1], final_y),
                    xytext=(xs[-1] + len(xs) * 0.01, final_y),
                    fontsize=7, color=color, fontweight="bold", va="center")

    ax.set_xlabel("Function evaluations", fontsize=8)
    ax.set_ylabel("Ising energy ⟨H⟩", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.20, linestyle=":")
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))