"""
qgc/demo/runner.py
==================
Orchestration layer — ties every module together into a single demo pipeline.

Role of each module in the pipeline
-------------------------------------
  core/hamiltonian.py    → Build QUBO Q, convert to Ising (J,h), build H_diag
  core/coloring.py       → Decode bitstrings, count conflicts, validate
  algorithms/qaoa.py     → Statevector QAOA for small graphs (n*k ≤ 12)
  algorithms/qudit.py    → Qudit-inspired gradient descent for larger graphs
  algorithms/greedy.py   → Classical greedy baseline (reference)
  visualization/graph_plots.py   → Colored graph figures + QUBO heatmap
  visualization/convergence.py   → Convergence curves + probability bars
  demo/runner.py         → (this file) orchestrates everything

Usage
-----
  from qgc.demo.runner import run_full_demo
  run_full_demo(output_dir="/mnt/user-data/outputs")
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Core
from qgc.core.hamiltonian import build_qubo, qubo_landscape_table

# ── Algorithms
from qgc.algorithms.qaoa   import QAOA
from qgc.algorithms.qudit  import QuditColoring
from qgc.algorithms.greedy import greedy_coloring

# ── Visualization
from qgc.visualization.graph_plots  import draw_colored_graph, draw_qubo_heatmap
from qgc.visualization.convergence  import draw_convergence, draw_probability_distribution


# ─────────────────────────────────────────────────────────────────────
# Test graph catalogue
# ─────────────────────────────────────────────────────────────────────

def get_test_cases() -> list:
    """
    Return a list of (name, graph, k, method) tuples.

    QAOA is used when n*k ≤ QAOA.MAX_QUBITS (=12).
    QuditColoring is used for larger graphs.
    """
    return [
        ("Triangle K₃",     nx.complete_graph(3),              3, "qaoa"),
        ("Cycle C₄",        nx.cycle_graph(4),                 2, "qaoa"),
        ("Star S₃",         nx.star_graph(3),                  2, "qaoa"),
        ("Petersen graph",  nx.petersen_graph(),               3, "qudit"),
        ("Random G(20,35)", nx.gnm_random_graph(20, 35, seed=7), 4, "qudit"),
    ]


# ─────────────────────────────────────────────────────────────────────
# QUBO landscape section
# ─────────────────────────────────────────────────────────────────────

def print_qubo_landscape() -> None:
    """
    Print and visualise the QUBO energy landscape for the simplest case:
    a single edge (2 nodes, 2 colors, 4 binary variables).
    Demonstrates that valid colorings = global minima of the QUBO.
    """
    print("\n" + "=" * 60)
    print("  MODULE: core/hamiltonian.py")
    print("  QUBO Energy Landscape — 2-node, 2-color (single edge)")
    print("=" * 60)

    G   = nx.Graph([(0, 1)])
    rows = qubo_landscape_table(G, k=2, penalty=4.0)

    print(f"\n{'Bits':>6} | {'x00':>4}{'x01':>4}{'x10':>4}{'x11':>4} | {'Energy':>8} | Valid?")
    print("-" * 52)
    for r in rows:
        b  = r["bits"]
        e  = r["energy"]
        ok = "✓  ← global minimum" if r["valid_coloring"] else "✗"
        print(f"{r['bitstring']:>6} | {''.join(f'{x:>4}' for x in b)} | {e:>8.1f} | {ok}")


# ─────────────────────────────────────────────────────────────────────
# Single-case solver (used inside the main loop)
# ─────────────────────────────────────────────────────────────────────

def solve_case(name: str, G: nx.Graph, k: int, method: str, mode: str = "comparative") -> dict:
    """
    Run greedy + quantum solver on one test case.

    Args:
        name:   Human-readable graph name.
        G:      NetworkX graph.
        k:      Number of colors.
        method: "qaoa" or "qudit".
        mode:   "comparative" (default) or "hybrid".

    Returns:
        dict with keys: name, G, k, method,
                        greedy_result, quantum_result,
                        quantum_label, solver (for extra fields like probs)
    """
    print(f"\n{'─' * 55}")
    print(f"  MODULE: algorithms/{'qaoa' if method == 'qaoa' else 'qudit'}.py  +  algorithms/greedy.py")
    print(f"  Graph : {name}  |  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  k={k}")

    # ── Greedy baseline ───────────────────────────────────────────────
    g_result = greedy_coloring(G)
    print(f"  Greedy  : {g_result['n_colors']} colors, {g_result['n_conflicts']} conflicts "
          f"→ {g_result['coloring']}")

    # ── Hybrid mode: return greedy if it already solved it ────────────
    if mode == "hybrid" and g_result["conflicts"] == 0:
        print(f"  Hybrid  : greedy solved it — skipping quantum solver")
        return {
            "name":           name,
            "G":              G,
            "k":              k,
            "method":         method,
            "greedy_result":  g_result,
            "quantum_result": g_result,
            "quantum_label":  "Hybrid (greedy sufficient)",
            "solver":         None,
        }

    # ── Quantum / quantum-inspired ────────────────────────────────────
    if method == "qaoa":
        solver       = QAOA(G, k=k, p=2, penalty=4.0)
        q_result     = solver.optimize(n_restarts=8, optimizer="COBYLA")
        q_label      = f"QAOA p=2  ({solver.n_qubits} qubits)" if mode == "comparative" else f"Hybrid/QAOA p=2  ({solver.n_qubits} qubits)"
        print(f"  QAOA    : {q_result['n_colors']} colors, {q_result['n_conflicts']} conflicts "
              f"→ {q_result['coloring']}  |  ⟨H⟩ = {q_result['energy']:.3f}")
    else:
        solver       = QuditColoring(G, k=k, gamma=0.5, lr=0.04, n_steps=800, n_runs=10, h=0.3)
        q_result     = solver.optimize()
        q_label      = "Qudit-Inspired (Jansen 2024)" if mode == "comparative" else "Hybrid/Qudit-Inspired"
        print(f"  Qudit   : {q_result['n_colors']} colors, {q_result['n_conflicts']} conflicts "
              f"→ {q_result['coloring']}")

    return {
        "name":           name,
        "G":              G,
        "k":              k,
        "method":         method,
        "greedy_result":  g_result,
        "quantum_result": q_result,
        "quantum_label":  q_label,
        "solver":         solver,
    }


# ─────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────

def build_results_figure(cases_data: list, output_path: str,
                         mode: str = "comparative") -> None:
    """
    Build and save the main results figure: one row per graph,
    three columns:
      Col 1 — Greedy coloring
      Col 2 — Quantum / Hybrid coloring
      Col 3 — QAOA: convergence curve + probability histogram (stacked)
               Qudit / Hybrid-greedy: summary text panel

    Args:
        cases_data:  List of dicts from solve_case().
        output_path: File path for the saved PNG.
        mode:        "comparative" or "hybrid" — affects title and col-3 content.
    """
    print(f"\n{'─' * 55}")
    print("  MODULE: visualization/graph_plots.py + visualization/convergence.py")
    print("  Building results figure …")

    from qgc.visualization.graph_plots import draw_colored_graph
    from qgc.visualization.convergence import (draw_convergence,
                                               draw_probability_distribution)

    n_rows = len(cases_data)
    # Each row: [graph_greedy | graph_quantum | convergence/probs]
    fig = plt.figure(figsize=(17, 5.2 * n_rows))
    fig.patch.set_facecolor("#FAFAFA")

    mode_tag = "Comparative" if mode == "comparative" else "Hybrid"
    fig.suptitle(
        f"Quantum Graph Coloring — {mode_tag} Mode\n"
        "QAOA (Oh et al. 2019)  ·  Qudit-Inspired GD (Jansen et al. 2024)  ·  Greedy baseline",
        fontsize=13, fontweight="bold", y=1.01, color="#222",
    )

    for row, case in enumerate(cases_data):
        name    = case["name"]
        G       = case["G"]
        g_res   = case["greedy_result"]
        q_res   = case["quantum_result"]
        q_label = case["quantum_label"]
        method  = case["method"]
        solver  = case["solver"]

        # Build subplot grid: cols 1-2 are graph panels (equal width),
        # col 3 splits into top (convergence) and bottom (probs) for QAOA
        gs_row = fig.add_gridspec(
            n_rows, 3,
            top=1 - row / n_rows - 0.01,
            bottom=1 - (row + 1) / n_rows + 0.02,
            left=0.03, right=0.97,
            wspace=0.32, hspace=0.55,
        )

        ax_g = fig.add_subplot(gs_row[row, 0])
        ax_q = fig.add_subplot(gs_row[row, 1])

        has_qaoa_data = (method == "qaoa" and solver is not None
                         and q_res.get("convergence"))
        has_probs     = (method == "qaoa" and solver is not None
                         and q_res.get("probs") is not None)

        if has_qaoa_data and has_probs:
            # Split col 3 vertically: convergence top, probs bottom
            inner = gs_row[row, 2].subgridspec(2, 1, hspace=0.55)
            ax_c  = fig.add_subplot(inner[0])
            ax_p  = fig.add_subplot(inner[1])
        else:
            ax_c = fig.add_subplot(gs_row[row, 2])
            ax_p = None

        # ── Column 1: greedy ──────────────────────────────────────────
        draw_colored_graph(
            G, g_res["coloring"],
            title=f"Greedy  [{name}]",
            ax=ax_g,
        )

        # ── Column 2: quantum / hybrid ────────────────────────────────
        draw_colored_graph(
            G, q_res["coloring"],
            title=q_label,
            ax=ax_q,
        )

        # ── Column 3: QAOA diagnostic or info panel ───────────────────
        if has_qaoa_data:
            draw_convergence(
                q_res["convergence"],
                title=f"Convergence  [{name}]",
                ax=ax_c,
                color="#E74C3C",
            )
        elif has_probs:
            draw_probability_distribution(
                q_res["probs"],
                n_qubits=solver.n_qubits,
                title=f"Measurement distribution  [{name}]",
                ax=ax_c,
                graph=G,
                k=case["k"],
            )
        elif mode == "hybrid" and solver is None:
            # Hybrid: greedy was sufficient — show greedy stats
            ax_c.set_facecolor("#F0FFF4")
            ax_c.text(0.5, 0.6,
                      "✓  Greedy solved this graph\n"
                      "   No quantum step needed.",
                      ha="center", va="center", transform=ax_c.transAxes,
                      fontsize=10, color="#27AE60", fontweight="bold",
                      linespacing=1.9)
            ax_c.text(0.5, 0.25,
                      f"Colors: {g_res['n_colors']}   Conflicts: {g_res['n_conflicts']}",
                      ha="center", va="center", transform=ax_c.transAxes,
                      fontsize=9, color="#555")
            ax_c.set_title(f"Hybrid result  [{name}]", fontsize=9, fontweight="bold")
            ax_c.axis("off")
        else:
            ax_c.text(
                0.5, 0.5,
                "Qudit-Inspired method:\nno quantum circuit.\n"
                "Optimises probability vectors\nvia projected Adam GD.",
                ha="center", va="center", transform=ax_c.transAxes,
                fontsize=9.5, color="#555", linespacing=1.9,
            )
            ax_c.set_title(f"Method note  [{name}]", fontsize=9, fontweight="bold")
            ax_c.axis("off")

        if ax_p is not None and has_probs:
            draw_probability_distribution(
                q_res["probs"],
                n_qubits=solver.n_qubits,
                title=f"Measurement distribution  [{name}]",
                ax=ax_p,
                graph=G,
                k=case["k"],
            )

    plt.savefig(output_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [✓] Main figure saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Standalone: QUBO heatmap figure
# ─────────────────────────────────────────────────────────────────────

def build_qubo_figure(cases_data: list, output_path: str) -> None:
    """
    Save a standalone figure showing the QUBO matrix heatmap for every
    QAOA graph (small enough to display meaningfully).

    Args:
        cases_data:  List of dicts from solve_case().
        output_path: File path for the saved PNG.
    """
    from qgc.core.hamiltonian import build_qubo
    from qgc.visualization.graph_plots import draw_qubo_heatmap

    qaoa_cases = [c for c in cases_data if c["method"] == "qaoa"]
    if not qaoa_cases:
        print("  [!] No QAOA cases — skipping QUBO figure.")
        return

    n      = len(qaoa_cases)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("QUBO Matrix Q for each QAOA graph\n"
                 "Diagonal = single-color penalty · Off-diagonal = conflict penalty",
                 fontsize=11, fontweight="bold")

    if n == 1:
        axes = [axes]

    for ax, case in zip(axes, qaoa_cases):
        Q = build_qubo(case["G"], case["k"], penalty=4.0)
        draw_qubo_heatmap(
            Q, ax=ax,
            title=f"{case['name']}  (k={case['k']})",
            k=case["k"],
            n=case["G"].number_of_nodes(),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [✓] QUBO figure saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Standalone: summary bar chart figure
# ─────────────────────────────────────────────────────────────────────

def build_summary_figure(cases_data: list, output_path: str,
                         mode: str = "comparative") -> None:
    """
    Save a standalone figure with two grouped bar charts:
      Left  — colors used per graph per solver.
      Right — conflicts per graph per solver.

    Args:
        cases_data:  List of dicts from solve_case().
        output_path: File path for the saved PNG.
        mode:        "comparative" or "hybrid".
    """
    from qgc.visualization.graph_plots import draw_summary_bars

    fig, (ax_col, ax_conf) = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.patch.set_facecolor("#FAFAFA")
    mode_tag = "Comparative" if mode == "comparative" else "Hybrid"
    fig.suptitle(f"Result Summary — {mode_tag} Mode\n"
                 "Colors used and chromatic conflicts per graph and solver",
                 fontsize=11, fontweight="bold")

    draw_summary_bars(cases_data, mode=mode, ax_colors=ax_col, ax_conf=ax_conf)
    ax_col.set_title("Colors Used", fontsize=10, fontweight="bold")
    ax_conf.set_title("Chromatic Conflicts", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [✓] Summary figure saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────

def print_summary_table(cases_data: list) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"{'Graph':<22} {'k':>2} {'Method':>12} │ {'G-col':>5} {'Q-col':>5} {'G-conf':>6} {'Q-conf':>6} {'Valid':>6}")
    print("─" * 72)
    for c in cases_data:
        g = c["greedy_result"]; q = c["quantum_result"]
        print(
            f"{c['name']:<22} {c['k']:>2} {c['method']:>12} │ "
            f"{g['n_colors']:>5} {q['n_colors']:>5} "
            f"{g['n_conflicts']:>6} {q['n_conflicts']:>6} "
            f"{'✓' if q['is_valid'] else '✗':>6}"
        )
    print("=" * 72)
    print("  col=colors used  │  conf=chromatic conflicts  │  Valid=conflict-free")


# ─────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────

def run_full_demo(output_dir: str = "/mnt/user-data/outputs", mode: str = "comparative") -> None:
    """
    Full end-to-end demo:
      1. Print QUBO landscape (core/hamiltonian.py)
      2. Solve each test case (algorithms/*.py)
      3. Build and save figures (visualization/*.py)
      4. Print summary table
    """
    print(f"\n  Mode: {mode}")
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: QUBO landscape ────────────────────────────────────────
    print_qubo_landscape()

    # ── Step 2: Solve all cases ───────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  MODULE: algorithms/* + core/*")
    print("  Running solvers …")
    print("=" * 60)

    cases_data = []
    for (name, G, k, method) in get_test_cases():
        cases_data.append(solve_case(name, G, k, method, mode=mode))

    # ── Step 3: Figures ───────────────────────────────────────────────
    build_results_figure(
        cases_data,
        output_path=os.path.join(output_dir, "quantum_graph_coloring_results.png"),
        mode=mode,
    )
    build_qubo_figure(
        cases_data,
        output_path=os.path.join(output_dir, "quantum_graph_coloring_qubo.png"),
    )
    build_summary_figure(
        cases_data,
        output_path=os.path.join(output_dir, "quantum_graph_coloring_summary.png"),
        mode=mode,
    )

    # ── Step 4: Summary ───────────────────────────────────────────────
    print_summary_table(cases_data)
    print("\n[Done]")