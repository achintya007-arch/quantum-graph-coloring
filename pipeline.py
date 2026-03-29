"""
pipeline.py
===========
Runs solvers on a set of test graphs and prints a comparison table.

Usage:
    python pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import networkx as nx

from qgc.algorithms.greedy import greedy_coloring
from qgc.algorithms.qaoa   import QAOA
from qgc.algorithms.qudit  import QuditColoring


def run_pipeline(graph: nx.Graph, k: int, name: str = "", mode: str = "comparative") -> dict:
    """
    Run solvers on one graph according to the selected mode.

    mode="comparative" — run greedy, QAOA (if small enough), and qudit.
    mode="hybrid"      — run greedy; if conflicts == 0 return greedy result,
                         else run QAOA and return that result as "hybrid".

    Returns:
        comparative: {"greedy": ..., "qaoa": ... | None, "qudit": ...}
        hybrid:      {"greedy": ..., "hybrid": ...}
    """
    label = name or f"G({graph.number_of_nodes()},{graph.number_of_edges()})"
    print(f"\n{'─' * 55}")
    print(f"  Graph : {label}  nodes={graph.number_of_nodes()}  edges={graph.number_of_edges()}  k={k}  mode={mode}")

    # ── Greedy (always runs) ──────────────────────────────────────────
    g_res = greedy_coloring(graph)
    print(f"  greedy : coloring={g_res['coloring']}  conflicts={g_res['conflicts']}")

    if mode == "hybrid":
        if g_res["conflicts"] == 0:
            print(f"  hybrid : greedy solved it — skipping QAOA")
            return {"greedy": g_res, "hybrid": g_res}
        else:
            if graph.number_of_nodes() * k <= QAOA.MAX_QUBITS:
                solver     = QAOA(graph, k=k, p=2, penalty=4.0)
                hybrid_res = solver.optimize(n_restarts=8, optimizer="COBYLA")
            else:
                solver     = QuditColoring(graph, k=k, gamma=0.5, lr=0.04, n_steps=800, n_runs=10, h=0.3)
                hybrid_res = solver.optimize()
            print(f"  hybrid : coloring={hybrid_res['coloring']}  conflicts={hybrid_res['conflicts']}")
            return {"greedy": g_res, "hybrid": hybrid_res}

    # ── comparative mode ─────────────────────────────────────────────
    qaoa_res = None
    if graph.number_of_nodes() * k <= QAOA.MAX_QUBITS:
        solver   = QAOA(graph, k=k, p=2, penalty=4.0)
        qaoa_res = solver.optimize(n_restarts=8, optimizer="COBYLA")
        print(f"  qaoa   : coloring={qaoa_res['coloring']}  conflicts={qaoa_res['conflicts']}")
    else:
        print(f"  qaoa   : skipped (n*k={graph.number_of_nodes()*k} > {QAOA.MAX_QUBITS})")

    solver    = QuditColoring(graph, k=k, gamma=0.5, lr=0.04, n_steps=800, n_runs=10, h=0.3)
    qudit_res = solver.optimize()
    print(f"  qudit  : coloring={qudit_res['coloring']}  conflicts={qudit_res['conflicts']}")

    return {"greedy": g_res, "qaoa": qaoa_res, "qudit": qudit_res}


if __name__ == "__main__":
    np.random.seed(42)

    test_cases = [
        ("Triangle K3",    nx.complete_graph(3),              3),
        ("Cycle C4",       nx.cycle_graph(4),                 2),
        ("Star S3",        nx.star_graph(3),                  2),
        ("Petersen graph", nx.petersen_graph(),               3),
        ("Random G(20,35)",nx.gnm_random_graph(20, 35, seed=7), 4),
    ]

    print("=" * 55)
    print("  pipeline.py — Quantum Graph Coloring Pipeline")
    print("=" * 55)

    mode = input("\nSelect mode [comparative/hybrid] (default: comparative): ").strip().lower()
    if mode not in ("comparative", "hybrid"):
        mode = "comparative"
    print(f"  Running in '{mode}' mode …")

    all_results = {}
    for name, G, k in test_cases:
        all_results[name] = run_pipeline(G, k, name=name, mode=mode)

    print("\n" + "=" * 72)
    if mode == "comparative":
        print(f"  {'Graph':<22} {'k':>2} | {'Greedy':>7} {'QAOA':>7} {'Qudit':>7}")
        print("  " + "─" * 50)
        for name, res in all_results.items():
            g  = res["greedy"]["conflicts"]
            qa = res["qaoa"]["conflicts"] if res["qaoa"] else "–"
            qd = res["qudit"]["conflicts"]
            print(f"  {name:<22} –  | {g:>7} {str(qa):>7} {qd:>7}")
    else:
        print(f"  {'Graph':<22} {'k':>2} | {'Greedy':>7} {'Hybrid':>7}")
        print("  " + "─" * 42)
        for name, res in all_results.items():
            g  = res["greedy"]["conflicts"]
            h  = res["hybrid"]["conflicts"]
            print(f"  {name:<22} –  | {g:>7} {h:>7}")
    print("=" * 72)
    print("  Values = chromatic conflicts (0 = valid proper coloring)")
    print("\n[Done]")