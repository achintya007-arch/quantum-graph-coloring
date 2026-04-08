"""
pipeline.py
===========
Runs solvers on a set of test graphs and prints a comparison table.

Usage:
    python pipeline.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import networkx as nx

from qgc.algorithms.greedy import greedy_coloring
from qgc.algorithms.qaoa   import QAOA
from qgc.algorithms.qudit  import QuditColoring


# ── Metric computation ────────────────────────────────────────────────

def compute_metrics(result: dict, graph: nx.Graph, k: int,
                    elapsed_ms: float) -> dict:
    """
    Derive standardised comparison metrics for one solver result.

    Metrics:
        colors_used       – number of distinct colors assigned
        conflicts         – chromatic conflicts (edges with same-color endpoints)
        conflict_rate     – conflicts / total edges  (0 = perfect)
        color_efficiency  – k_lower / colors_used   (1 = optimal, lower = wasteful)
        valid             – bool, conflicts == 0
        elapsed_ms        – wall-clock time in milliseconds
        energy            – final Ising energy (QAOA only, else None)
        energy_improvement– energy drop from first to last evaluation (QAOA only)
        n_evaluations     – number of cost-function calls (QAOA only, else None)
        score             – composite 0-100: rewards few colors, no conflicts, speed
    """
    n_edges      = graph.number_of_edges()
    colors_used  = len(set(result["coloring"]))
    conflicts    = result["conflicts"]

    # Lower bound on chromatic number = clique number
    try:
        cliques   = list(nx.find_cliques(graph))
        k_lower   = max(len(c) for c in cliques) if cliques else 1
    except Exception:
        k_lower   = 1

    conflict_rate    = conflicts / n_edges if n_edges > 0 else 0.0
    color_efficiency = k_lower / colors_used if colors_used > 0 else 0.0

    # QAOA-specific
    _energy_raw = result.get("energy")
    energy      = float(_energy_raw) if _energy_raw is not None else None    
    convergence        = result.get("convergence") or []
    n_evaluations      = len(convergence) if convergence else None
    energy_improvement = (convergence[0] - convergence[-1]) if len(convergence) >= 2 else None

    # Composite score (0-100): perfect coloring = 100
    # Penalise conflicts heavily, then color overuse, then time
    conflict_penalty  = conflicts * 20
    color_penalty     = max(0, colors_used - k_lower) * 5
    speed_bonus       = max(0, 10 - elapsed_ms / 500)   # up to +10 for being fast
    score             = max(0, min(100, 100 - conflict_penalty - color_penalty + speed_bonus))

    return {
        "colors_used":        colors_used,
        "conflicts":          conflicts,
        "conflict_rate":      round(conflict_rate, 4),
        "color_efficiency":   round(color_efficiency, 4),
        "valid":              conflicts == 0,
        "elapsed_ms":         round(elapsed_ms, 2),
        "energy":             round(energy, 4) if energy is not None else None,
        "energy_improvement": round(energy_improvement, 4) if energy_improvement is not None else None,
        "n_evaluations":      n_evaluations,
        "score":              round(score, 1),
        "k_lower_bound":      k_lower,
    }


# ── Main pipeline ─────────────────────────────────────────────────────

def run_pipeline(graph: nx.Graph, k: int, name: str = "",
                 mode: str = "comparative", force_qaoa: bool = False) -> dict:
    """
    Run solvers on one graph according to the selected mode.

    mode="comparative" — run greedy, QAOA (if small enough), and qudit.
    mode="hybrid"      — run greedy; if conflicts == 0 return greedy result,
                         else run QAOA/qudit and return as "hybrid".

    Every result dict gains a "metrics" key with standardised comparison data.

    Returns:
        comparative: {"greedy": ..., "qaoa": ... | None, "qudit": ...}
        hybrid:      {"greedy": ..., "hybrid": ...}
    """
    label = name or f"G({graph.number_of_nodes()},{graph.number_of_edges()})"
    print(f"\n{'─' * 55}")
    print(f"  Graph : {label}  nodes={graph.number_of_nodes()}  "
          f"edges={graph.number_of_edges()}  k={k}  mode={mode}")

    # ── Greedy ────────────────────────────────────────────────────────
    t0    = time.perf_counter()
    g_res = greedy_coloring(graph)
    g_ms  = (time.perf_counter() - t0) * 1000
    g_res["metrics"] = compute_metrics(g_res, graph, k, g_ms)
    print(f"  greedy : coloring={g_res['coloring']}  "
          f"conflicts={g_res['conflicts']}  time={g_ms:.1f}ms")

    if mode == "hybrid":
        if g_res["conflicts"] == 0 and not force_qaoa:
            print("  hybrid : greedy solved it — skipping QAOA")
            return {"greedy": g_res, "hybrid": g_res}
        else:
            if graph.number_of_nodes() * k <= QAOA.MAX_QUBITS:
                t0         = time.perf_counter()
                solver     = QAOA(graph, k=k, p=2, penalty=4.0)
                hybrid_res = solver.optimize(n_restarts=8, optimizer="COBYLA")
                h_ms       = (time.perf_counter() - t0) * 1000
            else:
                t0         = time.perf_counter()
                solver     = QuditColoring(graph, k=k, gamma=0.5, lr=0.04,
                                           n_steps=800, n_runs=10, h=0.3)
                hybrid_res = solver.optimize()
                h_ms       = (time.perf_counter() - t0) * 1000
            hybrid_res["metrics"] = compute_metrics(hybrid_res, graph, k, h_ms)
            print(f"  hybrid : coloring={hybrid_res['coloring']}  "
                  f"conflicts={hybrid_res['conflicts']}  time={h_ms:.1f}ms")
            return {"greedy": g_res, "hybrid": hybrid_res}

    # ── comparative mode ──────────────────────────────────────────────
    MAX_QAOA_QUBITS = 12
    total_qubits = graph.number_of_nodes() * k
    qaoa_res = None
    if total_qubits <= MAX_QAOA_QUBITS:
        t0       = time.perf_counter()
        solver   = QAOA(graph, k=k, p=2, penalty=4.0)
        qaoa_res = solver.optimize(n_restarts=8, optimizer="COBYLA")
        q_ms     = (time.perf_counter() - t0) * 1000
        qaoa_res["metrics"] = compute_metrics(qaoa_res, graph, k, q_ms)
        qaoa_res["demo_status"] = "executed"
        print(f"  qaoa   : coloring={qaoa_res['coloring']}  "
              f"conflicts={qaoa_res['conflicts']}  time={q_ms:.1f}ms")
    elif force_qaoa:
        print(f"  qaoa   : skipped in demo mode (n*k={total_qubits} > {MAX_QAOA_QUBITS})")
        qaoa_res = {
            "status": "skipped",
            "reason": "qubit_limit",
            "coloring": None,
            "conflicts": None,
            "energy": None,
            "demo_status": "skipped_too_large",
        }
    else:
        print(f"  qaoa   : skipped (n*k={total_qubits} > {MAX_QAOA_QUBITS})")

    t0        = time.perf_counter()
    solver    = QuditColoring(graph, k=k, gamma=0.5, lr=0.04,
                              n_steps=800, n_runs=10, h=0.3)
    qudit_res = solver.optimize()
    qd_ms     = (time.perf_counter() - t0) * 1000
    qudit_res["metrics"] = compute_metrics(qudit_res, graph, k, qd_ms)
    print(f"  qudit  : coloring={qudit_res['coloring']}  "
          f"conflicts={qudit_res['conflicts']}  time={qd_ms:.1f}ms")

    return {"greedy": g_res, "qaoa": qaoa_res, "qudit": qudit_res}


# ── CLI runner ────────────────────────────────────────────────────────

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

    # ── Conflict table ────────────────────────────────────────────────
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
            g = res["greedy"]["conflicts"]
            h = res["hybrid"]["conflicts"]
            print(f"  {name:<22} –  | {g:>7} {h:>7}")
    print("=" * 72)

    # ── Metrics table ─────────────────────────────────────────────────
    if mode == "comparative":
        print("\n" + "=" * 90)
        print("  COMPARISON METRICS")
        print("=" * 90)
        print(f"  {'Graph':<20} {'Solver':<8} {'Colors':>6} {'Conflicts':>9} "
              f"{'Conf%':>6} {'Efficiency':>10} {'Time(ms)':>9} {'Score':>6}")
        print("  " + "─" * 78)
        for name, res in all_results.items():
            for solver_name, r in res.items():
                if r is None:
                    continue
                m = r["metrics"]
                print(f"  {name:<20} {solver_name:<8} {m['colors_used']:>6} "
                      f"{m['conflicts']:>9} {m['conflict_rate']*100:>5.1f}% "
                      f"{m['color_efficiency']:>10.3f} {m['elapsed_ms']:>9.1f} "
                      f"{m['score']:>6.1f}")
        print("=" * 90)
        print("  Efficiency = clique_number / colors_used (1.0 = optimal)")
        print("  Score      = composite 0-100 (conflicts, colors, speed)")

    print("\n[Done]")