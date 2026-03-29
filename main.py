"""
main.py
=======
Entry point for the Quantum Graph Coloring package.

Run:
    python main.py

Package layout
--------------
    qgc/
    ├── core/
    │   ├── hamiltonian.py   — QUBO, Ising, Hamiltonian diagonal
    │   └── coloring.py      — decode, count conflicts, validate
    ├── algorithms/
    │   ├── qaoa.py          — statevector QAOA (small graphs)
    │   ├── qudit.py         — qudit-inspired gradient descent (large graphs)
    │   └── greedy.py        — classical greedy baseline
    ├── visualization/
    │   ├── graph_plots.py   — colored graph drawings, QUBO heatmap
    │   └── convergence.py   — convergence curves, probability bars
    └── demo/
        └── runner.py        — orchestrator: runs all steps, saves figures
"""

import sys
import os

# Ensure the package root is on the Python path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from qgc.demo.runner import run_full_demo

if __name__ == "__main__":
    print("Select mode:")
    print("  [1] comparative  — run greedy, QAOA, and qudit side-by-side")
    print("  [2] hybrid       — run greedy; escalate to QAOA only if needed")
    choice = input("Enter mode [comparative/hybrid] (default: comparative): ").strip().lower()
    if choice not in ("comparative", "hybrid", "1", "2"):
        choice = "comparative"
    if choice == "1":
        choice = "comparative"
    if choice == "2":
        choice = "hybrid"

    run_full_demo(output_dir="/mnt/user-data/outputs", mode=choice)