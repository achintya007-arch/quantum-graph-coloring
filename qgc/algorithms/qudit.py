"""
qgc/algorithms/qudit.py
========================
Qudit-inspired gradient descent for graph k-coloring on larger graphs.

Algorithm (Jansen et al. 2024, Phys. Rev. Applied 22, 064002)
--------------------------------------------------------------
Each node i is represented by a probability vector  p_i ∈ Δ^(k−1)
(the (k−1)-simplex: non-negative entries summing to 1).

Cost function minimised:
    E(P) = Σ_{(i,j)∈E} J_{ij} · p_i^T p_j
           + γ · Σ_i p_i^T log(p_i)      ← entropy regulariser

  - The edge term penalises nodes sharing probability mass on the same color.
    J_{ij} = 1 + h · Uniform(0,1)  adds stochastic perturbation to escape
    local minima (Jansen eq. 9).
  - The entropy term (coefficient γ) prevents premature collapse to one color
    and aids exploration.

Optimisation: Projected Adam gradient descent.
  - Gradient is computed analytically per-step.
  - Each iterate is projected back onto the simplex Δ^(k−1).

Decoding: argmax(p_i) assigns a definite color to each node.

Multiple random restarts select the coloring with fewest conflicts.

Scalability: O(n·k) parameters → scales to hundreds of nodes.
"""

import numpy as np
import networkx as nx

from qgc.core.coloring import count_conflicts, coloring_summary


class QuditColoring:
    """
    Qudit-inspired gradient descent coloring solver.

    Parameters
    ----------
    graph   : NetworkX graph.
    k       : Number of colors.
    gamma   : Entropy regulariser weight (Jansen eq. 10).
    lr      : Adam learning rate.
    n_steps : Gradient descent steps per run.
    n_runs  : Number of independent random restarts.
    h       : Stochastic interaction strength (Jansen eq. 9).
    """

    def __init__(
        self,
        graph:   nx.Graph,
        k:       int,
        gamma:   float = 0.5,
        lr:      float = 0.04,
        n_steps: int   = 800,
        n_runs:  int   = 10,
        h:       float = 0.3,
    ):
        self.graph   = graph
        self.k       = k
        self.gamma   = gamma
        self.lr      = lr
        self.n_steps = n_steps
        self.n_runs  = n_runs
        self.h       = h
        self.n_nodes = graph.number_of_nodes()
        self.edges   = list(graph.edges())

    # ── Simplex projection ────────────────────────────────────────────

    def _project_simplex(self, V: np.ndarray) -> np.ndarray:
        """
        Project each row of V onto the probability simplex Δ^(k−1).
        Uses the O(k log k) sorting algorithm (Duchi et al. 2008).
        """
        out = np.zeros_like(V)
        for i, row in enumerate(V):
            u   = np.sort(row)[::-1]
            rho = np.where(u > (np.cumsum(u) - 1) / (np.arange(len(u)) + 1))[0][-1]
            lam = (np.sum(u[: rho + 1]) - 1) / (rho + 1)
            out[i] = np.maximum(row - lam, 0)
        return out

    # ── Single optimisation run ───────────────────────────────────────

    def _single_run(self) -> tuple:
        """
        Run one full Adam optimisation from a random starting point.

        Returns:
            (coloring, n_conflicts)  — coloring is a list of plain int.
        """
        # Random initialisation on the simplex via softmax
        logits = np.random.randn(self.n_nodes, self.k)
        e_x    = np.exp(logits - logits.max(axis=1, keepdims=True))
        P      = e_x / e_x.sum(axis=1, keepdims=True)

        # Adam state
        m  = np.zeros_like(P)
        v  = np.zeros_like(P)
        b1, b2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.n_steps + 1):
            # ── Gradient of edge cost  Σ J_ij p_i^T p_j ──────────────
            g = np.zeros_like(P)
            for (i, j) in self.edges:
                J_ij  = 1.0 + self.h * np.random.rand()   # stochastic perturbation
                g[i] += J_ij * P[j]
                g[j] += J_ij * P[i]

            # ── Gradient of entropy regulariser  γ Σ p_i log p_i ─────
            g += self.gamma * (np.log(P + 1e-12) + 1.0)

            # ── Adam update ───────────────────────────────────────────
            m   = b1 * m + (1 - b1) * g
            v   = b2 * v + (1 - b2) * g ** 2
            m_h = m / (1 - b1 ** t)
            v_h = v / (1 - b2 ** t)
            P   = P - self.lr * m_h / (np.sqrt(v_h) + eps)

            # ── Project back onto simplex ─────────────────────────────
            P = self._project_simplex(P)

        # Decode: cast to plain int to avoid np.int64 in output
        coloring = [int(c) for c in np.argmax(P, axis=1)]
        return coloring, count_conflicts(coloring, self.graph)

    # ── Public interface ──────────────────────────────────────────────

    def optimize(self) -> dict:
        """
        Run n_runs independent optimisations and return the best result.

        Returns:
            result dict:
              "coloring"  – list of int, node-color assignment
              "conflicts" – int, number of chromatic conflicts
              "meta"      – {"n_colors", "is_valid", "conflict_edges"}
              (plus backward-compat flat keys n_colors, n_conflicts,
               is_valid, conflict_edges)
        """
        best_coloring  = None
        best_conflicts = 10 ** 9

        for _ in range(self.n_runs):
            coloring, n_conf = self._single_run()
            if n_conf < best_conflicts:
                best_coloring  = coloring
                best_conflicts = n_conf

        return coloring_summary(best_coloring, self.graph)
