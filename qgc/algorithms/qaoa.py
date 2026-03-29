"""
qgc/algorithms/qaoa.py
======================
Statevector QAOA solver for graph k-coloring on small graphs.

Algorithm (Oh et al. 2019, §5; Farhi et al. 2014)
---------------------------------------------------
QAOA alternates two parameterized unitaries for p layers:

  1. Cost unitary:   U_C(γ) = exp(−iγ H_C)
       H_C is diagonal in the computational basis (the Ising Hamiltonian).
       Applied as element-wise phase multiplication on the state vector.

  2. Mixer unitary:  U_B(β) = ⊗_i exp(−iβ X_i)
       Applies an X-rotation independently to each qubit.
       On qubit q:  |0⟩ → cos β|0⟩ − i sin β|1⟩
                    |1⟩ → −i sin β|0⟩ + cos β|1⟩

Starting state: uniform superposition |+⟩^⊗n (all qubits in |+⟩).

Classical optimisers (COBYLA, BFGS, SLSQP from Oh et al.) minimise ⟨H_C⟩.
Multiple random restarts guard against local minima.
Top-K decoding (from qgc.core.coloring) extracts the best valid coloring.

Scalability limit: 2^(n*k) statevector.  Safe up to n*k ≈ 12 qubits.
"""

import numpy as np
from scipy.optimize import minimize
import networkx as nx

from qgc.core.hamiltonian import build_qubo, qubo_to_ising, build_hamiltonian_diagonal
from qgc.core.coloring    import decode_top_k, count_conflicts, coloring_summary


class QAOA:
    """
    Statevector QAOA for graph k-coloring.

    Parameters
    ----------
    graph   : NetworkX graph.
    k       : Number of colors.
    p       : Number of QAOA layers (circuit depth).
    penalty : QUBO penalty coefficient P.
    """

    MAX_QUBITS = 12   # 2^12 = 4096 — stays interactive

    def __init__(self, graph: nx.Graph, k: int, p: int = 2, penalty: float = 4.0):
        self.graph    = graph
        self.k        = k
        self.p        = p
        self.n_nodes  = graph.number_of_nodes()
        self.n_qubits = self.n_nodes * k

        if self.n_qubits > self.MAX_QUBITS:
            raise ValueError(
                f"Graph requires {self.n_qubits} qubits "
                f"(>{self.MAX_QUBITS}). Use QuditColoring instead."
            )

        # Build Ising Hamiltonian diagonal once (reused every circuit call)
        Q          = build_qubo(graph, k, penalty)
        J, h, _    = qubo_to_ising(Q)
        self.H_diag = build_hamiltonian_diagonal(J, h)

    # ── Circuit ───────────────────────────────────────────────────────

    def run_circuit(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the QAOA circuit for given parameters and return the statevector.

        Args:
            params: 1-D array of length 2p.
                    params[:p]  = gamma angles (cost unitary).
                    params[p:]  = beta  angles (mixer unitary).

        Returns:
            state: Complex statevector of length 2^(n_qubits).
        """
        gammas = params[: self.p]
        betas  = params[self.p :]

        # Initialise to uniform superposition |+⟩^n
        dim   = 2 ** self.n_qubits
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)

        for layer in range(self.p):
            # ── Cost unitary: diagonal phase rotation ─────────────────
            state = state * np.exp(-1j * gammas[layer] * self.H_diag)

            # ── Mixer unitary: X-rotation on each qubit ───────────────
            sv = state.reshape([2] * self.n_qubits)
            c  = np.cos(betas[layer])
            s  = np.sin(betas[layer])

            for q in range(self.n_qubits):
                s0  = np.take(sv, 0, axis=q)
                s1  = np.take(sv, 1, axis=q)
                sv  = np.stack([c * s0 - 1j * s * s1,
                                -1j * s * s0 + c * s1], axis=q)
            state = sv.reshape(-1)

        return state

    # ── Cost function for classical optimiser ─────────────────────────

    def expectation(self, params: np.ndarray) -> float:
        """
        Compute ⟨ψ(params)|H_C|ψ(params)⟩ — the quantity QAOA minimises.

        Args:
            params: Circuit parameters (gammas + betas).

        Returns:
            Scalar expected energy.
        """
        probs = np.abs(self.run_circuit(params)) ** 2
        return float(probs @ self.H_diag)

    # ── Optimisation ──────────────────────────────────────────────────

    def optimize(
        self,
        n_restarts: int = 8,
        optimizer:  str = "COBYLA",
        max_iter:   int = 500,
    ) -> dict:
        """
        Classically optimise QAOA parameters with multiple random restarts.

        Args:
            n_restarts: Number of independent random starting points.
            optimizer:  Scipy method name ('COBYLA', 'BFGS', 'SLSQP').
            max_iter:   Maximum iterations per restart.

        Returns:
            result dict:
              "coloring"    – decoded node-color list (top-K)
              "conflicts"   – chromatic conflicts in that coloring
              "meta"        – {"n_colors", "is_valid", "conflict_edges",
                               "energy", "convergence", "probs", "params"}
              (plus backward-compat flat keys n_colors, n_conflicts,
               is_valid, conflict_edges, energy, convergence, probs, params)
        """
        best_energy  = None
        best_params  = None
        best_trace   = []

        for _ in range(n_restarts):
            x0    = np.random.uniform(0, 2 * np.pi, 2 * self.p)
            trace = []

            res = minimize(
                self.expectation, x0,
                method   = optimizer,
                callback = lambda xk: trace.append(self.expectation(xk)),
                options  = {"maxiter": max_iter, "rhobeg": 0.4},
            )

            if best_energy is None or res.fun < best_energy:
                best_energy  = res.fun
                best_params  = res.x
                best_trace   = trace

        # Decode: use top-K to avoid trivial argmax traps
        probs    = np.abs(self.run_circuit(best_params)) ** 2
        coloring = decode_top_k(probs, self.n_nodes, self.k, self.graph, top_k=100)
        summary  = coloring_summary(coloring, self.graph)

        # Merge algorithm-specific fields into meta and as flat keys
        algo_fields = {
            "energy":      best_energy,
            "convergence": best_trace,
            "probs":       probs,
            "params":      best_params,
        }
        summary["meta"].update(algo_fields)
        summary.update(algo_fields)   # backward-compat flat keys

        return summary
