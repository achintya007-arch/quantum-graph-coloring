"""
qgc/core/hamiltonian.py
=======================
Converts a graph k-coloring problem into its mathematical physics form.

Pipeline
--------
  Graph + k  →  QUBO matrix Q  →  Ising model (J, h)  →  Diagonal Hamiltonian

Theory (Oh et al. 2019, §3)
----------------------------
Variables: x_{i,c} = 1 if node i gets color c  (one-hot encoding)
Flat index: idx(i, c) = i * k + c

Two penalty terms added to the cost function:

  1) Each node gets exactly one color:
       P * (Σ_c x_{ic} − 1)²  for every node i

  2) Adjacent nodes must differ in color:
       P * x_{ip} * x_{jp}     for every edge (i,j) and color p

QUBO  →  Ising substitution:  x = (1 − z) / 2,  z ∈ {−1, +1}
  H_ising = Σ_{i<j} J_ij z_i z_j  +  Σ_i h_i z_i  +  const
"""

import numpy as np
import networkx as nx


# ──────────────────────────────────────────────────────────────────────
# QUBO
# ──────────────────────────────────────────────────────────────────────

def build_qubo(graph: nx.Graph, k: int, penalty: float = 4.0) -> np.ndarray:
    """
    Build the QUBO matrix Q for the k-coloring problem.

    Args:
        graph:   NetworkX graph to color.
        k:       Number of colors.
        penalty: Lagrange multiplier P for constraint violations.

    Returns:
        Q: (n*k) × (n*k) upper-triangular QUBO matrix.
    """
    n   = graph.number_of_nodes()
    dim = n * k
    Q   = np.zeros((dim, dim))

    # ── Constraint 1: exactly one color per node ──────────────────────
    # Expanding P*(Σ_c x_{ic} - 1)² gives:
    #   diagonal terms:   -P * x_{ic}      (from -2*1*x + x²  with x²=x)
    #   off-diagonal:     +2P * x_{ic1} * x_{ic2}  for c1 < c2
    for i in range(n):
        for c in range(k):
            Q[i*k + c, i*k + c] -= penalty                    # diagonal
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                Q[i*k + c1, i*k + c2] += 2 * penalty          # off-diagonal

    # ── Constraint 2: adjacent nodes have different colors ────────────
    # For edge (u,v): P * Σ_c x_{uc} * x_{vc}
    for (u, v) in graph.edges():
        for c in range(k):
            Q[u*k + c, v*k + c] += penalty

    return Q


def evaluate_qubo(Q: np.ndarray, coloring: list, k: int) -> float:
    """
    Compute the QUBO objective value for a given coloring assignment.

    Args:
        Q:        QUBO matrix.
        coloring: List of color indices, one per node.
        k:        Number of colors.

    Returns:
        Scalar QUBO energy (0.0 means a valid, conflict-free coloring).
    """
    n = len(coloring)
    x = np.zeros(n * k)
    for i, c in enumerate(coloring):
        x[i * k + c] = 1.0
    return float(x @ Q @ x)


# ──────────────────────────────────────────────────────────────────────
# Ising conversion
# ──────────────────────────────────────────────────────────────────────

def qubo_to_ising(Q: np.ndarray):
    """
    Convert a QUBO matrix to an Ising Hamiltonian via  x = (1 − z) / 2.

    H_ising = Σ_{i<j} J_ij z_i z_j  +  Σ_i h_i z_i  +  offset

    Args:
        Q: (n×n) QUBO matrix.

    Returns:
        J:      (n×n) symmetric coupling matrix (zero diagonal).
        h:      (n,) local field vector.
        offset: Constant energy offset.
    """
    n      = Q.shape[0]
    J      = np.zeros((n, n))
    h      = np.zeros(n)
    offset = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            w           = (Q[i, j] + Q[j, i]) / 4.0
            J[i, j]     = w
            J[j, i]     = w
            h[i]        -= w
            h[j]        -= w
            offset      += w
        diag     = Q[i, i]
        h[i]    += diag / 2.0
        offset  -= diag / 2.0

    return J, h, offset


# ──────────────────────────────────────────────────────────────────────
# Diagonal Hamiltonian (full 2^n vector, for statevector QAOA)
# ──────────────────────────────────────────────────────────────────────

def build_hamiltonian_diagonal(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Build the diagonal of the Ising Hamiltonian over the full 2^n Hilbert space.

    H_diag[s] = Σ_{i<j} J_ij z_i(s) z_j(s)  +  Σ_i h_i z_i(s)

    where z_i(s) = +1 if bit i of s is 0, else −1.

    This vectorized construction avoids Python loops over all 2^n states.

    Args:
        J: (n×n) Ising coupling matrix.
        h: (n,) local field vector.

    Returns:
        H_diag: (2^n,) array of Hamiltonian diagonal entries.
    """
    n    = len(h)
    dim  = 2 ** n
    idx  = np.arange(dim, dtype=np.int64)

    # z_matrix[s, i] = +1 if bit i of s is 0, else -1
    z = ((idx[:, None] >> np.arange(n)[None, :]) & 1) * -2 + 1   # shape (2^n, n)

    # Linear term:  Σ_i h_i z_i
    H_diag = z @ h

    # Quadratic term: Σ_{i<j} J_ij z_i z_j
    H_diag += np.einsum('bi, ij, bj -> b', z, np.triu(J, 1), z)

    return H_diag


# ──────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────

def qubo_landscape_table(graph: nx.Graph, k: int, penalty: float = 4.0) -> list:
    """
    Enumerate all 2^(n*k) bitstrings and return their QUBO energies.
    Intended for tiny graphs (n*k ≤ 16) for educational display.

    Returns:
        List of dicts: {bits, x_vec, energy, valid_coloring}
    """
    n   = graph.number_of_nodes()
    dim = n * k
    Q   = build_qubo(graph, k, penalty)
    rows = []

    for s in range(2 ** dim):
        bits = [(s >> i) & 1 for i in range(dim)]
        x    = np.array(bits, dtype=float)
        e    = float(x @ Q @ x)

        # A coloring is valid if: each node has exactly one color AND no conflict
        one_hot_ok = all(sum(bits[i*k:(i+1)*k]) == 1 for i in range(n))
        no_conflict = all(
            not (bits[u*k+c] == 1 and bits[v*k+c] == 1)
            for (u, v) in graph.edges()
            for c in range(k)
        )
        rows.append({
            "bits":           bits,
            "bitstring":      bin(s)[2:].zfill(dim),
            "energy":         e,
            "valid_coloring": one_hot_ok and no_conflict,
        })

    return rows
