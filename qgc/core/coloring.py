"""
qgc/core/coloring.py
====================
Shared utilities for representing, decoding, and validating graph colorings.

A "coloring" throughout this package is a plain Python list of integers:
    coloring[i] = color assigned to node i  (0-indexed)
"""

import networkx as nx


# ──────────────────────────────────────────────────────────────────────
# Decoding
# ──────────────────────────────────────────────────────────────────────

def decode_one_hot(bitstring_index: int, n_nodes: int, k: int) -> list:
    """
    Decode a one-hot QUBO bitstring index into a node-color list.

    Variable ordering: x_{i,c} at bit position i*k + c.
    If no bit is set for a node (invalid state), defaults to color 0.

    Args:
        bitstring_index: Integer index of the measured basis state.
        n_nodes:         Number of graph nodes.
        k:               Number of colors.

    Returns:
        coloring: List of length n_nodes with color indices in [0, k).
    """
    n_qubits = n_nodes * k
    bits     = [(bitstring_index >> q) & 1 for q in range(n_qubits)]
    coloring = []

    for i in range(n_nodes):
        one_hot = bits[i * k : (i + 1) * k]
        color = one_hot.index(1) if 1 in one_hot else 0
        coloring.append(color)

    return coloring


def decode_top_k(probability_vector, n_nodes: int, k: int,
                 graph: nx.Graph, top_k: int = 100) -> list:
    """
    Top-K decoding: examine the top_k most probable basis states and return
    the coloring with the fewest chromatic conflicts.

    Args:
        probability_vector: Array of shape (2^(n*k),) with measurement probs.
        n_nodes:            Number of graph nodes.
        k:                  Number of colors.
        graph:              NetworkX graph (used to count conflicts).
        top_k:              How many top states to inspect.

    Returns:
        best_coloring: List of color assignments with minimum conflicts.
    """
    import numpy as np
    top_indices         = np.argsort(probability_vector)[-top_k:][::-1]
    best_coloring       = None
    best_conflict_count = 10 ** 9

    for idx in top_indices:
        coloring = decode_one_hot(int(idx), n_nodes, k)
        n_conf   = count_conflicts(coloring, graph)
        if n_conf < best_conflict_count:
            best_coloring       = coloring
            best_conflict_count = n_conf

    return best_coloring


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────

def count_conflicts(coloring: list, graph: nx.Graph) -> int:
    """
    Count the number of edges where both endpoints share the same color.

    Args:
        coloring: Node-color assignment list.
        graph:    NetworkX graph.

    Returns:
        Number of conflicting edges (0 = valid proper coloring).
    """
    return sum(
        1 for u, v in graph.edges()
        if coloring[u] == coloring[v]
    )


def is_valid_coloring(coloring: list, graph: nx.Graph) -> bool:
    """Return True if the coloring has zero chromatic conflicts."""
    return count_conflicts(coloring, graph) == 0


def coloring_summary(coloring: list, graph: nx.Graph) -> dict:
    """
    Return a result dict for a coloring.

    Mandatory output format:
        {
          "coloring":  list of int,
          "conflicts": int,
          "meta": {
            "n_colors":       int,
            "is_valid":       bool,
            "conflict_edges": list of (u, v),
          }
        }

    Backward-compat flat keys (n_colors, n_conflicts, is_valid,
    conflict_edges) are also present so runner.py needs no changes.
    """
    coloring = [int(c) for c in coloring]   # normalise np.int64 -> int
    conflict_edges = [
        (u, v) for u, v in graph.edges()
        if coloring[u] == coloring[v]
    ]
    n_conflicts = len(conflict_edges)
    n_colors    = len(set(coloring))
    is_valid    = n_conflicts == 0

    return {
        # Mandatory output format
        "coloring":  coloring,
        "conflicts": n_conflicts,
        "meta": {
            "n_colors":       n_colors,
            "is_valid":       is_valid,
            "conflict_edges": conflict_edges,
        },
        # Backward-compat flat keys used by runner.py
        "n_colors":       n_colors,
        "n_conflicts":    n_conflicts,
        "is_valid":       is_valid,
        "conflict_edges": conflict_edges,
    }
