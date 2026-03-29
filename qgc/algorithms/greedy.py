"""
qgc/algorithms/greedy.py
=========================
Classical greedy graph coloring — the baseline every quantum method is compared to.

Algorithm (Oh et al. 2019, §4.1)
---------------------------------
The greedy algorithm visits nodes in decreasing degree order (largest-first)
and assigns each node the lowest-numbered color not already used by any
of its already-colored neighbours.

Properties:
  - Runs in O(n + m) time  (n = nodes, m = edges)
  - Always finds a valid proper coloring
  - May use more colors than the chromatic number χ(G)
  - Upper bound: greedy uses at most Δ+1 colors (Δ = max degree)

This serves as the classical reference point for QAOA and Qudit methods.
"""

import networkx as nx
from qgc.core.coloring import count_conflicts, coloring_summary


def greedy_coloring(graph: nx.Graph, strategy: str = "largest_first") -> dict:
    """
    Classic greedy graph coloring via NetworkX.

    Args:
        graph:    NetworkX graph.
        strategy: Node ordering strategy passed to nx.coloring.greedy_color.
                  Options: 'largest_first', 'random_sequential',
                           'smallest_last', 'saturation_largest_first', etc.

    Returns:
        result dict:
          "coloring"  – list of int, node-color assignment
          "conflicts" – int, always 0 for greedy
          "meta"      – {"n_colors", "is_valid", "conflict_edges"}
          (plus backward-compat flat keys n_colors, n_conflicts,
           is_valid, conflict_edges)
    """
    color_map = nx.coloring.greedy_color(graph, strategy=strategy)
    coloring  = [int(color_map[i]) for i in range(graph.number_of_nodes())]
    return coloring_summary(coloring, graph)
