"""
causallens.types
================
Shared data structures consumed and produced by all CausalLens components.

Design principles
-----------------
- These are pure data holders (dataclasses). Zero logic lives here.
- Discovery components PRODUCE a CausalGraph.
- Inference components CONSUME a CausalGraph.
- Visualization components RENDER a CausalGraph.
- Every component can be tested independently because the only coupling
  between them is through these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Enums ────────────────────────────────────────────────────────────────────


class EdgeType(Enum):
    """Direction status of an edge in a causal graph."""

    DIRECTED = "directed"       # A → B
    UNDIRECTED = "undirected"   # A — B  (direction unknown yet)
    BIDIRECTED = "bidirected"   # A ↔ B  (latent common cause)


# ── Edge ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Edge:
    """A single edge in a causal graph."""

    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECTED
    weight: Optional[float] = None
    p_value: Optional[float] = None

    def __repr__(self) -> str:
        sym = {"directed": "→", "undirected": "—", "bidirected": "↔"}[
            self.edge_type.value
        ]
        extra = f"  p={self.p_value:.4f}" if self.p_value is not None else ""
        return f"Edge({self.source} {sym} {self.target}{extra})"


# ── CausalGraph ──────────────────────────────────────────────────────────────


@dataclass
class CausalGraph:
    """
    The central data structure that flows between components.

    Parameters
    ----------
    variables : list[str]
        Ordered list of variable (column) names.
    edges : list[Edge]
        All edges discovered or provided.
    metadata : dict
        Arbitrary info (algorithm name, runtime, hyperparams, …).
    """

    variables: List[str]
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Read helpers (used by inference, viz, query) ─────────────────────

    def parents_of(self, variable: str) -> List[str]:
        """Direct causal parents via directed edges into *variable*."""
        return sorted(
            {e.source for e in self.edges
             if e.target == variable and e.edge_type == EdgeType.DIRECTED}
        )

    def children_of(self, variable: str) -> List[str]:
        """Direct causal children via directed edges out of *variable*."""
        return sorted(
            {e.target for e in self.edges
             if e.source == variable and e.edge_type == EdgeType.DIRECTED}
        )

    def neighbors_of(self, variable: str) -> List[str]:
        """All adjacent variables regardless of edge type or direction."""
        nbrs: set = set()
        for e in self.edges:
            if e.source == variable:
                nbrs.add(e.target)
            if e.target == variable:
                nbrs.add(e.source)
        return sorted(nbrs)

    def has_edge(self, source: str, target: str,
                 directed_only: bool = False) -> bool:
        """Check if an edge exists between *source* and *target*."""
        for e in self.edges:
            if directed_only:
                if (e.source == source and e.target == target
                        and e.edge_type == EdgeType.DIRECTED):
                    return True
            else:
                if {e.source, e.target} == {source, target}:
                    return True
        return False

    def adjacency_dict(self) -> Dict[str, List[str]]:
        """Return ``{child: [parent, …]}`` for directed edges only."""
        result: Dict[str, List[str]] = {v: [] for v in self.variables}
        for e in self.edges:
            if e.edge_type == EdgeType.DIRECTED:
                result[e.target].append(e.source)
        return result

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return ``(matrix, var_names)`` where ``matrix[i][j] == 1``
        means ``variables[i] → variables[j]``.
        """
        n = len(self.variables)
        idx = {v: i for i, v in enumerate(self.variables)}
        mat = np.zeros((n, n), dtype=np.int8)
        for e in self.edges:
            if e.edge_type == EdgeType.DIRECTED:
                mat[idx[e.source], idx[e.target]] = 1
        return mat, list(self.variables)

    def topological_sort(self) -> List[str]:
        """
        Return variables in causal order (parents before children).

        Raises ``ValueError`` if the directed sub-graph contains a cycle.
        Only considers directed edges.
        """
        adj = self.adjacency_dict()
        in_deg = {v: len(adj[v]) for v in self.variables}
        queue = [v for v in self.variables if in_deg[v] == 0]
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.children_of(node):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)

        if len(order) != len(self.variables):
            raise ValueError(
                "Directed sub-graph contains a cycle — "
                "topological sort is impossible."
            )
        return order

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_directed(self) -> int:
        return sum(1 for e in self.edges if e.edge_type == EdgeType.DIRECTED)

    @property
    def n_undirected(self) -> int:
        return sum(1 for e in self.edges if e.edge_type == EdgeType.UNDIRECTED)

    def __repr__(self) -> str:
        return (
            f"CausalGraph(variables={len(self.variables)}, "
            f"edges={self.n_directed} directed + "
            f"{self.n_undirected} undirected)"
        )
