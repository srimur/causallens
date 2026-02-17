"""
causallens.discovery.pc
=======================
The Peter-Clark (PC) algorithm for causal structure discovery from
observational data.

Algorithm overview
------------------
1. **Skeleton discovery** — start with a complete undirected graph.
   Remove the edge X—Y whenever a conditioning set S ⊆ adj(X)\\{Y}
   is found such that X ⊥ Y | S  (conditional independence).

2. **Edge orientation** — orient edges by detecting *colliders*
   (v-structures): for every unshielded triple X—Z—Y where X and Y
   are non-adjacent, if Z is NOT in sep(X,Y), orient as X→Z←Y.
   Then apply Meek's orientation rules to propagate directions.

Assumptions (standard PC)
-------------------------
1. Causal Markov condition
2. Faithfulness
3. Causal sufficiency (no latent common causes)
4. Acyclicity

References
----------
- Spirtes, Glymour, Scheines. *Causation, Prediction, and Search.* (2000)
- Meek. *Causal discovery and inference: concepts and recent methodological
  advances.* (1995)

Dependencies
------------
- ``causallens.types`` : CausalGraph, Edge, EdgeType  (shared interface)
- ``causallens.discovery.ci_tests`` : partial_correlation  (CI testing)
- numpy, pandas (data handling)

No other CausalLens components are imported.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from causallens.types import CausalGraph, Edge, EdgeType
from causallens.discovery.ci_tests import partial_correlation, _has_native_backend


class PCAlgorithm:
    """
    PC algorithm for causal discovery on tabular (i.i.d.) data.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for conditional independence tests.
        Lower values produce sparser graphs (fewer edges).
    max_cond_set_size : int or None, default None
        Maximum conditioning set size to consider.  ``None`` means
        unlimited (standard PC).  Setting a value like 3 or 4 makes
        the algorithm faster at the cost of potentially missing some
        independencies that require large conditioning sets.
    verbose : bool, default False
        Print progress information during discovery.

    Examples
    --------
    >>> import pandas as pd
    >>> from causallens.discovery.pc import PCAlgorithm
    >>> data = pd.read_csv("my_data.csv")
    >>> pc = PCAlgorithm(alpha=0.05)
    >>> graph = pc.fit(data)
    >>> print(graph)
    >>> graph.parents_of("sales")
    ['ad_spend', 'season']
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_cond_set_size is not None and max_cond_set_size < 0:
            raise ValueError("max_cond_set_size must be non-negative")

        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.verbose = verbose

        # Populated after fit()
        self._graph: Optional[CausalGraph] = None
        self._sep_sets: Dict[FrozenSet[str], Set[str]] = {}
        self._n_tests: int = 0
        self._runtime: float = 0.0

    # ── Public API ───────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """
        Discover causal structure from observational data.

        Parameters
        ----------
        data : pd.DataFrame
            Each column is a variable, each row is an observation.
            All columns must be numeric.  Missing values are NOT allowed.

        Returns
        -------
        CausalGraph
            Discovered causal graph with directed and possibly undirected edges.

        Raises
        ------
        ValueError
            If data contains non-numeric columns or NaN values.
        """
        self._validate_input(data)
        t0 = time.time()

        variables = list(data.columns)
        var_idx = {v: i for i, v in enumerate(variables)}
        data_np = np.ascontiguousarray(data.values, dtype=np.float64)

        if self.verbose:
            backend = "C++" if _has_native_backend() else "NumPy"
            print(f"[PC] Starting discovery on {len(variables)} variables, "
                  f"{len(data)} observations. CI backend: {backend}")

        # Step 1: Discover skeleton
        adjacency, sep_sets = self._discover_skeleton(
            data_np, variables, var_idx
        )

        # Step 2: Orient edges (v-structures + Meek rules)
        directed_edges, undirected_edges = self._orient_edges(
            variables, adjacency, sep_sets
        )

        # Build CausalGraph
        edges: List[Edge] = []
        for src, tgt, pval in directed_edges:
            edges.append(Edge(source=src, target=tgt,
                              edge_type=EdgeType.DIRECTED, p_value=pval))
        for v1, v2, pval in undirected_edges:
            edges.append(Edge(source=v1, target=v2,
                              edge_type=EdgeType.UNDIRECTED, p_value=pval))

        self._runtime = time.time() - t0
        self._sep_sets = sep_sets

        self._graph = CausalGraph(
            variables=variables,
            edges=edges,
            metadata={
                "algorithm": "PC",
                "alpha": self.alpha,
                "max_cond_set_size": self.max_cond_set_size,
                "n_ci_tests": self._n_tests,
                "runtime_seconds": round(self._runtime, 4),
                "ci_backend": "C++" if _has_native_backend() else "NumPy",
                "n_observations": len(data),
            },
        )

        if self.verbose:
            print(f"[PC] Done in {self._runtime:.2f}s — "
                  f"{self._n_tests} CI tests — "
                  f"{self._graph.n_directed} directed + "
                  f"{self._graph.n_undirected} undirected edges.")

        return self._graph

    @property
    def graph(self) -> CausalGraph:
        """Access the discovered graph. Raises if :meth:`fit` not called."""
        if self._graph is None:
            raise RuntimeError("Call .fit(data) before accessing .graph")
        return self._graph

    @property
    def separation_sets(self) -> Dict[FrozenSet[str], Set[str]]:
        """
        Return the separation sets found during skeleton discovery.

        Keys are ``frozenset({X, Y})``, values are the conditioning
        set that rendered X and Y independent.
        """
        return dict(self._sep_sets)

    # ── Step 1: Skeleton discovery ───────────────────────────────────────

    def _discover_skeleton(
        self,
        data: np.ndarray,
        variables: List[str],
        var_idx: Dict[str, int],
    ) -> Tuple[Dict[str, Set[str]], Dict[FrozenSet[str], Set[str]]]:
        """
        Remove edges where conditional independence is detected.

        Returns
        -------
        adjacency : dict[str, set[str]]
            Undirected adjacency structure after edge removal.
        sep_sets : dict[frozenset[str], set[str]]
            The conditioning set that separated each removed pair.
        """
        n_vars = len(variables)

        # Start fully connected
        adjacency: Dict[str, Set[str]] = {
            v: set(variables) - {v} for v in variables
        }
        sep_sets: Dict[FrozenSet[str], Set[str]] = {}

        # Track the p-value of the *strongest* remaining dependence per edge
        # (used later to annotate edges)
        edge_pvalues: Dict[FrozenSet[str], float] = {}

        depth = 0
        max_depth = (self.max_cond_set_size
                     if self.max_cond_set_size is not None
                     else n_vars - 2)

        while depth <= max_depth:
            any_removed = False

            # Iterate over a stable snapshot of current edges
            edge_list = [
                (x, y)
                for x in variables
                for y in sorted(adjacency[x])
                if x < y  # each unordered pair once
            ]

            for x, y in edge_list:
                # Check if edge still exists (may have been removed)
                if y not in adjacency[x]:
                    continue

                # Possible conditioning sets from neighbors of X, excluding Y
                possible_z = sorted(adjacency[x] - {y})

                if len(possible_z) < depth:
                    continue

                separated = False
                for z_tuple in combinations(possible_z, depth):
                    z_set = list(z_tuple)
                    z_idx = [var_idx[v] for v in z_set]

                    corr, pval = partial_correlation(
                        data, var_idx[x], var_idx[y], z_idx
                    )
                    self._n_tests += 1

                    if pval > self.alpha:
                        # Conditionally independent → remove edge
                        adjacency[x].discard(y)
                        adjacency[y].discard(x)
                        sep_sets[frozenset({x, y})] = set(z_set)
                        separated = True
                        any_removed = True

                        if self.verbose and depth > 0:
                            print(f"  [depth={depth}] Removed {x} — {y} "
                                  f"| {z_set}  (p={pval:.4f})")
                        break

                if not separated and y in adjacency[x]:
                    # Track the p-value for the surviving edge
                    # (use the minimum p-value seen = strongest dependence)
                    key = frozenset({x, y})
                    # We want the last (largest conditioning set) p-value
                    # as a measure of remaining dependence
                    if possible_z and depth > 0:
                        # Recompute with empty Z as baseline
                        pass
                    # Just store a representative p-value
                    if key not in edge_pvalues:
                        r, p = partial_correlation(
                            data, var_idx[x], var_idx[y], []
                        )
                        edge_pvalues[key] = p

            # Also try from Y's neighbors perspective (PC is order-dependent;
            # checking both directions makes it more robust)
            for x, y in edge_list:
                if y not in adjacency[x]:
                    continue

                possible_z = sorted(adjacency[y] - {x})
                if len(possible_z) < depth:
                    continue

                for z_tuple in combinations(possible_z, depth):
                    z_set = list(z_tuple)
                    z_idx = [var_idx[v] for v in z_set]

                    corr, pval = partial_correlation(
                        data, var_idx[x], var_idx[y], z_idx
                    )
                    self._n_tests += 1

                    if pval > self.alpha:
                        adjacency[x].discard(y)
                        adjacency[y].discard(x)
                        sep_sets[frozenset({x, y})] = set(z_set)
                        any_removed = True

                        if self.verbose and depth > 0:
                            print(f"  [depth={depth}] Removed {x} — {y} "
                                  f"| {z_set}  (p={pval:.4f})")
                        break

            depth += 1

            # If no edges were removed at this depth and no pair has enough
            # neighbors to test at depth+1, we can stop early
            if not any_removed:
                max_neighbors = max(
                    (len(adjacency[v]) for v in variables), default=0
                )
                if max_neighbors < depth:
                    break

        self._edge_pvalues = edge_pvalues
        return adjacency, sep_sets

    # ── Step 2: Edge orientation ─────────────────────────────────────────

    def _orient_edges(
        self,
        variables: List[str],
        adjacency: Dict[str, Set[str]],
        sep_sets: Dict[FrozenSet[str], Set[str]],
    ) -> Tuple[List[Tuple[str, str, Optional[float]]],
               List[Tuple[str, str, Optional[float]]]]:
        """
        Orient edges in the skeleton.

        Phase A: Detect v-structures (colliders).
        Phase B: Apply Meek's 3 orientation rules until no more changes.

        Returns
        -------
        directed : list of (source, target, p_value)
        undirected : list of (var1, var2, p_value)
        """
        # Internal directed adjacency: directed[x] = set of y where x→y
        directed_to: Dict[str, Set[str]] = {v: set() for v in variables}
        # Track which edges are still undirected
        undirected_pairs: Set[FrozenSet[str]] = set()

        for v in variables:
            for u in adjacency[v]:
                if v < u:
                    undirected_pairs.add(frozenset({v, u}))

        # ── Phase A: V-structure detection ───────────────────────────────

        for z in variables:
            neighbors_z = sorted(adjacency[z])
            for x, y in combinations(neighbors_z, 2):
                # x—z—y is an unshielded triple if x and y are NOT adjacent
                if y in adjacency[x]:
                    continue  # shielded — skip

                # Check if z is NOT in sep(x, y)
                sep = sep_sets.get(frozenset({x, y}), None)
                if sep is not None and z not in sep:
                    # Orient as x → z ← y (collider)
                    directed_to[x].add(z)
                    directed_to[y].add(z)
                    undirected_pairs.discard(frozenset({x, z}))
                    undirected_pairs.discard(frozenset({y, z}))

                    if self.verbose:
                        print(f"  [v-structure] {x} → {z} ← {y}")

        # ── Phase B: Meek's orientation rules ────────────────────────────

        changed = True
        while changed:
            changed = False

            for pair in list(undirected_pairs):
                v1, v2 = sorted(pair)

                # Rule 1: If a→b—c and a and c are not adjacent, orient b→c
                # (to avoid creating a new v-structure)
                oriented = self._meek_rule_1(
                    v1, v2, adjacency, directed_to, undirected_pairs
                )
                if not oriented:
                    oriented = self._meek_rule_1(
                        v2, v1, adjacency, directed_to, undirected_pairs
                    )
                if oriented:
                    changed = True
                    continue

                # Rule 2: If a→c→b and a—b, orient a→b (to avoid cycle)
                oriented = self._meek_rule_2(
                    v1, v2, directed_to, undirected_pairs
                )
                if not oriented:
                    oriented = self._meek_rule_2(
                        v2, v1, directed_to, undirected_pairs
                    )
                if oriented:
                    changed = True
                    continue

                # Rule 3: If a—c→b, a—d→b, and a—b, c and d not adjacent,
                # orient a→b
                oriented = self._meek_rule_3(
                    v1, v2, adjacency, directed_to, undirected_pairs
                )
                if not oriented:
                    oriented = self._meek_rule_3(
                        v2, v1, adjacency, directed_to, undirected_pairs
                    )
                if oriented:
                    changed = True

        # ── Assemble results ─────────────────────────────────────────────

        def _pval(a: str, b: str) -> Optional[float]:
            return self._edge_pvalues.get(frozenset({a, b}))

        directed_list = []
        for src in variables:
            for tgt in sorted(directed_to[src]):
                directed_list.append((src, tgt, _pval(src, tgt)))

        undirected_list = []
        for pair in sorted(undirected_pairs, key=lambda s: tuple(sorted(s))):
            v1, v2 = sorted(pair)
            undirected_list.append((v1, v2, _pval(v1, v2)))

        return directed_list, undirected_list

    # ── Meek orientation rules ───────────────────────────────────────────

    @staticmethod
    def _meek_rule_1(
        a: str, b: str,
        adjacency: Dict[str, Set[str]],
        directed_to: Dict[str, Set[str]],
        undirected: Set[FrozenSet[str]],
    ) -> bool:
        """
        Rule 1: if c→a — b and c and b are NOT adjacent, orient a→b.
        Check if *a* has a directed parent c that is not adjacent to *b*.
        """
        for c in list(directed_to.keys()):
            if a in directed_to[c]:  # c → a exists
                if b not in adjacency.get(c, set()):  # c and b not adjacent
                    # Orient a → b
                    directed_to[a].add(b)
                    undirected.discard(frozenset({a, b}))
                    return True
        return False

    @staticmethod
    def _meek_rule_2(
        a: str, b: str,
        directed_to: Dict[str, Set[str]],
        undirected: Set[FrozenSet[str]],
    ) -> bool:
        """
        Rule 2: if a→c→b and a—b, orient a→b (to avoid creating a cycle).
        """
        for c in directed_to[a]:       # a → c
            if b in directed_to[c]:    # c → b
                if frozenset({a, b}) in undirected:
                    directed_to[a].add(b)
                    undirected.discard(frozenset({a, b}))
                    return True
        return False

    @staticmethod
    def _meek_rule_3(
        a: str, b: str,
        adjacency: Dict[str, Set[str]],
        directed_to: Dict[str, Set[str]],
        undirected: Set[FrozenSet[str]],
    ) -> bool:
        """
        Rule 3: if a—c→b and a—d→b and c,d not adjacent, orient a→b.
        """
        # Find all c where a—c (undirected) and c→b (directed)
        candidates = []
        for c in adjacency.get(a, set()):
            if (frozenset({a, c}) in undirected
                    and b in directed_to.get(c, set())):
                candidates.append(c)

        # Check if any two candidates c, d are non-adjacent
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c, d = candidates[i], candidates[j]
                if d not in adjacency.get(c, set()):
                    directed_to[a].add(b)
                    undirected.discard(frozenset({a, b}))
                    return True
        return False

    # ── Input validation ─────────────────────────────────────────────────

    @staticmethod
    def _validate_input(data: pd.DataFrame) -> None:
        """Raise ValueError if data is unsuitable for PC algorithm."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(data).__name__}")

        if data.empty:
            raise ValueError("Data is empty")

        if data.shape[1] < 2:
            raise ValueError("Need at least 2 variables for causal discovery")

        # Check numeric
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Non-numeric columns found: {non_numeric}. "
                f"Encode categorical variables before calling fit()."
            )

        # Check NaN
        nan_cols = data.columns[data.isna().any()].tolist()
        if nan_cols:
            raise ValueError(
                f"NaN values found in columns: {nan_cols}. "
                f"Drop or impute missing values before calling fit()."
            )

        # Warn about low sample size
        n_vars = data.shape[1]
        if data.shape[0] < 5 * n_vars:
            import warnings
            warnings.warn(
                f"Sample size ({data.shape[0]}) is small relative to the "
                f"number of variables ({n_vars}). PC algorithm results "
                f"may be unreliable. Aim for at least 10× variables.",
                UserWarning,
                stacklevel=3,
            )
