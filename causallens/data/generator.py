"""
causallens.data.generator
=========================
Generate synthetic data from Structural Equation Models (SEMs).

Purpose
-------
This component creates datasets with **known ground truth** causal
structure.  This is essential for:

1. Testing discovery algorithms (PC, Granger) against known answers
2. Testing inference algorithms (counterfactuals, ATE) where we know
   the true causal effect
3. Demos and case studies
4. Benchmarking and ablation studies

How it works
------------
A Structural Equation Model defines each variable as a function of its
causal parents plus noise:

    X_i = f(parents(X_i)) + ε_i

This generator supports:
- Linear SEMs:  X_i = Σ(w_ij · parent_j) + ε_i
- Custom SEMs:  User-defined functions per variable
- Preset structures: chain, fork, collider, diamond, etc.

The generator produces both the data AND the ground truth CausalGraph,
so downstream components can validate their results.

Dependencies
------------
- ``causallens.types`` : CausalGraph, Edge, EdgeType
- numpy, pandas

No other CausalLens components are imported.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from causallens.types import CausalGraph, Edge, EdgeType


class SEMGenerator:
    """
    Generate synthetic data from a Structural Equation Model.

    The generator builds data column-by-column in topological order,
    so each variable's parents are already generated when it's computed.

    Parameters
    ----------
    seed : int or None, default None
        Random seed for reproducibility.

    Examples
    --------
    >>> from causallens.data.generator import SEMGenerator
    >>> gen = SEMGenerator(seed=42)
    >>> # Build a custom SEM
    >>> gen.add_variable("age", noise_std=10.0, noise_mean=40.0)
    >>> gen.add_variable("income", parents={"age": 500.0}, noise_std=5000.0)
    >>> gen.add_variable("spending", parents={"income": 0.3}, noise_std=2000.0)
    >>> data, graph = gen.generate(n=1000)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._variables: List[str] = []
        self._parents: Dict[str, Dict[str, float]] = {}      # var → {parent: weight}
        self._noise_std: Dict[str, float] = {}
        self._noise_mean: Dict[str, float] = {}
        self._custom_fn: Dict[str, Callable] = {}             # var → custom function
        self._intercept: Dict[str, float] = {}

    # ── Building the SEM ─────────────────────────────────────────────────

    def add_variable(
        self,
        name: str,
        parents: Optional[Dict[str, float]] = None,
        noise_std: float = 1.0,
        noise_mean: float = 0.0,
        intercept: float = 0.0,
        custom_fn: Optional[Callable] = None,
    ) -> "SEMGenerator":
        """
        Add a variable to the SEM.

        Parameters
        ----------
        name : str
            Variable name (becomes a DataFrame column).
        parents : dict or None
            ``{parent_name: weight}`` for linear relationships.
            None means this is a root node (no parents).
        noise_std : float
            Standard deviation of Gaussian noise added to this variable.
        noise_mean : float
            Mean of Gaussian noise (shifts the distribution).
        intercept : float
            Constant term added to the structural equation.
        custom_fn : callable or None
            If provided, overrides linear equation. Signature:
            ``fn(parent_values: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray``
            where parent_values maps parent names to their generated arrays.

        Returns
        -------
        self (for chaining)
        """
        if name in self._variables:
            raise ValueError(f"Variable '{name}' already exists")

        # Validate parents exist
        if parents:
            for p in parents:
                if p not in self._variables:
                    raise ValueError(
                        f"Parent '{p}' of '{name}' has not been added yet. "
                        f"Add variables in topological order (parents first)."
                    )

        self._variables.append(name)
        self._parents[name] = parents or {}
        self._noise_std[name] = noise_std
        self._noise_mean[name] = noise_mean
        self._intercept[name] = intercept
        self._custom_fn[name] = custom_fn

        return self

    def clear(self) -> "SEMGenerator":
        """Remove all variables. Returns self for chaining."""
        self._variables.clear()
        self._parents.clear()
        self._noise_std.clear()
        self._noise_mean.clear()
        self._custom_fn.clear()
        self._intercept.clear()
        return self

    # ── Generation ───────────────────────────────────────────────────────

    def generate(self, n: int = 1000) -> Tuple[pd.DataFrame, CausalGraph]:
        """
        Generate n samples from the SEM.

        Parameters
        ----------
        n : int
            Number of samples (rows) to generate.

        Returns
        -------
        (data, graph) : tuple[pd.DataFrame, CausalGraph]
            data: Generated dataset with columns = variables.
            graph: Ground truth CausalGraph with directed edges.
        """
        if not self._variables:
            raise RuntimeError("No variables added. Call add_variable() first.")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        rng = np.random.RandomState(self._seed)
        columns: Dict[str, np.ndarray] = {}

        for var in self._variables:
            noise = rng.normal(
                self._noise_mean[var],
                self._noise_std[var],
                n,
            )

            if self._custom_fn.get(var) is not None:
                # Custom function
                parent_values = {p: columns[p] for p in self._parents[var]}
                columns[var] = self._custom_fn[var](parent_values, noise)
            elif not self._parents[var]:
                # Root node
                columns[var] = self._intercept[var] + noise
            else:
                # Linear combination of parents + noise
                value = np.full(n, self._intercept[var], dtype=np.float64)
                for parent_name, weight in self._parents[var].items():
                    value = value + weight * columns[parent_name]
                value = value + noise
                columns[var] = value

        data = pd.DataFrame(columns)
        graph = self._build_graph()

        return data, graph

    def _build_graph(self) -> CausalGraph:
        """Build ground truth CausalGraph from the SEM definition."""
        edges: List[Edge] = []
        for var in self._variables:
            for parent, weight in self._parents[var].items():
                edges.append(Edge(
                    source=parent,
                    target=var,
                    edge_type=EdgeType.DIRECTED,
                    weight=weight,
                ))
        return CausalGraph(
            variables=list(self._variables),
            edges=edges,
            metadata={"source": "SEMGenerator", "type": "ground_truth"},
        )

    # ── Preset structures ────────────────────────────────────────────────

    @classmethod
    def chain(
        cls,
        n_vars: int = 4,
        weights: Optional[List[float]] = None,
        noise_std: float = 0.5,
        seed: Optional[int] = None,
    ) -> "SEMGenerator":
        """
        Preset: chain structure  A → B → C → D → ...

        Parameters
        ----------
        n_vars : int
            Number of variables in the chain (>= 2).
        weights : list[float] or None
            Edge weights. Length must be n_vars - 1. Default: all 0.8.
        noise_std : float
            Noise standard deviation for all variables.
        seed : int or None
            Random seed.

        Returns
        -------
        Configured SEMGenerator (call .generate(n) to produce data).
        """
        if n_vars < 2:
            raise ValueError(f"Chain needs >= 2 variables, got {n_vars}")
        if weights is None:
            weights = [0.8] * (n_vars - 1)
        if len(weights) != n_vars - 1:
            raise ValueError(
                f"Expected {n_vars - 1} weights, got {len(weights)}"
            )

        names = [chr(65 + i) if n_vars <= 26 else f"V{i}" for i in range(n_vars)]
        gen = cls(seed=seed)
        gen.add_variable(names[0], noise_std=1.0)
        for i in range(1, n_vars):
            gen.add_variable(
                names[i],
                parents={names[i - 1]: weights[i - 1]},
                noise_std=noise_std,
            )
        return gen

    @classmethod
    def fork(
        cls,
        n_children: int = 2,
        weights: Optional[List[float]] = None,
        noise_std: float = 0.4,
        seed: Optional[int] = None,
    ) -> "SEMGenerator":
        """
        Preset: fork structure  B ← A → C (→ D → ...)

        One root node A with n_children causal children.
        """
        if n_children < 1:
            raise ValueError(f"Fork needs >= 1 child, got {n_children}")
        if weights is None:
            weights = [0.8] * n_children
        if len(weights) != n_children:
            raise ValueError(
                f"Expected {n_children} weights, got {len(weights)}"
            )

        gen = cls(seed=seed)
        gen.add_variable("A", noise_std=1.0)
        for i in range(n_children):
            name = chr(66 + i)  # B, C, D, ...
            gen.add_variable(
                name,
                parents={"A": weights[i]},
                noise_std=noise_std,
            )
        return gen

    @classmethod
    def collider(
        cls,
        n_parents: int = 2,
        weights: Optional[List[float]] = None,
        noise_std: float = 0.3,
        seed: Optional[int] = None,
    ) -> "SEMGenerator":
        """
        Preset: collider structure  A → C ← B (← D ← ...)

        Multiple independent root nodes causing a single effect variable.
        """
        if n_parents < 2:
            raise ValueError(f"Collider needs >= 2 parents, got {n_parents}")
        if weights is None:
            weights = [0.7] * n_parents
        if len(weights) != n_parents:
            raise ValueError(
                f"Expected {n_parents} weights, got {len(weights)}"
            )

        gen = cls(seed=seed)
        parent_names = []
        for i in range(n_parents):
            name = chr(65 + i)  # A, B, C, ...
            gen.add_variable(name, noise_std=1.0)
            parent_names.append(name)

        effect_name = chr(65 + n_parents)  # next letter after parents
        parent_dict = {p: w for p, w in zip(parent_names, weights)}
        gen.add_variable(effect_name, parents=parent_dict, noise_std=noise_std)

        return gen

    @classmethod
    def diamond(
        cls,
        seed: Optional[int] = None,
    ) -> "SEMGenerator":
        """
        Preset: diamond structure

            A → B → D
            A → C → D
        """
        gen = cls(seed=seed)
        gen.add_variable("A", noise_std=1.0)
        gen.add_variable("B", parents={"A": 0.8}, noise_std=0.4)
        gen.add_variable("C", parents={"A": 0.7}, noise_std=0.4)
        gen.add_variable("D", parents={"B": 0.5, "C": 0.5}, noise_std=0.3)
        return gen

    @classmethod
    def full_demo(
        cls,
        seed: Optional[int] = None,
    ) -> "SEMGenerator":
        """
        Preset: realistic 6-variable business scenario.

            ad_spend → website_traffic → conversions → revenue
            season → website_traffic
            season → revenue
            ad_spend → brand_awareness → conversions

        Includes non-trivial structure with multiple paths and a
        confounder (season).
        """
        gen = cls(seed=seed)
        gen.add_variable("ad_spend", noise_std=1000.0, noise_mean=5000.0)
        gen.add_variable("season", noise_std=1.0, noise_mean=0.0)
        gen.add_variable(
            "brand_awareness",
            parents={"ad_spend": 0.01},
            noise_std=5.0,
            noise_mean=50.0,
        )
        gen.add_variable(
            "website_traffic",
            parents={"ad_spend": 0.5, "season": 500.0},
            noise_std=200.0,
            noise_mean=1000.0,
        )
        gen.add_variable(
            "conversions",
            parents={"website_traffic": 0.05, "brand_awareness": 2.0},
            noise_std=10.0,
        )
        gen.add_variable(
            "revenue",
            parents={"conversions": 100.0, "season": 2000.0},
            noise_std=5000.0,
            noise_mean=50000.0,
        )
        return gen

    @classmethod
    def nonlinear_demo(
        cls,
        seed: Optional[int] = None,
    ) -> "SEMGenerator":
        """
        Preset: nonlinear relationships using custom functions.

            X → Y (quadratic)
            Y → Z (interaction with threshold)
        """
        gen = cls(seed=seed)
        gen.add_variable("X", noise_std=1.0)
        gen.add_variable(
            "Y",
            parents={"X": 1.0},  # weight not used, just marks the edge
            custom_fn=lambda pv, noise: 0.5 * pv["X"] ** 2 + noise,
            noise_std=0.5,
        )
        gen.add_variable(
            "Z",
            parents={"Y": 1.0},
            custom_fn=lambda pv, noise: np.where(
                pv["Y"] > 0,
                0.8 * pv["Y"],
                0.2 * pv["Y"],
            ) + noise,
            noise_std=0.3,
        )
        return gen

    # ── Utility ──────────────────────────────────────────────────────────

    @property
    def variables(self) -> List[str]:
        """List of variable names in topological order."""
        return list(self._variables)

    @property
    def n_variables(self) -> int:
        return len(self._variables)

    def get_true_effect(self, cause: str, effect: str) -> Optional[float]:
        """
        Return the direct causal weight from cause to effect.
        Returns None if no direct edge exists.
        """
        if effect not in self._parents:
            return None
        return self._parents[effect].get(cause, None)

    def describe(self) -> str:
        """Return a human-readable description of the SEM."""
        lines = ["SEM Structure:", "=" * 40]
        for var in self._variables:
            parents = self._parents[var]
            if not parents:
                lines.append(f"  {var} = N({self._noise_mean[var]}, "
                             f"{self._noise_std[var]}²)")
            elif self._custom_fn.get(var) is not None:
                parent_str = ", ".join(parents.keys())
                lines.append(f"  {var} = f({parent_str}) + ε  [custom]")
            else:
                terms = [f"{w:.2f}·{p}" for p, w in parents.items()]
                eq = " + ".join(terms)
                if self._intercept[var] != 0:
                    eq = f"{self._intercept[var]:.2f} + " + eq
                lines.append(f"  {var} = {eq} + ε")
        lines.append("=" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SEMGenerator(variables={self.n_variables}, "
            f"edges={sum(len(p) for p in self._parents.values())})"
        )
