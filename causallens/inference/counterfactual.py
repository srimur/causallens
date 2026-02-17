"""
causallens.inference.counterfactual
====================================
Counterfactual reasoning engine — the core of CausalLens.

What it does
------------
Given a CausalGraph and observational data, this engine answers
"what if" questions:

    "What would Y have been if X had been set to x?"

It does this by:
1. Learning structural equations for each variable from data
   (each variable = f(its causal parents) + noise)
2. For a counterfactual query, overriding the intervention variable
3. Propagating the change through the causal graph in topological order
4. Reading off the counterfactual value of the target variable

This implements Pearl's 3-step counterfactual procedure:
    Step 1 (Abduction):    Infer exogenous noise from observed data
    Step 2 (Action):       Intervene — set X = x (do-operator)
    Step 3 (Prediction):   Propagate through modified SCM

Key concepts
------------
- **Intervention** (do-operator): Forcibly setting a variable to a value,
  breaking its dependence on parents.  do(X=5) means "set X to 5
  regardless of what its parents would have produced."

- **Counterfactual**: "Given what actually happened, what WOULD have
  happened if we had intervened?"  This requires abduction (learning
  the noise terms) to keep the non-intervened variables consistent
  with the observed data.

- **ATE** (Average Treatment Effect): The average causal effect of
  changing X from one value to another across the population.
  ATE = E[Y | do(X=1)] - E[Y | do(X=0)]

Dependencies
------------
- ``causallens.types`` : CausalGraph, Edge, EdgeType
- numpy, pandas, scikit-learn (RandomForestRegressor, LinearRegression)

No other CausalLens components are imported.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from causallens.types import CausalGraph, EdgeType


class CounterfactualEngine:
    """
    Counterfactual inference engine.

    Learns structural equations from data + causal graph, then
    computes counterfactuals and treatment effects.

    Parameters
    ----------
    model_type : str, default "linear"
        Type of model to learn for each structural equation.
        - "linear" : LinearRegression (fast, interpretable)
        - "forest" : RandomForestRegressor (handles nonlinearity)
    verbose : bool, default False
        Print progress during fitting.

    Examples
    --------
    >>> from causallens.inference.counterfactual import CounterfactualEngine
    >>> engine = CounterfactualEngine(model_type="linear")
    >>> engine.fit(data, graph)
    >>> result = engine.counterfactual(
    ...     observation=data.iloc[0],
    ...     intervention={"ad_spend": 10000},
    ...     target="revenue",
    ... )
    >>> print(result)
    """

    def __init__(
        self,
        model_type: str = "linear",
        verbose: bool = False,
    ) -> None:
        if model_type not in ("linear", "forest"):
            raise ValueError(
                f"model_type must be 'linear' or 'forest', got '{model_type}'"
            )
        self.model_type = model_type
        self.verbose = verbose

        # Populated after fit()
        self._graph: Optional[CausalGraph] = None
        self._models: Dict[str, Any] = {}         # var → fitted sklearn model
        self._r2_scores: Dict[str, float] = {}    # var → R² score
        self._topo_order: List[str] = []           # topological order
        self._data: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False
        self._runtime: float = 0.0

    # ── Public API: Fitting ──────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        graph: CausalGraph,
    ) -> "CounterfactualEngine":
        """
        Learn structural equations from data and causal graph.

        For each non-root variable, fits a regression model:
            variable = f(parents) + ε

        Parameters
        ----------
        data : pd.DataFrame
            Observational data. Columns must match graph.variables.
        graph : CausalGraph
            The causal graph (from discovery or ground truth).
            Must have at least some directed edges.

        Returns
        -------
        self
        """
        self._validate_fit_input(data, graph)
        t0 = time.time()

        self._graph = graph
        self._data = data.copy()
        self._topo_order = graph.topological_sort()

        if self.verbose:
            print(f"[Counterfactual] Fitting structural equations "
                  f"({self.model_type}) for {graph.n_variables} variables...")

        for var in self._topo_order:
            parents = graph.parents_of(var)

            if not parents:
                # Root node — no model needed, just store distribution info
                self._models[var] = None
                self._r2_scores[var] = None
                if self.verbose:
                    print(f"  {var}: root node (no parents)")
                continue

            X = data[parents].values
            y = data[var].values

            if self.model_type == "linear":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )

            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            self._models[var] = model
            self._r2_scores[var] = round(r2, 4)

            if self.verbose:
                parent_str = ", ".join(parents)
                print(f"  {var} = f({parent_str})  R²={r2:.4f}")

        self._is_fitted = True
        self._runtime = time.time() - t0

        if self.verbose:
            print(f"[Counterfactual] Fitted in {self._runtime:.3f}s")

        return self

    # ── Public API: Counterfactual Queries ────────────────────────────────

    def counterfactual(
        self,
        observation: Union[pd.Series, Dict[str, float]],
        intervention: Dict[str, float],
        target: str,
    ) -> Dict[str, Any]:
        """
        Compute a counterfactual: "Given this observation, what would
        *target* have been if we had set *intervention*?"

        Implements Pearl's 3-step procedure:
        1. Abduction:   Compute noise terms from observation
        2. Action:      Apply do(intervention)
        3. Prediction:  Propagate through causal graph

        Parameters
        ----------
        observation : pd.Series or dict
            The factual observation (actual values of all variables).
        intervention : dict
            ``{variable_name: new_value}`` — the "do" operation.
        target : str
            The variable whose counterfactual value we want.

        Returns
        -------
        dict with keys:
            - target : str
            - factual_value : float
            - counterfactual_value : float
            - effect : float (counterfactual - factual)
            - relative_effect : float (percentage change)
            - intervention : dict
            - all_values : dict (all counterfactual variable values)
        """
        self._check_fitted()

        if isinstance(observation, dict):
            observation = pd.Series(observation)

        # Validate
        for var in self._graph.variables:
            if var not in observation.index:
                raise ValueError(f"Observation missing variable '{var}'")
        for var in intervention:
            if var not in self._graph.variables:
                raise ValueError(
                    f"Intervention variable '{var}' not in graph"
                )
        if target not in self._graph.variables:
            raise ValueError(f"Target '{target}' not in graph")

        # Step 1: Abduction — compute noise terms for each variable
        noise = self._abduction(observation)

        # Step 2 & 3: Action + Prediction — propagate with intervention
        cf_values = self._predict_counterfactual(observation, intervention, noise)

        factual = float(observation[target])
        counterfactual_val = float(cf_values[target])
        effect = counterfactual_val - factual

        return {
            "target": target,
            "factual_value": factual,
            "counterfactual_value": counterfactual_val,
            "effect": effect,
            "relative_effect": (effect / abs(factual) * 100)
            if abs(factual) > 1e-10
            else float("inf") if effect != 0 else 0.0,
            "intervention": dict(intervention),
            "all_values": cf_values,
        }

    def counterfactual_batch(
        self,
        data: pd.DataFrame,
        intervention: Dict[str, float],
        target: str,
    ) -> pd.DataFrame:
        """
        Compute counterfactuals for every row in *data*.

        Parameters
        ----------
        data : pd.DataFrame
            Observations (each row is a separate factual scenario).
        intervention : dict
            The "do" operation applied to every row.
        target : str
            Target variable.

        Returns
        -------
        pd.DataFrame with columns:
            - factual : original target values
            - counterfactual : counterfactual target values
            - effect : difference
            - relative_effect : percentage change
        """
        self._check_fitted()

        results = []
        for idx, row in data.iterrows():
            r = self.counterfactual(row, intervention, target)
            results.append({
                "factual": r["factual_value"],
                "counterfactual": r["counterfactual_value"],
                "effect": r["effect"],
                "relative_effect": r["relative_effect"],
            })

        return pd.DataFrame(results, index=data.index)

    # ── Public API: Treatment Effects ────────────────────────────────────

    def ate(
        self,
        data: pd.DataFrame,
        treatment: str,
        target: str,
        treatment_value: float = 1.0,
        control_value: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute Average Treatment Effect (ATE).

        ATE = E[Y | do(T=treatment_value)] - E[Y | do(T=control_value)]

        Parameters
        ----------
        data : pd.DataFrame
            Observational data.
        treatment : str
            Treatment variable name.
        target : str
            Outcome variable name.
        treatment_value : float
            Value of treatment in the "treated" group.
        control_value : float
            Value of treatment in the "control" group.

        Returns
        -------
        dict with keys:
            - ate : float (average treatment effect)
            - treated_mean : float
            - control_mean : float
            - n_samples : int
            - individual_effects : np.ndarray
        """
        self._check_fitted()

        treated_results = self.counterfactual_batch(
            data,
            intervention={treatment: treatment_value},
            target=target,
        )
        control_results = self.counterfactual_batch(
            data,
            intervention={treatment: control_value},
            target=target,
        )

        treated_mean = treated_results["counterfactual"].mean()
        control_mean = control_results["counterfactual"].mean()
        individual_effects = (
            treated_results["counterfactual"].values
            - control_results["counterfactual"].values
        )

        return {
            "ate": float(treated_mean - control_mean),
            "treated_mean": float(treated_mean),
            "control_mean": float(control_mean),
            "n_samples": len(data),
            "individual_effects": individual_effects,
            "std_effect": float(individual_effects.std()),
        }

    # ── Public API: Inspection ───────────────────────────────────────────

    @property
    def r2_scores(self) -> Dict[str, Optional[float]]:
        """R² scores for each learned structural equation."""
        self._check_fitted()
        return dict(self._r2_scores)

    @property
    def graph(self) -> CausalGraph:
        self._check_fitted()
        return self._graph

    def get_model(self, variable: str) -> Any:
        """Return the fitted sklearn model for *variable*."""
        self._check_fitted()
        if variable not in self._models:
            raise ValueError(f"'{variable}' not in graph")
        return self._models[variable]

    def get_coefficients(self, variable: str) -> Optional[Dict[str, float]]:
        """
        Return learned causal coefficients for *variable* (linear only).

        Returns None for root nodes or forest models.
        """
        self._check_fitted()
        model = self._models.get(variable)
        if model is None:
            return None
        if not isinstance(model, LinearRegression):
            return None
        parents = self._graph.parents_of(variable)
        return dict(zip(parents, model.coef_))

    # ── Internal: Pearl's 3-step procedure ───────────────────────────────

    def _abduction(
        self,
        observation: pd.Series,
    ) -> Dict[str, float]:
        """
        Step 1: Abduction — infer exogenous noise from observed values.

        For each variable with parents:
            noise_i = observed_i - f(observed_parents_i)

        For root nodes:
            noise_i = observed_i (the value IS the noise)
        """
        noise = {}
        for var in self._topo_order:
            model = self._models[var]
            if model is None:
                # Root node
                noise[var] = float(observation[var])
            else:
                parents = self._graph.parents_of(var)
                parent_vals = np.array(
                    [[float(observation[p]) for p in parents]]
                )
                predicted = model.predict(parent_vals)[0]
                noise[var] = float(observation[var]) - predicted
        return noise

    def _predict_counterfactual(
        self,
        observation: pd.Series,
        intervention: Dict[str, float],
        noise: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Steps 2 & 3: Action + Prediction.

        Propagate through the causal graph in topological order.
        Intervention variables are set to their do-values.
        Other variables are computed from their parents + abducted noise.
        """
        cf_values: Dict[str, float] = {}

        for var in self._topo_order:
            if var in intervention:
                # do(var = value) — override, ignore parents and noise
                cf_values[var] = intervention[var]
            else:
                model = self._models[var]
                if model is None:
                    # Root node (not intervened on) — keep original value
                    cf_values[var] = noise[var]
                else:
                    parents = self._graph.parents_of(var)
                    parent_vals = np.array(
                        [[cf_values[p] for p in parents]]
                    )
                    predicted = model.predict(parent_vals)[0]
                    # Add back the abducted noise to maintain consistency
                    cf_values[var] = predicted + noise[var]

        return cf_values

    # ── Validation ───────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Engine not fitted. Call .fit(data, graph) first."
            )

    @staticmethod
    def _validate_fit_input(data: pd.DataFrame, graph: CausalGraph) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(data).__name__}")
        if not isinstance(graph, CausalGraph):
            raise TypeError(
                f"Expected CausalGraph, got {type(graph).__name__}"
            )
        if data.empty:
            raise ValueError("Data is empty")

        # Check columns match graph variables
        data_cols = set(data.columns)
        graph_vars = set(graph.variables)
        missing = graph_vars - data_cols
        if missing:
            raise ValueError(
                f"Data is missing columns for graph variables: {missing}"
            )

        # Check for NaN
        nan_cols = data.columns[data.isna().any()].tolist()
        if nan_cols:
            raise ValueError(f"NaN values found in columns: {nan_cols}")

        # Check we have directed edges
        if graph.n_directed == 0:
            raise ValueError(
                "Graph has no directed edges. Counterfactual inference "
                "requires a directed causal graph."
            )
