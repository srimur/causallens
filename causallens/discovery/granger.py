"""
causallens.discovery.granger
=============================
Granger Causality test for time-series causal discovery.

Algorithm overview
------------------
For each pair (X, Y), test whether past values of X help predict Y
beyond Y's own past values:

    Restricted model:  Y_t = a0 + a1·Y_{t-1} + … + aL·Y_{t-L} + ε
    Unrestricted model: Y_t = a0 + a1·Y_{t-1} + … + aL·Y_{t-L}
                              + b1·X_{t-1} + … + bL·X_{t-L} + ε

If the unrestricted model is significantly better (F-test), then
"X Granger-causes Y", i.e. X → Y.

Key difference from PC algorithm
---------------------------------
- PC works on i.i.d. (cross-sectional) data
- Granger works on TIME SERIES data (rows = time steps, order matters)
- Granger gives DIRECTED edges (X→Y doesn't imply Y→X)
- Granger captures temporal/lagged causation, not instantaneous

Assumptions
-----------
1. Covariance stationarity (or data has been differenced)
2. Linear relationships
3. No latent common causes with specific lag structure

References
----------
- Granger, C.W.J. (1969). "Investigating causal relations by econometric
  models and cross-spectral methods." Econometrica, 37(3), 424-438.

Dependencies
------------
- ``causallens.types`` : CausalGraph, Edge, EdgeType
- numpy, pandas, statsmodels (OLS + F-test), scipy

No other CausalLens components are imported.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from causallens.types import CausalGraph, Edge, EdgeType


class GrangerCausality:
    """
    Pairwise Granger Causality test for time-series causal discovery.

    Parameters
    ----------
    max_lag : int, default 5
        Maximum number of lags to test.  The algorithm tests lags
        1, 2, …, max_lag and selects the one with the smallest p-value
        (strongest evidence of Granger causation).
    alpha : float, default 0.05
        Significance level for the F-test.  Edges with p < alpha are
        included in the output graph.
    verbose : bool, default False
        Print progress information during discovery.

    Examples
    --------
    >>> import pandas as pd
    >>> from causallens.discovery.granger import GrangerCausality
    >>> data = pd.read_csv("stock_prices.csv")  # columns = time series
    >>> gc = GrangerCausality(max_lag=5, alpha=0.05)
    >>> graph = gc.fit(data)
    >>> print(graph)
    >>> graph.parents_of("AAPL")
    ['SPY', 'QQQ']
    """

    def __init__(
        self,
        max_lag: int = 5,
        alpha: float = 0.05,
        verbose: bool = False,
    ) -> None:
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.max_lag = max_lag
        self.alpha = alpha
        self.verbose = verbose

        # Populated after fit()
        self._graph: Optional[CausalGraph] = None
        self._results_matrix: Optional[pd.DataFrame] = None
        self._lag_matrix: Optional[pd.DataFrame] = None
        self._n_tests: int = 0
        self._runtime: float = 0.0

    # ── Public API ───────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """
        Discover Granger-causal relationships from time series data.

        Parameters
        ----------
        data : pd.DataFrame
            Each column is a time series variable.  Rows must be in
            chronological order.  All columns must be numeric with no NaN.
            Minimum rows = max_lag + 3 (for meaningful regression).

        Returns
        -------
        CausalGraph
            Discovered causal graph with directed edges X → Y where
            "X Granger-causes Y".
        """
        self._validate_input(data)
        t0 = time.time()

        variables = list(data.columns)
        n_vars = len(variables)

        if self.verbose:
            print(f"[Granger] Testing {n_vars} variables, "
                  f"{len(data)} time steps, max_lag={self.max_lag}")

        # Pairwise test results
        p_matrix = pd.DataFrame(
            np.ones((n_vars, n_vars)),
            index=variables, columns=variables,
        )
        lag_matrix = pd.DataFrame(
            np.zeros((n_vars, n_vars), dtype=int),
            index=variables, columns=variables,
        )
        f_matrix = pd.DataFrame(
            np.zeros((n_vars, n_vars)),
            index=variables, columns=variables,
        )

        # Test all pairs
        for cause in variables:
            for effect in variables:
                if cause == effect:
                    continue

                best_p = 1.0
                best_lag = 0
                best_f = 0.0

                for lag in range(1, self.max_lag + 1):
                    f_stat, p_val, df_num, df_denom = self._granger_test_single(
                        data[effect].values,
                        data[cause].values,
                        lag,
                    )
                    self._n_tests += 1

                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag
                        best_f = f_stat

                p_matrix.loc[cause, effect] = best_p
                lag_matrix.loc[cause, effect] = best_lag
                f_matrix.loc[cause, effect] = best_f

                if self.verbose and best_p < self.alpha:
                    print(f"  {cause} → {effect}  "
                          f"(p={best_p:.4f}, lag={best_lag}, F={best_f:.2f})")

        # Build CausalGraph from significant results
        edges: List[Edge] = []
        for cause in variables:
            for effect in variables:
                if cause == effect:
                    continue
                p = p_matrix.loc[cause, effect]
                if p < self.alpha:
                    edges.append(Edge(
                        source=cause,
                        target=effect,
                        edge_type=EdgeType.DIRECTED,
                        weight=f_matrix.loc[cause, effect],
                        p_value=p,
                    ))

        self._runtime = time.time() - t0
        self._results_matrix = p_matrix
        self._lag_matrix = lag_matrix

        self._graph = CausalGraph(
            variables=variables,
            edges=edges,
            metadata={
                "algorithm": "Granger",
                "max_lag": self.max_lag,
                "alpha": self.alpha,
                "n_tests": self._n_tests,
                "runtime_seconds": round(self._runtime, 4),
                "n_observations": len(data),
            },
        )

        if self.verbose:
            print(f"[Granger] Done in {self._runtime:.2f}s — "
                  f"{self._n_tests} tests — "
                  f"{self._graph.n_directed} directed edges found.")

        return self._graph

    def test_pair(
        self,
        data: pd.DataFrame,
        cause: str,
        effect: str,
        lag: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a Granger test for a single pair (cause → effect).

        Parameters
        ----------
        data : pd.DataFrame
            Time series data.
        cause : str
            Name of the potential cause variable.
        effect : str
            Name of the potential effect variable.
        lag : int or None
            Specific lag to test.  If None, tests all lags up to max_lag
            and returns the one with the strongest evidence.

        Returns
        -------
        dict with keys:
            - granger_causes : bool
            - p_value : float
            - f_statistic : float
            - best_lag : int
            - all_lags : list[dict]  (results for each lag tested)
        """
        self._validate_input(data)
        if cause not in data.columns:
            raise ValueError(f"'{cause}' not found in data columns")
        if effect not in data.columns:
            raise ValueError(f"'{effect}' not found in data columns")
        if cause == effect:
            raise ValueError("cause and effect must be different variables")

        x = data[cause].values
        y = data[effect].values

        if lag is not None:
            if lag < 1:
                raise ValueError(f"lag must be >= 1, got {lag}")
            f_stat, p_val, df_num, df_denom = self._granger_test_single(y, x, lag)
            return {
                "granger_causes": p_val < self.alpha,
                "p_value": p_val,
                "f_statistic": f_stat,
                "best_lag": lag,
                "df_numerator": df_num,
                "df_denominator": df_denom,
                "all_lags": [{"lag": lag, "p_value": p_val, "f_statistic": f_stat}],
            }

        # Test all lags, return best
        all_lags = []
        best_p = 1.0
        best_result = None

        for L in range(1, self.max_lag + 1):
            f_stat, p_val, df_num, df_denom = self._granger_test_single(y, x, L)
            lag_result = {
                "lag": L,
                "p_value": p_val,
                "f_statistic": f_stat,
                "df_numerator": df_num,
                "df_denominator": df_denom,
            }
            all_lags.append(lag_result)
            if p_val < best_p:
                best_p = p_val
                best_result = lag_result

        return {
            "granger_causes": best_p < self.alpha,
            "p_value": best_result["p_value"],
            "f_statistic": best_result["f_statistic"],
            "best_lag": best_result["lag"],
            "df_numerator": best_result["df_numerator"],
            "df_denominator": best_result["df_denominator"],
            "all_lags": all_lags,
        }

    @property
    def graph(self) -> CausalGraph:
        """Access the discovered graph. Raises if :meth:`fit` not called."""
        if self._graph is None:
            raise RuntimeError("Call .fit(data) before accessing .graph")
        return self._graph

    @property
    def p_value_matrix(self) -> pd.DataFrame:
        """
        Matrix of p-values. ``p_matrix.loc[X, Y]`` is the p-value for
        "X Granger-causes Y". Available after :meth:`fit`.
        """
        if self._results_matrix is None:
            raise RuntimeError("Call .fit(data) before accessing .p_value_matrix")
        return self._results_matrix.copy()

    @property
    def lag_matrix(self) -> pd.DataFrame:
        """
        Matrix of optimal lags. ``lag_matrix.loc[X, Y]`` is the lag that
        gave the strongest Granger evidence for X → Y.
        """
        if self._lag_matrix is None:
            raise RuntimeError("Call .fit(data) before accessing .lag_matrix")
        return self._lag_matrix.copy()

    # ── Core statistical test ────────────────────────────────────────────

    @staticmethod
    def _granger_test_single(
        y: np.ndarray,
        x: np.ndarray,
        lag: int,
    ) -> Tuple[float, float, int, int]:
        """
        Single Granger causality F-test: does X Granger-cause Y at
        the given lag?

        Builds restricted (Y's own lags only) and unrestricted (Y's lags
        + X's lags) OLS models via statsmodels, then computes the
        F-statistic comparing residual sum of squares.

        Parameters
        ----------
        y : array, shape (T,)
            The effect variable time series.
        x : array, shape (T,)
            The potential cause variable time series.
        lag : int
            Number of lags to include.

        Returns
        -------
        (f_statistic, p_value, df_numerator, df_denominator)
        """
        T = len(y)
        if T <= 2 * lag + 1:
            return 0.0, 1.0, lag, 1

        # Build lag matrices
        y_target = y[lag:]
        n = len(y_target)

        # Restricted regressors: Y_{t-1}, ..., Y_{t-lag}
        y_lags = np.column_stack([y[lag - i - 1: T - i - 1] for i in range(lag)])

        # Unrestricted regressors: Y lags + X lags
        x_lags = np.column_stack([x[lag - i - 1: T - i - 1] for i in range(lag)])

        # Add constant (intercept)
        Z_restricted = add_constant(y_lags)
        Z_unrestricted = add_constant(np.column_stack([y_lags, x_lags]))

        # Fit OLS models via statsmodels
        try:
            model_r = OLS(y_target, Z_restricted).fit()
            model_u = OLS(y_target, Z_unrestricted).fit()
        except Exception:
            return 0.0, 1.0, lag, 1

        # F-test: compare residual sum of squares
        ssr_r = model_r.ssr
        ssr_u = model_u.ssr
        df_num = lag
        df_denom = n - 2 * lag - 1

        if df_denom <= 0 or ssr_u <= 0:
            return 0.0, 1.0, df_num, max(df_denom, 1)

        f_stat = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_denom)

        if f_stat < 0:
            f_stat = 0.0

        p_value = float(1.0 - f_dist.cdf(f_stat, df_num, df_denom))

        return f_stat, p_value, df_num, df_denom

    # ── Stationarity check (helper) ──────────────────────────────────────

    @staticmethod
    def check_stationarity(
        series: pd.Series,
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run the Augmented Dickey-Fuller test for stationarity.

        Parameters
        ----------
        series : pd.Series
            Time series to test.
        significance : float
            Significance level.

        Returns
        -------
        dict with keys: is_stationary, adf_statistic, p_value,
                        n_lags_used, n_observations, critical_values
        """
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "is_stationary": result[1] < significance,
            "adf_statistic": result[0],
            "p_value": result[1],
            "n_lags_used": result[2],
            "n_observations": result[3],
            "critical_values": result[4],
        }

    # ── Input validation ─────────────────────────────────────────────────

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Raise ValueError if data is unsuitable for Granger tests."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(data).__name__}")

        if data.empty:
            raise ValueError("Data is empty")

        if data.shape[1] < 2:
            raise ValueError("Need at least 2 variables for Granger causality")

        # Check numeric
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Non-numeric columns found: {non_numeric}. "
                f"All columns must be numeric time series."
            )

        # Check NaN
        nan_cols = data.columns[data.isna().any()].tolist()
        if nan_cols:
            raise ValueError(
                f"NaN values found in columns: {nan_cols}. "
                f"Drop or impute missing values before calling fit()."
            )

        # Check minimum length
        min_required = self.max_lag + 3
        if len(data) < min_required:
            raise ValueError(
                f"Need at least {min_required} time steps for max_lag={self.max_lag}, "
                f"got {len(data)}."
            )

        # Warn about near-zero variance
        import warnings
        for col in data.columns:
            std = data[col].std()
            if std < 1e-10:
                warnings.warn(
                    f"Column '{col}' has near-zero variance — "
                    f"Granger test results will be unreliable.",
                    UserWarning,
                    stacklevel=3,
                )
