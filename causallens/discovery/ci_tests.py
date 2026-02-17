"""
causallens.discovery.ci_tests
=============================
Conditional Independence (CI) tests used by causal discovery algorithms.

Provides :func:`partial_correlation` which:
- Uses a compiled C++ backend for speed when available (~20-50× faster)
- Falls back to a pure NumPy implementation otherwise

This module is independently testable — it has NO dependency on other
CausalLens components.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

# ── Try to load C++ backend ─────────────────────────────────────────────────

_LIB: Optional[ctypes.CDLL] = None
_NATIVE_DIR = Path(__file__).parent / "native"
_SO_PATH = _NATIVE_DIR / "libpartialcorr.so"

if _SO_PATH.exists():
    try:
        _LIB = ctypes.CDLL(str(_SO_PATH))
        _LIB.partial_correlation.restype = ctypes.c_int
        _LIB.partial_correlation.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.c_int,                      # n_rows
            ctypes.c_int,                      # n_cols_total
            ctypes.c_int,                      # x_idx
            ctypes.c_int,                      # y_idx
            ctypes.POINTER(ctypes.c_int),      # z_indices
            ctypes.c_int,                      # z_len
            ctypes.POINTER(ctypes.c_double),   # out_corr
            ctypes.POINTER(ctypes.c_double),   # out_pvalue
        ]
    except OSError:
        _LIB = None


def _has_native_backend() -> bool:
    """Return True if the C++ partial correlation library is loaded."""
    return _LIB is not None


# ── Public API ───────────────────────────────────────────────────────────────


def partial_correlation(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    z_indices: List[int],
) -> Tuple[float, float]:
    """
    Compute partial correlation between columns ``x_idx`` and ``y_idx``
    of *data*, conditioned on columns in ``z_indices``.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_variables)
        Observation matrix.  **Must be C-contiguous float64.**
    x_idx, y_idx : int
        Column indices of the two variables being tested.
    z_indices : list[int]
        Column indices of the conditioning set (may be empty).

    Returns
    -------
    (correlation, p_value) : tuple[float, float]
        Partial correlation coefficient in [-1, 1] and its two-sided p-value.

    Notes
    -----
    When ``z_indices`` is empty this reduces to Pearson correlation.
    When non-empty, both x and y are OLS-residualized on [1, Z] and the
    residuals are Pearson-correlated.  The p-value is computed from the
    t-statistic with ``df = n - |Z| - 2``.
    """
    if _LIB is not None:
        return _partial_corr_native(data, x_idx, y_idx, z_indices)
    return _partial_corr_numpy(data, x_idx, y_idx, z_indices)


# ── C++ backend ──────────────────────────────────────────────────────────────


def _partial_corr_native(
    data: np.ndarray, x_idx: int, y_idx: int, z_indices: List[int],
) -> Tuple[float, float]:
    """Call into the compiled C++ shared library."""
    assert _LIB is not None

    # Ensure contiguous float64
    if not data.flags["C_CONTIGUOUS"] or data.dtype != np.float64:
        data = np.ascontiguousarray(data, dtype=np.float64)

    n_rows, n_cols = data.shape
    z_arr = (ctypes.c_int * len(z_indices))(*z_indices) if z_indices else (ctypes.c_int * 0)()
    out_corr = ctypes.c_double(0.0)
    out_pval = ctypes.c_double(1.0)

    rc = _LIB.partial_correlation(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_rows),
        ctypes.c_int(n_cols),
        ctypes.c_int(x_idx),
        ctypes.c_int(y_idx),
        z_arr,
        ctypes.c_int(len(z_indices)),
        ctypes.byref(out_corr),
        ctypes.byref(out_pval),
    )

    if rc != 0:
        # Numerical failure — treat as independent
        return 0.0, 1.0

    return out_corr.value, out_pval.value


# ── NumPy fallback ───────────────────────────────────────────────────────────


def _partial_corr_numpy(
    data: np.ndarray, x_idx: int, y_idx: int, z_indices: List[int],
) -> Tuple[float, float]:
    """Pure NumPy implementation — no compiled dependencies needed."""
    n = data.shape[0]
    x = data[:, x_idx].astype(np.float64)
    y = data[:, y_idx].astype(np.float64)

    if len(z_indices) == 0:
        r, p = sp_stats.pearsonr(x, y)
        return float(r), float(p)

    # Build regressor matrix [intercept, z1, z2, ...]
    Z = np.column_stack([
        np.ones(n, dtype=np.float64),
        data[:, z_indices].astype(np.float64),
    ])

    # OLS residuals via least-squares
    try:
        beta_x, _, _, _ = np.linalg.lstsq(Z, x, rcond=None)
        beta_y, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    resid_x = x - Z @ beta_x
    resid_y = y - Z @ beta_y

    # Pearson correlation of residuals
    std_x = np.std(resid_x, ddof=0)
    std_y = np.std(resid_y, ddof=0)
    if std_x < 1e-15 or std_y < 1e-15:
        return 0.0, 1.0

    r = float(np.corrcoef(resid_x, resid_y)[0, 1])
    r = max(-1.0, min(1.0, r))  # clamp

    # p-value from t-distribution
    df = n - len(z_indices) - 2
    if df <= 0:
        return r, 1.0

    if abs(r) >= 1.0 - 1e-15:
        return r, 0.0

    t_stat = r * np.sqrt(df / (1.0 - r * r))
    p_value = float(2.0 * sp_stats.t.sf(abs(t_stat), df))
    return r, p_value
