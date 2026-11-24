"""Benjamini–Hochberg FDR adjustment for multiplicity in post-hoc inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

DEFAULT_ALPHA = 0.05


def apply_bh_fdr(
    p_values: np.ndarray | pd.Series,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_values, reject) from Benjamini–Hochberg FDR at level alpha."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    reject = np.zeros(p.shape, dtype=bool)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q, reject
    rj, qv, _, _ = multipletests(p[valid], alpha=alpha, method="fdr_bh")
    q[valid] = qv
    reject[valid] = rj
    return q, reject


def add_fdr_to_or_table(
    or_table: pd.DataFrame,
    *,
    p_col: str = "p_value",
    term_col: str = "term",
    exclude_intercept: bool = True,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Add q_value_fdr_bh and significant_fdr_05 columns; intercept left NA."""
    out = or_table.copy()
    test_mask = np.ones(len(out), dtype=bool)
    if exclude_intercept and term_col in out.columns:
        test_mask = ~out[term_col].astype(str).str.contains("Intercept", case=False, na=False)

    q_all = np.full(len(out), np.nan)
    rej_all = np.zeros(len(out), dtype=bool)
    if test_mask.any():
        q_sub, rej_sub = apply_bh_fdr(out.loc[test_mask, p_col], alpha=alpha)
        q_all[test_mask] = q_sub
        rej_all[test_mask] = rej_sub

    out["q_value_fdr_bh"] = q_all
    out["significant_fdr_05"] = rej_all
    out["fdr_method"] = "Benjamini-Hochberg"
    out["fdr_alpha"] = alpha
    out["n_tests_fdr"] = int(test_mask.sum())
    return out
