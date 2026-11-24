"""Knowledge-graph edges derived solely from processed BRFSS train-split CSVs.

No LASSO lists, GLM tables, permutation CSVs, ontology priors, or any file under
05_results/01–05 or posthoc/. All associations are computed fresh from rows in
02_datasets/processed/*_train.csv.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

OUTCOME_NODE = "Stroke"
OUTCOME_COL = "stroke"

META_COLS = frozenset({
    "survey_year", "state_fips", "split", "stroke",
    "survey_weight", "strata", "psu", "seqno",
})


def predictor_columns(df: pd.DataFrame) -> list[str]:
    """All harmonized predictors present in a processed split file."""
    return [c for c in df.columns if c not in META_COLS]


def prepare_train_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Numeric train matrix with median imputation (fit on this split only).

    Survey BRFSS predictors have item non-response; listwise deletion would
    remove all rows. Imputation is computed fresh here — not from ML pipelines.
    """
    cols = features + [OUTCOME_COL]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    medians = sub.median(numeric_only=True)
    return sub.fillna(medians)


def partial_correlation_matrix(
    df: pd.DataFrame, features: list[str], *, min_abs_r: float = 0.12
) -> list[dict[str, Any]]:
    """Pairwise partial correlations among features (controls: other features + Stroke)."""
    sub = prepare_train_matrix(df, features).dropna(subset=[OUTCOME_COL])
    if len(sub) < 50:
        return []

    data = sub.values
    precision = np.linalg.pinv(np.cov(data, rowvar=False))
    edges: list[dict[str, Any]] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a, b = i, j
            if precision[a, a] <= 0 or precision[b, b] <= 0:
                continue
            r = -precision[a, b] / np.sqrt(precision[a, a] * precision[b, b])
            if abs(r) < min_abs_r:
                continue
            w = abs(float(r))
            edges.append(
                {
                    "source": features[a],
                    "target": features[b],
                    "weight": w,
                    "source_type": "partial_corr",
                    "relation": "conditional_association",
                    "partial_r": float(r),
                }
            )
    return edges


def partial_correlation_to_outcome(
    df: pd.DataFrame, features: list[str], *, min_abs_r: float = 0.05
) -> list[dict[str, Any]]:
    """Partial correlation feature → Stroke controlling all other predictors."""
    sub = prepare_train_matrix(df, features).dropna(subset=[OUTCOME_COL])
    if len(sub) < 50:
        return []

    data = sub[features + [OUTCOME_COL]].values
    precision = np.linalg.pinv(np.cov(data, rowvar=False))
    outcome_idx = len(features)
    if precision[outcome_idx, outcome_idx] <= 0:
        return []

    edges: list[dict[str, Any]] = []
    for i, feat in enumerate(features):
        if precision[i, i] <= 0:
            continue
        r = -precision[i, outcome_idx] / np.sqrt(precision[i, i] * precision[outcome_idx, outcome_idx])
        if abs(r) < min_abs_r:
            continue
        w = abs(float(r))
        edges.append(
            {
                "source": feat,
                "target": OUTCOME_NODE,
                "weight": w,
                "source_type": "partial_corr_stroke",
                "relation": "partial_association_with_stroke",
                "partial_r": float(r),
            }
        )
    return edges


def marginal_logistic_edges(
    df: pd.DataFrame, features: list[str], *, min_abs_log_or: float = 0.05
) -> list[dict[str, Any]]:
    """Univariate logistic(stroke ~ feature) on train rows — fit in-memory, no saved models."""
    sub = prepare_train_matrix(df, features).dropna(subset=[OUTCOME_COL])
    if len(sub) < 50 or sub[OUTCOME_COL].nunique() < 2:
        return []

    y = sub[OUTCOME_COL].astype(int).values
    edges: list[dict[str, Any]] = []
    for feat in features:
        x = sub[[feat]].values
        if np.nanstd(x) < 1e-12:
            continue
        try:
            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            clf.fit(x, y)
            log_or = float(clf.coef_.ravel()[0])
            or_val = float(np.exp(log_or))
        except Exception:
            continue
        if abs(log_or) < min_abs_log_or:
            continue
        weight = abs(log_or) * np.sqrt(len(x) / 1000.0)
        if weight <= 0:
            continue
        edges.append(
            {
                "source": feat,
                "target": OUTCOME_NODE,
                "weight": float(weight),
                "source_type": "marginal_logistic",
                "relation": "univariate_logistic_association",
                "log_or": log_or,
                "or": or_val,
            }
        )
    return edges


def point_biserial_edges(
    df: pd.DataFrame, features: list[str], *, min_abs_r: float = 0.05
) -> list[dict[str, Any]]:
    """Marginal Pearson correlation between each numeric feature and Stroke."""
    sub = prepare_train_matrix(df, features).dropna(subset=[OUTCOME_COL])
    if len(sub) < 50:
        return []

    y = sub[OUTCOME_COL].astype(float)
    edges: list[dict[str, Any]] = []
    for feat in features:
        x = sub[feat].astype(float)
        if x.nunique() < 2:
            continue
        r, p = stats.pearsonr(x, y)
        if abs(r) < min_abs_r:
            continue
        weight = abs(float(r)) * min(-np.log10(max(p, 1e-300)), 50.0)
        edges.append(
            {
                "source": feat,
                "target": OUTCOME_NODE,
                "weight": float(weight),
                "source_type": "point_biserial",
                "relation": "marginal_correlation_with_stroke",
                "pearson_r": float(r),
                "p_value": float(p),
            }
        )
    return edges
