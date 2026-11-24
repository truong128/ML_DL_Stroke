#!/usr/bin/env python3
"""
Usage:
  python scripts/run_ml_pipeline.py --dataset ny
  python scripts/run_ml_pipeline.py --dataset all_states --quick
  python scripts/run_ml_pipeline.py --dataset ny --n-jobs -1   # all CPUs (default)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
from joblib import Parallel, delayed, effective_n_jobs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def _find_data_root() -> Path:
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        if (base / "02_datasets" / "processed").exists() or (base / "02_datasets" / "brfss").exists():
            return base / "02_datasets"
        if (base / "datasets" / "processed").exists() or (base / "datasets" / "brfss").exists():
            return base / "datasets"
    raise SystemExit("Could not locate datasets/processed (run bootstrap or build_ml_datasets first).")

DATA_ROOT = _find_data_root()
PROCESSED = DATA_ROOT / "processed"

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        if (base / "results").is_dir() and (base / "datasets").is_dir():
            return base
        if (base / "05_results").is_dir() and (base / "02_datasets").is_dir():
            return base
    raise SystemExit("Could not locate project root (run bootstrap_project.sh).")


ROOT = _find_project_root()


def pipeline_results_dir(pipeline_name: str, dataset: str) -> Path:
    """Resolve output dir for bootstrapped project/ or package layout."""
    bootstrapped = {
        "ml": ("results", "ml"),
        "ml_voting_soft": ("results", "ml_voting_soft"),
    }
    package = {
        "ml": ("05_results", "01_python_ml_standard"),
        "ml_voting_soft": ("05_results", "02_python_ml_voting_soft"),
    }
    if (ROOT / "results").is_dir():
        a, b = bootstrapped[pipeline_name]
        return ROOT / a / b / dataset
    a, b = package[pipeline_name]
    return ROOT / a / b / dataset


META_COLS = {
    "survey_year", "state_fips", "split", "stroke",
    "survey_weight", "strata", "psu", "seqno",
}
BATCH_COL = "survey_year"
TARGET = "stroke"

# Set by init_parallel() / init_gpu() before run_pipeline()
N_JOBS: int = -1
GPU_CFG: dict = {
    "enabled": False,
    "device": "cpu",
    "gradient_boosting_backend": "sklearn_hist_gb",
    "quick": False,
}


def year_stratified_group_cv(
    n_splits: int, *, shuffle: bool = True, random_state: int = 42
) -> StratifiedGroupKFold:
    """CV folds do not mix survey_year across train/test (temporal groups)."""
    return StratifiedGroupKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )


def init_gpu(use_gpu: bool = False, quick: bool = False) -> dict:
    """Enable CUDA XGBoost for gradient_boosting when --use-gpu and GPU is available."""
    global GPU_CFG
    cfg = {
        "enabled": False,
        "device": "cpu",
        "gradient_boosting_backend": "sklearn_hist_gb",
        "quick": quick,
        "nvidia_visible": False,
        "message": "GPU: disabled (CPU-only run).",
    }
    if not use_gpu:
        GPU_CFG = cfg
        return cfg
    try:
        import subprocess

        subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        cfg["nvidia_visible"] = True
    except (OSError, subprocess.SubprocessError):
        cfg["message"] = "GPU: --use-gpu set but nvidia-smi failed; using CPU."
        GPU_CFG = cfg
        return cfg
    try:
        import xgboost as xgb

        X = np.random.randn(80, 8).astype(np.float32)
        y = np.random.randint(0, 2, 80)
        probe = xgb.XGBClassifier(
            n_estimators=2, max_depth=2, device="cuda", tree_method="hist"
        )
        probe.fit(X, y)
        cfg["enabled"] = True
        cfg["device"] = "cuda"
        cfg["gradient_boosting_backend"] = "xgboost_cuda"
        cfg["message"] = (
            f"GPU: enabled for gradient_boosting (xgboost {xgb.__version__}, device=cuda)."
        )
    except ImportError:
        cfg["message"] = (
            "GPU: --use-gpu set but xgboost not installed; "
            "pip install xgboost in your venv. Using sklearn HistGradientBoosting on CPU."
        )
    except Exception as e:
        cfg["message"] = f"GPU: CUDA probe failed ({e}); using CPU."
    GPU_CFG = cfg
    return cfg


def init_parallel(n_jobs: int = -1) -> int:
    """Use all (or N) CPU cores; cap BLAS threads per worker to reduce oversubscription."""
    global N_JOBS
    N_JOBS = effective_n_jobs(n_jobs)
    cpus = os.cpu_count() or 1
    per_worker = max(1, cpus // max(N_JOBS, 1))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(per_worker)
    return N_JOBS


class BatchMeanCorrector(BaseEstimator, TransformerMixin):
    """Remove survey_year mean per feature (train-fitted). ComBat-lite for tabular data."""

    def __init__(self, batch_col_idx: int):
        self.batch_col_idx = batch_col_idx
        self.batch_means_: dict = {}
        self.global_mean_: np.ndarray | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        batches = X[:, self.batch_col_idx]
        self.global_mean_ = np.nanmean(X[:, 1:], axis=0)  # skip batch col in feature means
        self.batch_means_ = {}
        for b in np.unique(batches):
            mask = batches == b
            self.batch_means_[b] = np.nanmean(X[mask, 1:], axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        batches = X[:, self.batch_col_idx]
        for i, b in enumerate(batches):
            if b in self.batch_means_:
                X[i, 1:] -= self.batch_means_[b] - self.global_mean_
        return X


def load_splits(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED / f"{prefix}_train.csv", low_memory=False)
    val = pd.read_csv(PROCESSED / f"{prefix}_val.csv", low_memory=False)
    test = pd.read_csv(PROCESSED / f"{prefix}_test.csv", low_memory=False)
    return train, val, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def fit_batch_correction(
    X_ref: np.ndarray, batches_ref: np.ndarray
) -> tuple[SimpleImputer, np.ndarray, dict]:
    """Median impute + remove survey_year means (fit on reference split only)."""
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_ref)
    global_mean = np.nanmean(X_imp, axis=0)
    year_means: dict = {}
    for yr in np.unique(batches_ref):
        year_means[yr] = np.nanmean(X_imp[batches_ref == yr], axis=0)
    return imp, global_mean, year_means


def apply_batch_correction(
    X: np.ndarray,
    batches: np.ndarray,
    imp: SimpleImputer,
    global_mean: np.ndarray,
    year_means: dict,
) -> np.ndarray:
    X_bc = imp.transform(X).copy()
    for i, yr in enumerate(batches):
        if yr in year_means:
            X_bc[i] -= year_means[yr] - global_mean
    return X_bc


def _youden_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Youden's J = sensitivity + specificity - 1."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens + spec - 1.0


def choose_classification_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "youden",
) -> tuple[float, dict]:
    """
    Tune threshold on validation probabilities.
    Default: Youden's J (sensitivity + specificity - 1).
    Skips degenerate thresholds that predict only one class.
    """
    candidates = np.unique(
        np.concatenate([np.quantile(y_prob, np.linspace(0.05, 0.95, 50)), [0.5]])
    )
    best_t, best_score = 0.5, -1.0
    for t in candidates:
        pred = (y_prob >= t).astype(int)
        if pred.min() == pred.max():
            continue
        if metric == "youden":
            score = _youden_j(y_true, pred)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        else:
            score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    info = {
        "metric": metric,
        "val_score": float(best_score if best_score >= 0 else np.nan),
        "n_candidates": int(len(candidates)),
    }
    return best_t, info


def prepare_xy(
    df: pd.DataFrame, feature_cols: list[str], include_batch: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = feature_cols.copy()
    if include_batch and BATCH_COL in df.columns and BATCH_COL not in cols:
        cols = [BATCH_COL] + cols
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df[TARGET].astype(int).values
    return X.values, y, list(X.columns)


def plot_batch_pca(X, y, batches, out_path: Path, title: str) -> None:
    from sklearn.decomposition import PCA

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(StandardScaler().fit_transform(X_imp))
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in np.unique(batches):
        m = batches == b
        ax.scatter(Z[m, 0], Z[m, 1], alpha=0.35, s=12, label=str(int(b)))
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def get_classifiers() -> dict:
    """Sixteen supervised classifiers (+ ensembles built from top performers)."""
    nj = N_JOBS
    quick = bool(GPU_CFG.get("quick", False))
    dt_est = DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=42,
    )
    if GPU_CFG.get("enabled"):
        import xgboost as xgb

        gb_clf = xgb.XGBClassifier(
            n_estimators=100 if quick else 200,
            max_depth=4,
            learning_rate=0.05,
            device="cuda",
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
        )
    else:
        gb_clf = HistGradientBoostingClassifier(
            max_iter=100 if quick else 200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42,
        ),
        # sklearn 1.8+: distinct linear model (saga) — R side uses glmnet multinom_reg
        "multinom_reg": LogisticRegression(
            solver="saga",
            l1_ratio=0,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "svm_linear": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
        "decision_tree": clone(dt_est),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=nj,
        ),
        "gradient_boosting": gb_clf,
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=nj,
        ),
        "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "bag_tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=42,
            ),
            n_estimators=25,
            max_samples=0.8,
            random_state=42,
            n_jobs=nj,
        ),
        "bagging": BaggingClassifier(
            estimator=clone(dt_est),
            n_estimators=100, random_state=42, n_jobs=nj,
        ),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(reg_param=0.5),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True,
            validation_fraction=0.15, random_state=42,
        ),
    }


def make_pipeline(
    clf, use_smote: bool, use_quantile: bool, y: np.ndarray | None = None
) -> ImbPipeline | Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if use_quantile:
        steps.append(("quantile", QuantileTransformer(output_distribution="normal", random_state=42)))
    steps.append(("scaler", StandardScaler()))
    if use_smote:
        k = 5
        if y is not None:
            n_pos = int((y == 1).sum())
            k = min(5, max(1, n_pos - 1))
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=k)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def _lasso_logistic(C: float) -> LogisticRegression:
    """L1 logistic (LASSO). liblinear converges reliably vs saga on BRFSS-scale data."""
    return LogisticRegression(
        l1_ratio=1.0,
        solver="liblinear",
        C=C,
        max_iter=5000,
        tol=1e-4,
        class_weight="balanced",
        random_state=42,
    )


def _lasso_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lasso", _lasso_logistic(C)),
    ])


def lasso_select(
    X_train,
    y_train,
    feature_names: list[str],
    C: float = 0.05,
    max_features: int = 25,
) -> list[str]:
    """LASSO on z-scored train (manuscript-style). Fit only on train."""
    pipe = _lasso_pipeline(C)
    pipe.fit(X_train, y_train)
    coef = pipe.named_steps["lasso"].coef_.ravel()
    names = [n for n, c in zip(feature_names, coef) if abs(c) > 1e-6]
    if len(names) > max_features:
        order = np.argsort(np.abs(coef))[-max_features:]
        names = [feature_names[i] for i in order if abs(coef[i]) > 1e-6]
    return names


def _inner_cv_tune_lasso_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_grid: list[float],
    inner_splits: int,
) -> tuple[float, pd.DataFrame]:
    """Inner loop: pick LASSO C by year-grouped CV ROC-AUC (no outer-test peeking)."""
    inner_cv = year_stratified_group_cv(inner_splits)

    def _score_c(C: float) -> dict:
        pipe = _lasso_pipeline(C)
        scores = cross_val_score(
            pipe, X, y, groups=groups, cv=inner_cv,
            scoring="roc_auc", n_jobs=1, error_score=np.nan,
        )
        return {
            "C": C,
            "roc_auc_mean": float(np.nanmean(scores)),
            "roc_auc_std": float(np.nanstd(scores)),
        }

    rows = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_score_c)(C) for C in C_grid
    )
    grid_df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    best_C = float(grid_df.iloc[0]["C"])
    return best_C, grid_df


def lasso_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    *,
    C_grid: list[float] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    max_features: int = 30,
    stability_threshold: float = 0.5,
    eval_classifier: str = "logistic_regression",
) -> dict:
    """
    Nested CV for LASSO:
      - Data: train split only (no val/test).
      - Groups: survey_year (StratifiedGroupKFold — years not mixed across folds).
      - Inner CV: tune LASSO C on outer-train folds only.
      - Outer CV: unbiased ROC-AUC after feature selection on outer-test folds.
    """
    if C_grid is None:
        C_grid = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    clf = get_classifiers()[eval_classifier]
    outer_cv = year_stratified_group_cv(outer_splits)

    outer_rows = []
    inner_grid_all = []
    feature_counts: dict[str, int] = {f: 0 for f in feature_names}

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_otr, X_ote = X[tr_idx], X[te_idx]
        y_otr, y_ote = y[tr_idx], y[te_idx]
        g_otr = groups[tr_idx]

        best_C, inner_df = _inner_cv_tune_lasso_C(X_otr, y_otr, g_otr, C_grid, inner_splits)
        inner_df = inner_df.copy()
        inner_df["outer_fold"] = fold
        inner_grid_all.append(inner_df)

        selected_fold = lasso_select(
            X_otr, y_otr, feature_names, C=best_C, max_features=max_features
        )
        for f in selected_fold:
            feature_counts[f] += 1

        if len(selected_fold) < 2:
            outer_rows.append({
                "outer_fold": fold,
                "best_C": best_C,
                "n_features": len(selected_fold),
                "outer_roc_auc": np.nan,
                "note": "too_few_features",
            })
            continue

        sel_idx = [feature_names.index(f) for f in selected_fold]
        pipe = make_pipeline(clone(clf), use_smote=True, use_quantile=False, y=y_otr)
        pipe.fit(X_otr[:, sel_idx], y_otr)
        y_prob = pipe.predict_proba(X_ote[:, sel_idx])[:, 1]
        outer_auc = roc_auc_score(y_ote, y_prob) if len(np.unique(y_ote)) > 1 else np.nan

        outer_rows.append({
            "outer_fold": fold,
            "best_C": best_C,
            "n_features": len(selected_fold),
            "outer_roc_auc": outer_auc,
            "note": "",
        })

    outer_df = pd.DataFrame(outer_rows)
    inner_all_df = pd.concat(inner_grid_all, ignore_index=True)

    stability_df = pd.DataFrame([
        {
            "feature": f,
            "times_selected": feature_counts[f],
            "stability": feature_counts[f] / outer_splits,
        }
        for f in feature_names
    ]).sort_values("stability", ascending=False)
    stable_features = stability_df.loc[
        stability_df["stability"] >= stability_threshold, "feature"
    ].tolist()

    # Fallback if stability selection is too sparse
    if len(stable_features) < 5:
        median_C = float(outer_df["best_C"].median())
        stable_features = lasso_select(
            X, y, feature_names, C=median_C, max_features=max_features
        )
        selection_method = f"fallback_median_C={median_C:.4g}"
    else:
        selection_method = f"stability>={stability_threshold}"

    summary = {
        "outer_roc_auc_mean": float(outer_df["outer_roc_auc"].mean()),
        "outer_roc_auc_std": float(outer_df["outer_roc_auc"].std()),
        "median_best_C": float(outer_df["best_C"].median()),
        "n_stable_features": len(stable_features),
        "stable_features": stable_features,
        "selection_method": selection_method,
    }

    outer_df.to_csv(out_dir / "02a_lasso_nested_cv_outer_folds.csv", index=False)
    inner_all_df.to_csv(out_dir / "02b_lasso_nested_cv_inner_C_grid.csv", index=False)
    stability_df.to_csv(out_dir / "02c_lasso_feature_stability.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "02d_lasso_nested_cv_summary.csv", index=False)

    # Plot: outer fold AUC + top stable features
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(outer_df["outer_fold"].astype(str), outer_df["outer_roc_auc"], color="steelblue")
    axes[0].axhline(summary["outer_roc_auc_mean"], color="red", ls="--", label="mean")
    axes[0].set_xlabel("Outer fold")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Nested CV — outer-test performance")
    axes[0].legend(fontsize=8)

    top_stab = stability_df.head(15)
    axes[1].barh(top_stab["feature"], top_stab["stability"], color="seagreen")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Selection frequency")
    axes[1].set_title("LASSO feature stability (outer folds)")
    fig.tight_layout()
    fig.savefig(out_dir / "02e_lasso_nested_cv_plots.png", dpi=150)
    plt.close(fig)

    return summary


def _cv_evaluate_one_model(
    name: str,
    clf,
    X,
    y,
    groups: np.ndarray,
    cv,
    scoring: dict,
    use_smote: bool,
    use_quantile: bool,
) -> dict:
    pipe = make_pipeline(clone(clf), use_smote=use_smote, use_quantile=use_quantile, y=y)
    scores = cross_validate(
        pipe, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan
    )
    return {
        "model": name,
        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
        "f1_mean": scores["test_f1"].mean(),
        "balanced_acc_mean": scores["test_balanced_accuracy"].mean(),
        "pr_auc_mean": scores["test_average_precision"].mean(),
    }


def select_best_model_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    use_quantile: bool,
) -> tuple[str, pd.DataFrame]:
    """Pick best classifier by validation ROC-AUC (train fit, no SMOTE)."""
    rows = []
    for name, clf in get_classifiers().items():
        pipe = make_pipeline(clone(clf), use_smote=False, use_quantile=use_quantile, y=y_train)
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        rows.append({
            "model": name,
            "val_roc_auc": float(roc_auc_score(y_val, y_prob)),
            "val_pr_auc": float(average_precision_score(y_val, y_prob)),
        })
    df = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False)
    return str(df.iloc[0]["model"]), df


def cv_evaluate_models(
    X,
    y,
    groups: np.ndarray,
    selected_features: list[str],
    out_dir: Path,
    use_smote: bool,
    use_quantile: bool,
    n_splits: int = 5,
) -> pd.DataFrame:
    cv = year_stratified_group_cv(n_splits)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "average_precision": "average_precision",
    }
    rows = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_cv_evaluate_one_model)(
            name, clf, X, y, groups, cv, scoring, use_smote, use_quantile
        )
        for name, clf in get_classifiers().items()
    )
    df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    df.to_csv(out_dir / "03_cv_model_comparison.csv", index=False)
    return df


def _bootstrap_one(
    seed: int, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
) -> dict | None:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, n)
    yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
    if len(np.unique(yt)) < 2:
        return None
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(yt, ypr),
        "accuracy": accuracy_score(yt, yp),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
        "f1": f1_score(yt, yp, zero_division=0),
        "brier": brier_score_loss(yt, ypr),
        "pr_auc": average_precision_score(yt, ypr),
    }


def bootstrap_metrics(y_true, y_prob, y_pred, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_boot)
    boot_rows = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_bootstrap_one)(int(s), y_true, y_prob, y_pred) for s in seeds
    )
    records = {m: [] for m in ["roc_auc", "accuracy", "sensitivity", "specificity", "f1", "brier", "pr_auc"]}
    for row in boot_rows:
        if row is None:
            continue
        for m, v in row.items():
            records[m].append(v)

    rows = []
    for m, vals in records.items():
        a = np.array(vals)
        rows.append({
            "metric": m,
            "point": np.nan,
            "ci_low": np.percentile(a, 2.5),
            "ci_high": np.percentile(a, 97.5),
            "mean_boot": a.mean(),
        })
    return pd.DataFrame(rows)


def run_pipeline(
    dataset: str, quick: bool = False, n_jobs: int = -1, use_gpu: bool = False
) -> None:
    gpu_cfg = init_gpu(use_gpu, quick=quick)
    nj = init_parallel(n_jobs)
    cpus = os.cpu_count() or 1
    prefix = "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"
    out_dir = pipeline_results_dir("ml", dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ML pipeline: {dataset} ===")
    print(gpu_cfg["message"])
    print(f"Parallel workers: {nj} (system CPUs: {cpus})")
    train, val, test = load_splits(prefix)

    feature_cols = get_feature_columns(train)
    X_fit_raw, y_fit, _ = prepare_xy(train, feature_cols, include_batch=False)
    X_val_raw, y_val_only, _ = prepare_xy(val, feature_cols, include_batch=False)
    X_te, y_te, _ = prepare_xy(test, feature_cols, include_batch=False)
    batches_fit = train[BATCH_COL].values
    batches_val = val[BATCH_COL].values
    batches_te = test[BATCH_COL].values

    # Batch correction: fit on train only; apply to val/test
    imp_final, gm_final, ym_final = fit_batch_correction(X_fit_raw, batches_fit)
    X_fit_bc = apply_batch_correction(
        X_fit_raw, batches_fit, imp_final, gm_final, ym_final
    )
    X_val_bc = apply_batch_correction(
        X_val_raw, batches_val, imp_final, gm_final, ym_final
    )
    X_test_bc = apply_batch_correction(
        X_te, batches_te, imp_final, gm_final, ym_final
    )

    # --- 1. Batch diagnostics (PCA by survey_year, train only) ---
    plot_batch_pca(
        X_fit_raw, y_fit, batches_fit,
        out_dir / "01_batch_pca_before.png",
        "PCA before batch correction (train, by survey_year)",
    )
    plot_batch_pca(
        X_fit_bc, y_fit, batches_fit,
        out_dir / "01_batch_pca_after.png",
        "PCA after year-mean correction (train-fitted)",
    )

    # --- 2. LASSO on train only + nested year-grouped CV (Comment #9) ---
    n_boot = 200 if quick else 1000
    n_splits = 3 if quick else 5
    outer_lasso = 3 if quick else 5
    inner_lasso = 2 if quick else 3
    use_quantile = False  # z-score primary (manuscript); set True for sensitivity

    C_grid = [0.01, 0.05, 0.1, 0.5] if quick else [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    print(
        f"Nested CV for LASSO on train only "
        f"(StratifiedGroupKFold by {BATCH_COL}, outer={outer_lasso}, inner={inner_lasso})..."
    )
    lasso_summary = lasso_nested_cv(
        X_fit_bc,
        y_fit,
        batches_fit,
        feature_cols,
        out_dir,
        C_grid=C_grid,
        outer_splits=outer_lasso,
        inner_splits=inner_lasso,
        max_features=30,
        stability_threshold=0.5,
    )
    selected = lasso_summary["stable_features"]
    (out_dir / "02_lasso_selected_features.txt").write_text(
        "\n".join(selected) + f"\n\n# {lasso_summary['selection_method']}\n"
        f"# nested outer AUC = {lasso_summary['outer_roc_auc_mean']:.4f} "
        f"+/- {lasso_summary['outer_roc_auc_std']:.4f}\n"
    )
    pd.DataFrame({"feature": selected}).to_csv(out_dir / "02_lasso_coefficients.csv", index=False)
    print(
        f"LASSO nested CV: outer AUC={lasso_summary['outer_roc_auc_mean']:.4f} "
        f"(+/-{lasso_summary['outer_roc_auc_std']:.4f}), "
        f"{len(selected)} stable features"
    )

    sel_idx = [feature_cols.index(f) for f in selected]
    X_fit_sel = X_fit_bc[:, sel_idx]
    X_val_sel = X_val_bc[:, sel_idx]
    X_test_sel = X_test_bc[:, sel_idx]

    # --- 3. Model comparison: year-grouped train CV + val ROC for selection ---
    cv_df = cv_evaluate_models(
        X_fit_sel, y_fit, batches_fit, selected, out_dir,
        use_smote=True, use_quantile=use_quantile, n_splits=n_splits,
    )
    best_model_name, val_df = select_best_model_on_val(
        X_fit_sel, y_fit, X_val_sel, y_val_only, use_quantile=use_quantile
    )
    val_df.to_csv(out_dir / "03b_val_model_selection.csv", index=False)
    print(f"Best model (val ROC-AUC): {best_model_name} ({val_df.iloc[0]['val_roc_auc']:.4f})")

    # --- 4. Ensembles ---
    base_clfs = get_classifiers()
    top3 = cv_df.head(3)["model"].tolist()
    estimators = [(n, clone(base_clfs[n])) for n in top3]
    vote = VotingClassifier(estimators=estimators, voting="soft")
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        passthrough=False, cv=3, n_jobs=N_JOBS,
    )
    ens_rows = []
    for ens_name, ens in [("voting_soft", vote), ("stacking", stack)]:
        pipe = make_pipeline(ens, use_smote=True, use_quantile=use_quantile, y=y_fit)
        cv = year_stratified_group_cv(n_splits)
        aucs = cross_validate(
            pipe, X_fit_sel, y_fit, groups=batches_fit, cv=cv,
            scoring={"auc": "roc_auc"}, n_jobs=N_JOBS,
        )
        ens_rows.append({"ensemble": ens_name, "roc_auc_mean": aucs["test_auc"].mean()})
    ens_df = pd.DataFrame(ens_rows)
    ens_df.to_csv(out_dir / "04_ensemble_comparison.csv", index=False)

    # --- 5. Final model: train only (no SMOTE), calibrate + threshold on val, test once ---
    final_clf = clone(base_clfs[best_model_name])
    final_pipe = make_pipeline(
        final_clf, use_smote=False, use_quantile=use_quantile, y=y_fit
    )
    final_pipe.fit(X_fit_sel, y_fit)

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(final_pipe), method="isotonic"
    )
    calibrated.fit(X_val_sel, y_val_only)

    y_prob_va = calibrated.predict_proba(X_val_sel)[:, 1]
    best_threshold, threshold_info = choose_classification_threshold(y_val_only, y_prob_va)

    y_prob_te = calibrated.predict_proba(X_test_sel)[:, 1]
    y_pred_te = (y_prob_te >= best_threshold).astype(int)
    cm = confusion_matrix(y_te, y_pred_te)
    if cm.shape == (2, 2) and (cm[0, 0] == 0 or cm[1, 1] == 0):
        warnings.warn(
            f"Degenerate confusion matrix on test (thr={best_threshold:.3f}); "
            "check calibration/threshold.",
            stacklevel=2,
        )
    print(
        f"Threshold (val, Youden J): {best_threshold:.3f} | "
        f"val P(min/med/max)=({y_prob_va.min():.3f}/{np.median(y_prob_va):.3f}/{y_prob_va.max():.3f}) | "
        f"test pred positive rate={y_pred_te.mean():.3f} (prevalence={y_te.mean():.3f}) | "
        f"test sens={cm[1,1]/max(cm[1].sum(),1):.3f} spec={cm[0,0]/max(cm[0].sum(),1):.3f}"
    )

    # Bootstrap CI on test (Comment #2)
    boot_df = bootstrap_metrics(y_te, y_prob_te, y_pred_te, n_boot=n_boot)
    for i, m in enumerate(["roc_auc", "accuracy", "f1", "brier", "pr_auc"]):
        if m == "roc_auc":
            boot_df.loc[boot_df["metric"] == m, "point"] = roc_auc_score(y_te, y_prob_te)
        elif m == "accuracy":
            boot_df.loc[boot_df["metric"] == m, "point"] = accuracy_score(y_te, y_pred_te)
        elif m == "f1":
            boot_df.loc[boot_df["metric"] == m, "point"] = f1_score(y_te, y_pred_te)
        elif m == "brier":
            boot_df.loc[boot_df["metric"] == m, "point"] = brier_score_loss(y_te, y_prob_te)
        elif m == "pr_auc":
            boot_df.loc[boot_df["metric"] == m, "point"] = average_precision_score(y_te, y_prob_te)
    boot_df.to_csv(out_dir / "05_test_metrics_bootstrap.csv", index=False)

    # Confusion matrix & ROC/PR
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title(f"Test confusion matrix (thr={best_threshold:.3f}, isotonic)")
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
    fpr, tpr, _ = roc_curve(y_te, y_prob_te)
    ax[1].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_te, y_prob_te):.3f}")
    ax[1].plot([0, 1], [0, 1], "k--")
    ax[1].set_title("ROC (test)"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    # Calibration
    prob_true, prob_pred = calibration_curve(y_te, y_prob_te, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted prob"); ax.set_ylabel("Fraction positive")
    ax.set_title("Calibration (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_calibration_test.png", dpi=150)
    plt.close(fig)

    # Per-year score violins (Comment #1 analogue)
    test_df = test.copy()
    test_df["score"] = y_prob_te
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.violinplot(data=test_df, x=BATCH_COL, y="score", hue=TARGET, split=True, ax=ax)
    ax.axhline(0.5, color="gray", ls="--")
    ax.set_title("Predicted P(stroke) by survey_year (test)")
    fig.tight_layout()
    fig.savefig(out_dir / "08_per_year_score_violin.png", dpi=150)
    plt.close(fig)

    # --- 6. Permutation importance (validation only — no test peeking) ---
    n_perm = 5 if quick else 15
    perm = permutation_importance(
        calibrated, X_val_sel, y_val_only, n_repeats=n_perm, random_state=42,
        n_jobs=N_JOBS, scoring="roc_auc",
    )
    perm_df = pd.DataFrame({
        "feature": selected,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(out_dir / "09_permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    top = perm_df.head(20)
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.invert_yaxis()
    ax.set_title("Permutation importance (validation, ROC-AUC)")
    fig.tight_layout()
    fig.savefig(out_dir / "09_permutation_importance.png", dpi=150)
    plt.close(fig)

    # --- 7. SHAP ---
    try:
        import shap

        # Use underlying classifier after preprocessing for tree/linear
        if best_model_name in (
            "random_forest", "gradient_boosting", "extra_trees",
            "decision_tree", "bag_tree", "bagging",
        ):
            pre = Pipeline(final_pipe.steps[:-2])  # through smote problematic for shap - refit without SMOTE
            pre = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            X_fit = pre.fit_transform(X_fit_sel)
            clf = clone(base_clfs[best_model_name])
            clf.fit(X_fit, y_fit)
            explainer = shap.TreeExplainer(clf)
            n_shap = min(2000, len(X_val_sel)) if not quick else min(500, len(X_val_sel))
            idx = np.random.choice(len(X_val_sel), n_shap, replace=False)
            shv = explainer.shap_values(pre.transform(X_val_sel[idx]))
            if isinstance(shv, list):
                shv = shv[1]
            fig = plt.figure()
            shap.summary_plot(shv, pre.transform(X_val_sel[idx]), feature_names=selected, show=False)
            plt.tight_layout()
            plt.savefig(out_dir / "10_shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
        else:
            (out_dir / "10_shap_summary.txt").write_text(
                f"SHAP skipped for {best_model_name}; use tree model or increase quick=False"
            )
    except Exception as e:
        (out_dir / "10_shap_error.txt").write_text(str(e))

    joblib.dump(calibrated, out_dir / "11_best_model.joblib")
    classifier_names = list(get_classifiers().keys())
    config = {
        "dataset": dataset,
        "n_classifiers_evaluated": len(classifier_names),
        "classifiers_evaluated": classifier_names,
        "selected_features": selected,
        "lasso_nested_cv": lasso_summary,
        "best_model": best_model_name,
        "classification_threshold": best_threshold,
        "threshold_tuning": threshold_info,
        "lasso_data": "train_only",
        "cv_scheme": "StratifiedGroupKFold_by_survey_year",
        "final_fit": "train_only",
        "final_smote": False,
        "probability_calibration": "isotonic_on_val",
        "interpretability_data": "validation_only",
        "n_train_fit": len(y_fit),
        "n_val_threshold": len(y_val_only),
        "n_test": len(y_te),
        "test_roc_auc": float(roc_auc_score(y_te, y_prob_te)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred_te)),
        "use_gpu_requested": use_gpu,
        "use_gpu_enabled": bool(gpu_cfg.get("enabled")),
        "gpu_backend": gpu_cfg.get("gradient_boosting_backend"),
    }
    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        raise TypeError(type(obj))

    (out_dir / "pipeline_config.json").write_text(
        json.dumps(config, indent=2, default=_json_default)
    )

    report = f"""# ML Pipeline Report — {dataset}

## Design
- Temporal split: train 2015–19 → val 2020–22 → test 2023–24 (test touched once).
- LASSO + model CV: **train only**, **StratifiedGroupKFold by survey_year** (no year mixing).
- Model choice: **validation ROC-AUC**; threshold: **validation Youden's J** (isotonic calibration).
- Final fit: train only, no SMOTE.
- LASSO outer AUC: {lasso_summary['outer_roc_auc_mean']:.4f} (+/- {lasso_summary['outer_roc_auc_std']:.4f})
- Stable features: {len(selected)} ({lasso_summary['selection_method']})
- Bootstrap {n_boot} reps for test metrics CI.
- Permutation importance + SHAP on **validation** (not test).

## Results
- Best model (val ROC-AUC): **{best_model_name}**
- Features after LASSO: {len(selected)}
- Test ROC-AUC: {config['test_roc_auc']:.4f}
- Test balanced accuracy @ val threshold: {config['test_balanced_accuracy']:.4f}
- Classification threshold (val): {best_threshold:.4f}

See CSV/PNG outputs in this folder.
"""
    (out_dir / "WORKFLOW_REPORT.md").write_text(report)
    print(f"Done. Outputs: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ny", "all_states", "both"], default="ny")
    parser.add_argument("--quick", action="store_true", help="Fewer bootstrap/SHAP/CV reps")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel CPU workers (-1 = all cores, default)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use CUDA XGBoost for gradient_boosting (requires xgboost + NVIDIA GPU)",
    )
    args = parser.parse_args()
    datasets = ["ny", "all_states"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_pipeline(
            ds, quick=args.quick, n_jobs=args.n_jobs, use_gpu=args.use_gpu
        )


if __name__ == "__main__":
    main()
