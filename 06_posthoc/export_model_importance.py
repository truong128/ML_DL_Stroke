#!/usr/bin/env python3
"""
Export tree-model feature importance and mean |SHAP| from locked 11_best_model.joblib.

Outputs (per results folder):
  12_tree_feature_importance.csv   — sklearn feature_importances_ when available
  12_shap_mean_abs.csv             — mean |SHAP| on validation (positive class)
  12_shap_summary.png              — optional beeswarm (if matplotlib + shap)

For VotingClassifier, reports per-member importances and an unweighted average.
HistGradientBoosting has no feature_importances_; SHAP is the main model-based ranking.

Usage:
  python scripts/export_model_importance.py --dataset both
  python scripts/export_model_importance.py --dataset ny --pipeline ml
  python scripts/export_model_importance.py --results-dir results/ml/all_states --shap-max-rows 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ml_pipeline import (  # noqa: E402
    BATCH_COL,
    apply_batch_correction,
    fit_batch_correction,
    get_feature_columns,
    load_splits,
    prepare_xy,
)

PIPELINE_ROOTS = {
    "ml": ROOT / "results" / "ml",
    "voting_soft": ROOT / "results" / "ml_voting_soft",
}


def _dataset_prefix(dataset: str) -> str:
    return "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"


def _unwrap_fitted_pipeline(calibrated) -> Pipeline:
    """FrozenEstimator inside CalibratedClassifierCV → fitted imputer/scaler/clf."""
    cc = calibrated.calibrated_classifiers_[0]
    return cc.estimator


def _build_val_matrix(dataset: str, selected: list[str]) -> tuple[np.ndarray, np.ndarray]:
    prefix = _dataset_prefix(dataset)
    train_df, val_df, _ = load_splits(prefix)
    feature_cols = get_feature_columns(train_df)
    X_train_raw, _, _ = prepare_xy(train_df, feature_cols, include_batch=False)
    X_val_raw, y_val, _ = prepare_xy(val_df, feature_cols, include_batch=False)
    imp, gm, ym = fit_batch_correction(X_train_raw, train_df[BATCH_COL].values)
    X_val_bc = apply_batch_correction(
        X_val_raw, val_df[BATCH_COL].values, imp, gm, ym
    )
    sel_idx = [feature_cols.index(f) for f in selected]
    return X_val_bc[:, sel_idx], y_val


def _preprocessed_train_val(
    dataset: str, selected: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = _dataset_prefix(dataset)
    train_df, val_df, _ = load_splits(prefix)
    tr = train_df[train_df["split"] == "train"]
    feature_cols = get_feature_columns(train_df)
    X_train_raw, y_fit, _ = prepare_xy(tr, feature_cols, include_batch=False)
    X_val_raw, y_val, _ = prepare_xy(val_df, feature_cols, include_batch=False)
    imp, gm, ym = fit_batch_correction(X_train_raw, tr[BATCH_COL].values)
    X_fit = apply_batch_correction(X_train_raw, tr[BATCH_COL].values, imp, gm, ym)
    X_val = apply_batch_correction(X_val_raw, val_df[BATCH_COL].values, imp, gm, ym)
    sel_idx = [feature_cols.index(f) for f in selected]
    return X_fit[:, sel_idx], y_fit, X_val[:, sel_idx], y_val


def export_tree_importance(
    clf, features: list[str], model_label: str
) -> pd.DataFrame | None:
    rows: list[dict] = []

    if isinstance(clf, VotingClassifier):
        for name, est in zip(clf.named_estimators_.keys(), clf.estimators_):
            if not hasattr(est, "feature_importances_"):
                continue
            imp = np.asarray(est.feature_importances_, dtype=float)
            total = imp.sum()
            for feat, v in zip(features, imp):
                rows.append({
                    "feature": feat,
                    "importance": v,
                    "importance_normalized": v / total if total > 0 else np.nan,
                    "source": f"{model_label}:{name}",
                })
        if not rows:
            return None
        wide = pd.DataFrame(rows)
        avg = (
            wide.groupby("feature", as_index=False)["importance"]
            .mean()
            .rename(columns={"importance": "importance_mean_members"})
        )
        avg["importance_normalized"] = (
            avg["importance_mean_members"] / avg["importance_mean_members"].sum()
        )
        avg["source"] = f"{model_label}:voting_mean"
        member = wide.pivot_table(
            index="feature", columns="source", values="importance_normalized"
        ).reset_index()
        return avg.merge(member, on="feature", how="left")

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
        total = imp.sum()
        for feat, v in zip(features, imp):
            rows.append({
                "feature": feat,
                "importance": v,
                "importance_normalized": v / total if total > 0 else np.nan,
                "source": model_label,
            })
        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    return None


def _shap_for_fitted_classifier(
    clf_fit,
    features: list[str],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    *,
    max_rows: int,
    random_state: int,
) -> tuple[np.ndarray, str]:
    """Return mean |SHAP| vector and method note."""
    import shap

    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Xf = pre.fit_transform(X_fit)
    n = min(max_rows, len(X_val))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_val), size=n, replace=False)
    X_sample = pre.transform(X_val[idx])

    clf_fit = clone(clf_fit)
    clf_fit.fit(Xf, y_fit)
    shap_note_extra = ""

    def predict_positive(x: np.ndarray) -> np.ndarray:
        return clf_fit.predict_proba(x)[:, 1]

    try:
        explainer = shap.Explainer(clf_fit, X_sample)
        sv = explainer(X_sample)
    except Exception as e1:
        try:
            explainer = shap.TreeExplainer(clf_fit)
            sv = explainer(X_sample)
        except Exception as e2:
            n_perm = min(300, len(X_sample))
            X_perm = X_sample[:n_perm]
            explainer = shap.PermutationExplainer(
                predict_positive, X_perm, seed=random_state
            )
            sv = explainer(X_perm)
            shap_note_extra = f"; PermutationExplainer n={n_perm}"

    vals = np.asarray(sv.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    note = f"n={vals.shape[0]}{shap_note_extra}"
    return np.abs(vals).mean(axis=0), note


def export_shap(
    clf,
    features: list[str],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    *,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame | None, str]:
    try:
        import shap  # noqa: F401
    except ImportError:
        return None, "shap not installed"

    if isinstance(clf, VotingClassifier):
        member_shap: list[np.ndarray] = []
        notes: list[str] = []
        for name, est in clf.named_estimators_.items():
            if "hist_gradient" in name.lower() or "gradient_boosting" in name.lower():
                if "HistGradient" in type(est).__name__:
                    continue
            try:
                vec, note = _shap_for_fitted_classifier(
                    est, features, X_fit, y_fit, X_val,
                    max_rows=max_rows, random_state=random_state,
                )
                member_shap.append(vec)
                notes.append(f"{name}({note})")
            except Exception as e:
                notes.append(f"{name}(failed:{e})")
        if not member_shap:
            return None, "VotingClassifier: no member supported SHAP"
        mean_abs = np.mean(np.vstack(member_shap), axis=0)
        note = "SHAP mean across voting members: " + "; ".join(notes)
    else:
        mean_abs, sub = _shap_for_fitted_classifier(
            clf, features, X_fit, y_fit, X_val,
            max_rows=max_rows, random_state=random_state,
        )
        note = f"SHAP on validation ({sub}; median impute + scale; refit on train)"

    total = mean_abs.sum()
    df = pd.DataFrame({
        "feature": features,
        "shap_mean_abs": mean_abs,
        "shap_mean_abs_normalized": mean_abs / total if total > 0 else np.nan,
    }).sort_values("shap_mean_abs", ascending=False)
    return df, note


def process_results_dir(
    results_dir: Path,
    dataset: str,
    *,
    shap_max_rows: int,
    save_plot: bool,
) -> None:
    cfg_path = results_dir / "pipeline_config.json"
    model_path = results_dir / "11_best_model.joblib"
    if not cfg_path.exists() or not model_path.exists():
        print(f"  skip (missing config or joblib): {results_dir}", file=sys.stderr)
        return

    cfg = json.loads(cfg_path.read_text())
    features = list(cfg["selected_features"])
    best = cfg.get("best_model") or cfg.get("final_model", "unknown")
    calibrated = joblib.load(model_path)
    pipe = _unwrap_fitted_pipeline(calibrated)
    clf = pipe.named_steps["clf"]

    print(f"\n=== {results_dir} | model={best} | n_features={len(features)} ===")

    tree_df = export_tree_importance(clf, features, best)
    if tree_df is not None:
        out_tree = results_dir / "12_tree_feature_importance.csv"
        tree_df.to_csv(out_tree, index=False)
        print(f"  wrote {out_tree.name} ({len(tree_df)} rows)")
        sort_col = "importance" if "importance" in tree_df.columns else "importance_mean_members"
        top = tree_df.sort_values(sort_col, ascending=False).head(5)["feature"].tolist()
        print(f"  top tree importance: {top}")
    else:
        msg = (
            "HistGradientBoosting or ensemble without feature_importances_; "
            "see SHAP file instead."
        )
        (results_dir / "12_tree_importance_note.txt").write_text(msg + "\n")
        print(f"  no sklearn feature_importances_ ({msg})")

    X_fit, y_fit, X_val, _ = _preprocessed_train_val(dataset, features)
    shap_df, note = export_shap(
        clf, features, X_fit, y_fit, X_val,
        max_rows=shap_max_rows,
        random_state=42,
    )
    if shap_df is not None:
        out_shap = results_dir / "12_shap_mean_abs.csv"
        shap_df.to_csv(out_shap, index=False)
        meta = results_dir / "12_shap_meta.json"
        meta.write_text(json.dumps({
            "model": best,
            "dataset": dataset,
            "note": note,
            "n_features": len(features),
        }, indent=2))
        print(f"  wrote {out_shap.name} — {note}")
        print(f"  top SHAP: {shap_df.head(5)['feature'].tolist()}")

        if save_plot:
            try:
                import matplotlib.pyplot as plt
                import shap

                pre = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])
                Xf = pre.fit_transform(X_fit)
                n = min(shap_max_rows, len(X_val))
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_val), size=n, replace=False)
                X_sample = pre.transform(X_val[idx])
                clf_fit = clone(clf)
                clf_fit.fit(Xf, y_fit)
                explainer = shap.Explainer(clf_fit, X_sample)
                sv = explainer(X_sample[: min(500, len(X_sample))])
                vals = np.asarray(sv.values)
                if vals.ndim == 3:
                    vals = vals[:, :, 1]
                plt.figure()
                shap.summary_plot(
                    vals, X_sample[: vals.shape[0]], feature_names=features, show=False
                )
                plt.tight_layout()
                plt.savefig(results_dir / "12_shap_summary.png", dpi=150, bbox_inches="tight")
                plt.close()
                print("  wrote 12_shap_summary.png")
            except Exception as e:
                (results_dir / "12_shap_plot_error.txt").write_text(str(e))
                print(f"  SHAP plot skipped: {e}", file=sys.stderr)
    else:
        (results_dir / "12_shap_mean_abs.csv").write_text(
            f"# SHAP not computed\n# {note}\n"
        )
        print(f"  SHAP skipped: {note}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export tree importance + SHAP from joblib models")
    ap.add_argument("--dataset", choices=["ny", "all_states", "both"], default="both")
    ap.add_argument(
        "--pipeline", choices=["ml", "voting_soft", "both"], default="both",
        help="Which results root to scan",
    )
    ap.add_argument("--results-dir", type=Path, default=None, help="Single folder override")
    ap.add_argument("--shap-max-rows", type=int, default=1500, help="Validation rows for SHAP")
    ap.add_argument("--no-plot", action="store_true", help="Skip 12_shap_summary.png")
    args = ap.parse_args()

    if args.results_dir is not None:
        ds = args.dataset if args.dataset != "both" else "all_states"
        if "ny" in str(args.results_dir):
            ds = "ny"
        process_results_dir(
            args.results_dir.resolve(),
            ds,
            shap_max_rows=args.shap_max_rows,
            save_plot=not args.no_plot,
        )
        return

    datasets = ["ny", "all_states"] if args.dataset == "both" else [args.dataset]
    pipelines = ["ml", "voting_soft"] if args.pipeline == "both" else [args.pipeline]

    for pipe_name in pipelines:
        root = PIPELINE_ROOTS[pipe_name]
        for ds in datasets:
            process_results_dir(
                root / ds,
                ds,
                shap_max_rows=args.shap_max_rows,
                save_plot=not args.no_plot,
            )


if __name__ == "__main__":
    main()
