#!/usr/bin/env python3
"""
BRFSS stroke ML pipeline — final model = soft-voting ensemble (top-3 by validation ROC-AUC).

Outputs: results/ml_voting_soft/<dataset>/  (does not touch results/ml/)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.frozen import FrozenEstimator
import joblib

# Reuse core pipeline from scripts/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import run_ml_pipeline as base  # noqa: E402

RESULTS_SUBDIR = "ml_voting_soft"
VOTING_TOP_K = 3


def _eval_val_auc(pipe, X_val, y_val) -> float:
    return float(roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1]))


def run_pipeline(
    dataset: str, quick: bool = False, n_jobs: int = -1, use_gpu: bool = False
) -> None:
    gpu_cfg = base.init_gpu(use_gpu, quick=quick)
    nj = base.init_parallel(n_jobs)
    cpus = __import__("os").cpu_count() or 1
    prefix = "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"
    out_dir = base.pipeline_results_dir(RESULTS_SUBDIR, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ML pipeline (voting_soft): {dataset} ===")
    print(f"Outputs: {out_dir}")
    print(gpu_cfg["message"])
    print(f"Parallel workers: {nj} (system CPUs: {cpus})")

    train, val, test = base.load_splits(prefix)
    feature_cols = base.get_feature_columns(train)
    X_fit_raw, y_fit, _ = base.prepare_xy(train, feature_cols, include_batch=False)
    X_val_raw, y_val_only, _ = base.prepare_xy(val, feature_cols, include_batch=False)
    X_te, y_te, _ = base.prepare_xy(test, feature_cols, include_batch=False)
    batches_fit = train[base.BATCH_COL].values
    batches_val = val[base.BATCH_COL].values
    batches_te = test[base.BATCH_COL].values

    imp_final, gm_final, ym_final = base.fit_batch_correction(X_fit_raw, batches_fit)
    X_fit_bc = base.apply_batch_correction(
        X_fit_raw, batches_fit, imp_final, gm_final, ym_final
    )
    X_val_bc = base.apply_batch_correction(
        X_val_raw, batches_val, imp_final, gm_final, ym_final
    )
    X_test_bc = base.apply_batch_correction(
        X_te, batches_te, imp_final, gm_final, ym_final
    )

    base.plot_batch_pca(
        X_fit_raw, y_fit, batches_fit,
        out_dir / "01_batch_pca_before.png",
        "PCA before batch correction (train, by survey_year)",
    )
    base.plot_batch_pca(
        X_fit_bc, y_fit, batches_fit,
        out_dir / "01_batch_pca_after.png",
        "PCA after year-mean correction (train-fitted)",
    )

    n_boot = 200 if quick else 1000
    n_splits = 3 if quick else 5
    outer_lasso = 3 if quick else 5
    inner_lasso = 2 if quick else 3
    use_quantile = False
    C_grid = [0.01, 0.05, 0.1, 0.5] if quick else [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    print(
        f"Nested CV for LASSO on train only "
        f"(StratifiedGroupKFold by {base.BATCH_COL}, outer={outer_lasso}, inner={inner_lasso})..."
    )
    lasso_summary = base.lasso_nested_cv(
        X_fit_bc, y_fit, batches_fit, feature_cols, out_dir,
        C_grid=C_grid, outer_splits=outer_lasso, inner_splits=inner_lasso,
        max_features=30, stability_threshold=0.5,
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
        f"(+/-{lasso_summary['outer_roc_auc_std']:.4f}), {len(selected)} stable features"
    )

    sel_idx = [feature_cols.index(f) for f in selected]
    X_fit_sel = X_fit_bc[:, sel_idx]
    X_val_sel = X_val_bc[:, sel_idx]
    X_test_sel = X_test_bc[:, sel_idx]

    base_clfs = base.get_classifiers()
    classifier_names = list(base_clfs.keys())
    print(f"Evaluating {len(classifier_names)} classifiers: {', '.join(classifier_names)}")

    cv_df = base.cv_evaluate_models(
        X_fit_sel, y_fit, batches_fit, selected, out_dir,
        use_smote=True, use_quantile=use_quantile, n_splits=n_splits,
    )
    best_single_name, val_df = base.select_best_model_on_val(
        X_fit_sel, y_fit, X_val_sel, y_val_only, use_quantile=use_quantile
    )
    val_df.to_csv(out_dir / "03b_val_model_selection.csv", index=False)
    (out_dir / "03c_classifiers_evaluated.txt").write_text(
        f"n_classifiers={len(classifier_names)}\n" + "\n".join(classifier_names) + "\n"
    )
    print(f"Best single model (val ROC-AUC): {best_single_name} ({val_df.iloc[0]['val_roc_auc']:.4f})")
    top3_val = val_df.head(VOTING_TOP_K)["model"].tolist()
    pd.DataFrame({
        "rank": range(1, len(top3_val) + 1),
        "model": top3_val,
        "val_roc_auc": [float(val_df.loc[val_df["model"] == m, "val_roc_auc"].iloc[0]) for m in top3_val],
    }).to_csv(out_dir / "04b_voting_soft_members.csv", index=False)

    estimators = [(n, clone(base_clfs[n])) for n in top3_val]
    vote = VotingClassifier(estimators=estimators, voting="soft", n_jobs=base.N_JOBS)

    # Train CV benchmark (same as standard pipeline, top-3 from train CV for reference)
    top3_train = cv_df.head(VOTING_TOP_K)["model"].tolist()
    vote_train_top3 = VotingClassifier(
        estimators=[(n, clone(base_clfs[n])) for n in top3_train],
        voting="soft",
        n_jobs=base.N_JOBS,
    )
    ens_rows = []
    stack = StackingClassifier(
        estimators=[(n, clone(base_clfs[n])) for n in top3_val],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        passthrough=False, cv=3, n_jobs=base.N_JOBS,
    )
    for label, ens in [
        ("voting_soft_val_top3", vote),
        ("voting_soft_traincv_top3", vote_train_top3),
        ("stacking_val_top3", stack),
    ]:
        pipe_cv = base.make_pipeline(ens, use_smote=True, use_quantile=use_quantile, y=y_fit)
        cv = base.year_stratified_group_cv(n_splits)
        aucs = cross_validate(
            pipe_cv, X_fit_sel, y_fit, groups=batches_fit, cv=cv,
            scoring={"auc": "roc_auc"}, n_jobs=base.N_JOBS,
        )
        ens_rows.append({"ensemble": label, "roc_auc_mean_train_cv": aucs["test_auc"].mean()})
    pd.DataFrame(ens_rows).to_csv(out_dir / "04_ensemble_comparison.csv", index=False)

    # Val comparison: best single vs voting_soft (members = val top-3)
    single_pipe = base.make_pipeline(
        clone(base_clfs[best_single_name]), use_smote=False, use_quantile=use_quantile, y=y_fit
    )
    single_pipe.fit(X_fit_sel, y_fit)
    vote_pipe = base.make_pipeline(vote, use_smote=False, use_quantile=use_quantile, y=y_fit)
    vote_pipe.fit(X_fit_sel, y_fit)
    val_auc_single = _eval_val_auc(single_pipe, X_val_sel, y_val_only)
    val_auc_vote = _eval_val_auc(vote_pipe, X_val_sel, y_val_only)
    pd.DataFrame([
        {"model": best_single_name, "val_roc_auc": val_auc_single, "role": "best_single"},
        {"model": "voting_soft", "val_roc_auc": val_auc_vote, "role": "final_ensemble",
         "members": ",".join(top3_val)},
    ]).to_csv(out_dir / "04c_val_final_model_comparison.csv", index=False)
    print(
        f"Val ROC-AUC — single ({best_single_name}): {val_auc_single:.4f} | "
        f"voting_soft ({', '.join(top3_val)}): {val_auc_vote:.4f}"
    )

    # --- Final: voting_soft, calibrate on val, test once ---
    final_pipe = vote_pipe
    calibrated = CalibratedClassifierCV(FrozenEstimator(final_pipe), method="isotonic")
    calibrated.fit(X_val_sel, y_val_only)

    y_prob_va = calibrated.predict_proba(X_val_sel)[:, 1]
    best_threshold, threshold_info = base.choose_classification_threshold(y_val_only, y_prob_va)

    y_prob_te = calibrated.predict_proba(X_test_sel)[:, 1]
    y_pred_te = (y_prob_te >= best_threshold).astype(int)
    cm = confusion_matrix(y_te, y_pred_te)
    if cm.shape == (2, 2) and (cm[0, 0] == 0 or cm[1, 1] == 0):
        warnings.warn(
            f"Degenerate confusion matrix on test (thr={best_threshold:.3f})",
            stacklevel=2,
        )
    print(
        f"Threshold (val, Youden J): {best_threshold:.3f} | "
        f"test pred+={y_pred_te.mean():.3f} (prev={y_te.mean():.3f}) | "
        f"sens={cm[1,1]/max(cm[1].sum(),1):.3f} spec={cm[0,0]/max(cm[0].sum(),1):.3f}"
    )

    boot_df = base.bootstrap_metrics(y_te, y_prob_te, y_pred_te, n_boot=n_boot)
    for m, fn in [
        ("roc_auc", lambda: roc_auc_score(y_te, y_prob_te)),
        ("accuracy", lambda: accuracy_score(y_te, y_pred_te)),
        ("f1", lambda: f1_score(y_te, y_pred_te)),
        ("brier", lambda: brier_score_loss(y_te, y_prob_te)),
        ("pr_auc", lambda: average_precision_score(y_te, y_prob_te)),
    ]:
        boot_df.loc[boot_df["metric"] == m, "point"] = fn()
    boot_df.to_csv(out_dir / "05_test_metrics_bootstrap.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title(f"Test CM — voting_soft (thr={best_threshold:.3f})")
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
    fpr, tpr, _ = roc_curve(y_te, y_prob_te)
    ax[1].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_te, y_prob_te):.3f}")
    ax[1].plot([0, 1], [0, 1], "k--")
    ax[1].set_title("ROC (test)"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    prob_true, prob_pred = calibration_curve(y_te, y_prob_te, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, "s-", label="voting_soft")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted prob"); ax.set_ylabel("Fraction positive")
    ax.set_title("Calibration (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_calibration_test.png", dpi=150)
    plt.close(fig)

    test_df = test.copy()
    test_df["score"] = y_prob_te
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.violinplot(data=test_df, x=base.BATCH_COL, y="score", hue=base.TARGET, split=True, ax=ax)
    ax.axhline(best_threshold, color="gray", ls="--")
    ax.set_title("Predicted P(stroke) by survey_year (test)")
    fig.tight_layout()
    fig.savefig(out_dir / "08_per_year_score_violin.png", dpi=150)
    plt.close(fig)

    n_perm = 5 if quick else 15
    perm = permutation_importance(
        calibrated, X_val_sel, y_val_only, n_repeats=n_perm, random_state=42,
        n_jobs=base.N_JOBS, scoring="roc_auc",
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
    ax.set_title("Permutation importance (val, voting_soft ensemble)")
    fig.tight_layout()
    fig.savefig(out_dir / "09_permutation_importance.png", dpi=150)
    plt.close(fig)

    (out_dir / "10_shap_summary.txt").write_text(
        "SHAP not computed for voting_soft ensemble. "
        f"See single-model SHAP in results/ml/{dataset}/ or run SHAP per member: {top3_val}"
    )

    joblib.dump(calibrated, out_dir / "11_best_model.joblib")
    config = {
        "dataset": dataset,
        "pipeline_variant": "voting_soft",
        "n_classifiers_evaluated": len(classifier_names),
        "classifiers_evaluated": classifier_names,
        "selected_features": selected,
        "lasso_nested_cv": lasso_summary,
        "final_model": "voting_soft",
        "voting_members": top3_val,
        "best_single_model": best_single_name,
        "val_roc_auc_single": val_auc_single,
        "val_roc_auc_voting_soft": val_auc_vote,
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
    }

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        raise TypeError(type(obj))

    (out_dir / "pipeline_config.json").write_text(
        json.dumps(config, indent=2, default=_json_default)
    )

    report = f"""# ML Pipeline Report — {dataset} (voting_soft)

## Final model
- **Soft voting** over: {', '.join(top3_val)} (top {VOTING_TOP_K} by validation ROC-AUC)
- Val ROC-AUC: single **{best_single_name}** = {val_auc_single:.4f}; **voting_soft** = {val_auc_vote:.4f}

## Test (2023–2024, one-time)
- ROC-AUC: {config['test_roc_auc']:.4f}
- Balanced accuracy: {config['test_balanced_accuracy']:.4f}
- Threshold (val): {best_threshold:.4f}

Outputs: `{out_dir}` (does not overwrite `results/ml/`).
"""
    (out_dir / "WORKFLOW_REPORT.md").write_text(report)
    print(f"Done. Outputs: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stroke ML pipeline — voting_soft final model")
    parser.add_argument("--dataset", choices=["ny", "all_states", "both"], default="ny")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="CUDA XGBoost for gradient_boosting (see scripts/run_ml_pipeline.py)",
    )
    args = parser.parse_args()
    datasets = ["ny", "all_states"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_pipeline(
            ds, quick=args.quick, n_jobs=args.n_jobs, use_gpu=args.use_gpu
        )


if __name__ == "__main__":
    main()
