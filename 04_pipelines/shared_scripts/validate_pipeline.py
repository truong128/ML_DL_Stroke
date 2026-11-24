#!/usr/bin/env python3
"""Sanity checks before full ML pipeline run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ml_pipeline import PROCESSED, TARGET, run_pipeline  # noqa: E402


def check_dataset(dataset: str) -> list[str]:
    errors: list[str] = []
    out = ROOT / "results" / "ml" / dataset
    cfg_path = out / "pipeline_config.json"
    if not cfg_path.exists():
        return [f"{dataset}: missing pipeline_config.json"]

    cfg = json.loads(cfg_path.read_text())
    for key in (
        "lasso_data",
        "cv_scheme",
        "interpretability_data",
        "final_fit",
        "final_smote",
    ):
        if key not in cfg:
            errors.append(f"{dataset}: config missing {key}")

    if cfg.get("lasso_data") != "train_only":
        errors.append(f"{dataset}: lasso_data should be train_only")
    if cfg.get("cv_scheme") != "StratifiedGroupKFold_by_survey_year":
        errors.append(f"{dataset}: wrong cv_scheme")
    if cfg.get("interpretability_data") != "validation_only":
        errors.append(f"{dataset}: permutation/SHAP should use validation_only")
    if cfg.get("final_smote") is not False:
        errors.append(f"{dataset}: final_smote must be False")
    n_clf = cfg.get("n_classifiers_evaluated")
    if n_clf is not None and n_clf != 16:
        errors.append(f"{dataset}: expected 16 classifiers, got {n_clf}")

    prefix = "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"
    test = pd.read_csv(PROCESSED / f"{prefix}_test.csv", low_memory=False)
    model = joblib.load(out / "11_best_model.joblib")
    feats = cfg["selected_features"]
    X = test[feats].apply(pd.to_numeric, errors="coerce").values
    y = test[TARGET].astype(int).values
    thr = cfg["classification_threshold"]
    p = model.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    if cm.shape != (2, 2):
        errors.append(f"{dataset}: bad confusion matrix shape")
    elif cm[0, 0] == 0 or cm[1, 1] == 0:
        errors.append(f"{dataset}: degenerate CM TN={cm[0,0]} TP={cm[1,1]}")
    auc = roc_auc_score(y, p)
    if auc < 0.55:
        errors.append(f"{dataset}: suspiciously low ROC-AUC {auc:.3f}")
    if not (0.05 <= pred.mean() <= 0.95):
        errors.append(f"{dataset}: pred positive rate {pred.mean():.3f} out of range")

    print(
        f"  {dataset}: OK — AUC={auc:.3f}, thr={thr:.3f}, "
        f"pred+={pred.mean():.3f}, prev={y.mean():.3f}, CM=\n{cm}"
    )
    return errors


def main() -> int:
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["ny"]
    all_errors: list[str] = []
    for ds in datasets:
        all_errors.extend(check_dataset(ds))
    if all_errors:
        print("FAILED:")
        for e in all_errors:
            print(" -", e)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
