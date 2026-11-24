#!/usr/bin/env python3
"""
Compare locked test-set metrics across ML pipeline result folders.

Usage:
  python scripts/compare_pipelines.py --dataset ny
  python scripts/compare_pipelines.py --dataset both
  python scripts/compare_pipelines.py --dataset ny --pipelines ml,ml_voting_soft,tidymodels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ml_pipeline import PROCESSED, TARGET  # noqa: E402

PIPELINE_DIRS = {
    "ml": "results/ml",
    "ml_voting_soft": "results/ml_voting_soft",
    "tidymodels": "results/tidymodels",
    "tidymodels_single": "results/tidymodels_single",
}


def _dataset_prefix(dataset: str) -> str:
    return "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"


def _load_features(out_dir: Path) -> list[str]:
    txt = out_dir / "02_lasso_selected_features.txt"
    if txt.exists():
        return [
            ln.strip()
            for ln in txt.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    cfg = json.loads((out_dir / "pipeline_config.json").read_text())
    return list(cfg["selected_features"])


def _metrics_from_model(out_dir: Path, dataset: str) -> dict | None:
    """Recompute test metrics from saved sklearn model if present."""
    model_path = out_dir / "11_best_model.joblib"
    if not model_path.exists():
        return None
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    from score_cohort import build_feature_matrix  # noqa: WPS433

    cfg = json.loads((out_dir / "pipeline_config.json").read_text())
    selected = cfg.get("selected_features") or _load_features(out_dir)
    thr = float(cfg["classification_threshold"])
    prefix = _dataset_prefix(dataset)
    train = pd.read_csv(PROCESSED / f"{prefix}_train.csv", low_memory=False)
    test = pd.read_csv(PROCESSED / f"{prefix}_test.csv", low_memory=False)
    X = build_feature_matrix(train, test, selected)
    y = test[TARGET].astype(int).values
    model = joblib.load(model_path)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    return {
        "test_roc_auc": float(roc_auc_score(y, p)),
        "test_pr_auc": float(average_precision_score(y, p)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "test_accuracy": float(accuracy_score(y, pred)),
        "test_f1": float(f1_score(y, pred, zero_division=0)),
        "test_brier": float(brier_score_loss(y, p)),
        "test_pred_positive_rate": float(pred.mean()),
        "test_prevalence": float(y.mean()),
        "classification_threshold": thr,
        "n_test": int(len(y)),
    }


def _load_pipeline_row(pipeline: str, dataset: str) -> dict | None:
    out_dir = ROOT / PIPELINE_DIRS[pipeline] / dataset
    cfg_path = out_dir / "pipeline_config.json"
    if not cfg_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text())
    row = {
        "pipeline": pipeline,
        "dataset": dataset,
        "final_model": cfg.get("best_model") or cfg.get("final_model", ""),
        "voting_members": cfg.get("voting_members", ""),
        "n_classifiers": cfg.get("n_classifiers_evaluated", ""),
        "n_train": cfg.get("n_train_fit", ""),
        "n_val": cfg.get("n_val_threshold", ""),
        "n_test": cfg.get("n_test", ""),
        "test_roc_auc_config": cfg.get("test_roc_auc", np.nan),
        "test_balanced_accuracy_config": cfg.get("test_balanced_accuracy", np.nan),
        "classification_threshold": cfg.get("classification_threshold", np.nan),
        "val_roc_auc_single": cfg.get("val_roc_auc_single", np.nan),
        "val_roc_auc_voting_soft": cfg.get("val_roc_auc_voting_soft", np.nan),
    }

    boot_path = out_dir / "05_test_metrics_bootstrap.csv"
    if boot_path.exists():
        boot = pd.read_csv(boot_path)
        for _, r in boot.iterrows():
            m = r["metric"]
            row[f"{m}_point"] = r.get("point", np.nan)
            row[f"{m}_ci_low"] = r.get("ci_low", np.nan)
            row[f"{m}_ci_high"] = r.get("ci_high", np.nan)

    recomputed = _metrics_from_model(out_dir, dataset)
    if recomputed:
        for k, v in recomputed.items():
            row[f"{k}_recomputed"] = v

    row["has_joblib_model"] = (out_dir / "11_best_model.joblib").exists()
    row["has_rds_model"] = (out_dir / "11_best_model.rds").exists()
    row["output_dir"] = str(out_dir)
    return row


def compare(
    datasets: list[str],
    pipelines: list[str],
) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for pipe in pipelines:
            if pipe not in PIPELINE_DIRS:
                raise ValueError(f"Unknown pipeline: {pipe}. Choose from {list(PIPELINE_DIRS)}")
            row = _load_pipeline_row(pipe, ds)
            if row is None:
                print(f"  [skip] {pipe}/{ds}: no pipeline_config.json", file=sys.stderr)
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ML pipeline test metrics")
    parser.add_argument("--dataset", choices=["ny", "all_states", "both"], default="both")
    parser.add_argument(
        "--pipelines",
        default="ml,ml_voting_soft,tidymodels,tidymodels_single",
        help="Comma-separated: ml, ml_voting_soft, tidymodels, tidymodels_single",
    )
    args = parser.parse_args()
    datasets = ["ny", "all_states"] if args.dataset == "both" else [args.dataset]
    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]

    df = compare(datasets, pipelines)
    if df.empty:
        print("No pipeline outputs found to compare.", file=sys.stderr)
        return 1

    out_dir = ROOT / "results" / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"pipeline_comparison_{args.dataset}.csv"
    df.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}\n")
    cols_show = [
        "pipeline", "dataset", "final_model", "test_roc_auc_config",
        "test_balanced_accuracy_config", "roc_auc_point", "has_joblib_model", "has_rds_model",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    print(df[cols_show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
