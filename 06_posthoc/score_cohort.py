#!/usr/bin/env python3
"""
Batch-score train/val/test (or combined) with the locked final calibrated model.

Loads 11_best_model.joblib and applies the same train-fitted batch correction +
LASSO feature subset as run_ml_pipeline.py.

Usage:
  python scripts/score_cohort.py --dataset ny --split test
  python scripts/score_cohort.py --dataset all_states --split all --pipeline voting_soft
  python scripts/score_cohort.py --dataset ny --model results/ml/ny/11_best_model.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ml_pipeline import (  # noqa: E402
    BATCH_COL,
    META_COLS,
    PROCESSED,
    TARGET,
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


def load_model_config(results_dir: Path) -> tuple[Path, dict]:
    cfg_path = results_dir / "pipeline_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    model_path = results_dir / "11_best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Python scoring requires joblib; "
            "R tidymodels outputs 11_best_model.rds — score in R or re-export."
        )
    return model_path, cfg


def build_feature_matrix(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    selected: list[str],
) -> np.ndarray:
    """Train-fitted median impute + year-mean correction; LASSO-selected columns only."""
    feature_cols = get_feature_columns(train_df)
    X_train_raw, _, _ = prepare_xy(train_df, feature_cols, include_batch=False)
    X_score_raw, _, _ = prepare_xy(score_df, feature_cols, include_batch=False)
    batches_train = train_df[BATCH_COL].values
    batches_score = score_df[BATCH_COL].values
    imp, gm, ym = fit_batch_correction(X_train_raw, batches_train)
    X_score_bc = apply_batch_correction(X_score_raw, batches_score, imp, gm, ym)
    sel_idx = [feature_cols.index(f) for f in selected]
    return X_score_bc[:, sel_idx]


def score_dataframe(
    model,
    X: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return prob, pred


def score_split(
    *,
    dataset: str,
    split: str,
    results_dir: Path,
    model_path: Path | None,
    keep_meta: bool,
) -> pd.DataFrame:
    prefix = _dataset_prefix(dataset)
    train, val, test = load_splits(prefix)
    _, cfg = load_model_config(results_dir)
    selected = list(cfg["selected_features"])
    threshold = float(cfg["classification_threshold"])
    model = joblib.load(model_path or (results_dir / "11_best_model.joblib"))

    if split == "all":
        score_df = pd.concat([train, val, test], ignore_index=True)
        score_df["cohort_split"] = (
            ["train"] * len(train) + ["val"] * len(val) + ["test"] * len(test)
        )
    else:
        score_df = {"train": train, "val": val, "test": test}[split].copy()
        score_df["cohort_split"] = split

    X = build_feature_matrix(train, score_df, selected)
    prob, pred = score_dataframe(model, X, threshold)

    out = score_df.copy()
    out[f"{TARGET}_prob"] = prob
    out[f"{TARGET}_pred"] = pred
    out["classification_threshold"] = threshold

    if not keep_meta:
        keep = [TARGET, f"{TARGET}_prob", f"{TARGET}_pred", "classification_threshold", "cohort_split"]
        optional_id = [c for c in ("seqno", "survey_year", "state_fips") if c in out.columns]
        out = out[optional_id + keep]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Score cohort with locked ML model")
    parser.add_argument("--dataset", choices=["ny", "all_states"], required=True)
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="test",
    )
    parser.add_argument(
        "--pipeline", choices=list(PIPELINE_ROOTS), default="ml",
        help="Which results tree (ml or voting_soft)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override e.g. results/ml/ny",
    )
    parser.add_argument("--model", type=Path, default=None, help="Path to 11_best_model.joblib")
    parser.add_argument(
        "--keep-meta",
        action="store_true",
        help="Keep all original columns in output CSV",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    results_dir = args.results_dir or (PIPELINE_ROOTS[args.pipeline] / args.dataset)
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}", file=sys.stderr)
        return 1

    scored = score_split(
        dataset=args.dataset,
        split=args.split,
        results_dir=results_dir,
        model_path=args.model,
        keep_meta=args.keep_meta,
    )

    out_path = args.out or (
        ROOT / "results" / "posthoc" / f"scored_{args.dataset}_{args.split}_{args.pipeline}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(f"Wrote {len(scored)} rows to {out_path}")
    print(
        f"  prevalence={scored[TARGET].mean():.4f} "
        f"pred_rate={scored[f'{TARGET}_pred'].mean():.4f} "
        f"mean_prob={scored[f'{TARGET}_prob'].mean():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
