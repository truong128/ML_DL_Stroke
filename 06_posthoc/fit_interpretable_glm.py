#!/usr/bin/env python3
"""
Interpretable binomial GLM on LASSO-selected features (epidemiologic inference layer).

Fits on train by default (not val/test). Uses median imputation only — no batch correction
or SMOTE — so odds ratios are standard epi interpretation (differs from ML preprocessing).

Usage:
  python scripts/fit_interpretable_glm.py --dataset ny
  python scripts/fit_interpretable_glm.py --dataset all_states --weighted
  python scripts/fit_interpretable_glm.py --dataset ny --features-from results/ml/ny
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ml_pipeline import PROCESSED, TARGET  # noqa: E402
from stats_fdr import add_fdr_to_or_table  # noqa: E402

WEIGHT_COL = "survey_weight"


def _dataset_prefix(dataset: str) -> str:
    return "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"


def load_features(features_from: Path | None, dataset: str) -> list[str]:
    if features_from is not None:
        base = Path(features_from)
        if base.is_dir():
            txt = base / "02_lasso_selected_features.txt"
        else:
            txt = base
        if not txt.exists():
            raise FileNotFoundError(f"Feature list not found: {txt}")
        return [
            ln.strip()
            for ln in txt.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    default = ROOT / "results" / "ml" / dataset / "02_lasso_selected_features.txt"
    if default.exists():
        return [
            ln.strip()
            for ln in default.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    cfg_path = ROOT / "results" / "ml" / dataset / "pipeline_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())["selected_features"]
    raise FileNotFoundError(
        f"No feature list for {dataset}. Pass --features-from or run ML pipeline first."
    )


def median_impute(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in features:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out[c].median()
        out[c] = out[c].fillna(med)
    return out


def fit_glm(
    df: pd.DataFrame,
    features: list[str],
    *,
    weighted: bool,
) -> tuple[object, pd.DataFrame]:
    data = median_impute(df, features)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce").astype(int)
    formula = f"{TARGET} ~ " + " + ".join(f"Q('{f}')" for f in features)

    if weighted:
        w = pd.to_numeric(data[WEIGHT_COL], errors="coerce")
        w = w.fillna(w.median())
        w = w / w.mean()
        model = smf.glm(formula, data=data, family=sm.families.Binomial(), freq_weights=w).fit()
        weight_note = "frequency_weights_survey_weight"
    else:
        model = smf.glm(formula, data=data, family=sm.families.Binomial()).fit()
        weight_note = "unweighted"

    params = model.params
    conf = model.conf_int()
    or_table = pd.DataFrame({
        "term": params.index,
        "log_odds": params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values,
        "or": np.exp(params.values),
        "or_ci_low": np.exp(conf[0].values),
        "or_ci_high": np.exp(conf[1].values),
    })
    or_table["weighting"] = weight_note
    or_table = add_fdr_to_or_table(or_table)
    return model, or_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Interpretable GLM on LASSO features")
    parser.add_argument("--dataset", choices=["ny", "all_states"], default="ny")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train",
        help="Data split to fit GLM (default train for inference)",
    )
    parser.add_argument(
        "--features-from",
        type=Path,
        default=None,
        help="Path to 02_lasso_selected_features.txt or results/<pipeline>/<dataset>/ dir",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use survey_weight as frequency weights (approximate; use R svyglm for design-based SEs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default results/posthoc/glm_<dataset>)",
    )
    args = parser.parse_args()

    prefix = _dataset_prefix(args.dataset)
    split_path = PROCESSED / f"{prefix}_{args.split}.csv"
    if not split_path.exists():
        print(f"Missing {split_path}", file=sys.stderr)
        return 1

    features = load_features(args.features_from, args.dataset)
    df = pd.read_csv(split_path, low_memory=False)
    print(f"Fitting GLM on {args.split} (n={len(df)}), {len(features)} features, weighted={args.weighted}")

    model, or_table = fit_glm(df, features, weighted=args.weighted)

    out_dir = args.out_dir or (ROOT / "results" / "posthoc" / f"glm_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "weighted" if args.weighted else "unweighted"
    or_table.to_csv(out_dir / f"glm_or_table_{args.split}_{suffix}.csv", index=False)
    (out_dir / f"glm_summary_{args.split}_{suffix}.txt").write_text(model.summary().as_text())

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "weighted": args.weighted,
        "n_obs": int(len(df)),
        "n_features": len(features),
        "features": features,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "pseudo_r2": float(getattr(model, "prsquared", np.nan)),
        "fdr_correction": "Benjamini-Hochberg across LASSO-selected predictors (intercept excluded)",
        "fdr_alpha": 0.05,
        "n_significant_fdr_05": int(or_table["significant_fdr_05"].sum()),
        "note": "Median imputation only; no batch correction. Report q_value_fdr_bh for inference, not raw p_value alone. For survey-design SEs use scripts/fit_interpretable_glm.R",
    }
    (out_dir / f"glm_meta_{args.split}_{suffix}.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote OR table and summary to {out_dir}")
    print(or_table.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
