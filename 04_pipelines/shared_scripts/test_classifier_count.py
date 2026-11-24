#!/usr/bin/env python3
"""Assert all Python pipelines expose exactly 16 base classifiers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ml_pipeline import get_classifiers  # noqa: E402

EXPECTED = 16
REQUIRED = {
    "logistic_regression", "multinom_reg", "svm_rbf", "svm_linear", "decision_tree",
    "random_forest", "gradient_boosting", "extra_trees", "adaboost", "bag_tree",
    "bagging", "knn", "lda", "qda", "naive_bayes", "mlp",
}

clfs = get_classifiers()
names = set(clfs)
errors = []
if len(clfs) != EXPECTED:
    errors.append(f"Expected {EXPECTED} classifiers, got {len(clfs)}: {sorted(names)}")
missing = REQUIRED - names
extra = names - REQUIRED
if missing:
    errors.append(f"Missing: {sorted(missing)}")
if extra:
    errors.append(f"Unexpected: {sorted(extra)}")

if errors:
    print("FAILED:")
    for e in errors:
        print(" ", e)
    sys.exit(1)

print(f"OK: {EXPECTED} classifiers — {', '.join(sorted(names))}")
sys.exit(0)
