#!/usr/bin/env bash
# Quick smoke test (ny only) for Python standard, voting_soft, and tidymodels.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"
VENV="${VENV:-$HOME/.venvs/15_7_stroke}"
LOG="${ROOT}/results/quick_tests_all_pipelines.log"
source "${VENV}/bin/activate"

{
  echo "========== $(date -Iseconds) quick tests (ny) =========="
  python scripts/test_classifier_count.py
  Rscript tidymodels_pipeline/scripts/test_classifier_count.R
  echo "--- [1/3] standard ml ---"
  python scripts/run_ml_pipeline.py --dataset ny --quick --n-jobs -1
  echo "--- [2/3] voting_soft ---"
  python voting_soft_pipeline/run_ml_pipeline.py --dataset ny --quick --n-jobs -1
  echo "--- [3/3] tidymodels ---"
  Rscript tidymodels_pipeline/scripts/run_ml_pipeline.R --dataset ny --quick
  python scripts/validate_pipeline.py ny || true
  echo "========== $(date -Iseconds) quick tests DONE =========="
} 2>&1 | tee "$LOG"
