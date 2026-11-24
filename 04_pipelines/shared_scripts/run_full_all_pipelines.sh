#!/usr/bin/env bash
# Full runs (ny + all_states) for all three pipelines.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"
VENV="${VENV:-$HOME/.venvs/15_7_stroke}"
LOG="${ROOT}/results/full_all_pipelines.log"
source "${VENV}/bin/activate"

{
  echo "========== $(date -Iseconds) FULL runs (both datasets) =========="
  python scripts/test_classifier_count.py
  echo "--- [1/3] standard ml ---"
  python scripts/run_ml_pipeline.py --dataset both --n-jobs -1
  echo "--- [2/3] voting_soft ---"
  python voting_soft_pipeline/run_ml_pipeline.py --dataset both --n-jobs -1
  echo "--- [3/3] tidymodels ---"
  Rscript tidymodels_pipeline/scripts/run_ml_pipeline.R --dataset both
  python scripts/validate_pipeline.py ny all_states || true
  echo "========== $(date -Iseconds) FULL runs DONE =========="
} 2>&1 | tee "$LOG"
