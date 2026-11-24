#!/usr/bin/env bash
# Standard + voting_soft (Python) + tidymodels (R) for ny and all_states.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$HOME/.venvs/15_7_stroke}"
export PROJECT_ROOT="$ROOT"
LOG="${ROOT}/results/run_all_pipelines_full.log"
QUICK="${1:-}"

source "${VENV}/bin/activate"
python scripts/test_classifier_count.py

ARGS=()
RARGS=(--dataset both)
if [[ "$QUICK" == "--quick" ]]; then
  ARGS+=(--quick)
  RARGS+=(--quick)
fi

mkdir -p results/ml results/ml_voting_soft results/tidymodels

{
  echo "========== $(date -Iseconds) START =========="
  echo "[1/3] Python standard → results/ml/"
  python scripts/run_ml_pipeline.py --dataset both --n-jobs -1 "${ARGS[@]}"
  echo "[2/3] Python voting_soft → results/ml_voting_soft/"
  python voting_soft_pipeline/run_ml_pipeline.py --dataset both --n-jobs -1 "${ARGS[@]}"
  echo "[3/3] R tidymodels → results/tidymodels/"
  Rscript tidymodels_pipeline/scripts/run_ml_pipeline.R "${RARGS[@]}"
  python scripts/validate_pipeline.py ny all_states || true
  echo "========== $(date -Iseconds) DONE =========="
} 2>&1 | tee -a "$LOG"
