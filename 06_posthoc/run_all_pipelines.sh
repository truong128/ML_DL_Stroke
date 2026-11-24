#!/usr/bin/env bash
# Run full ML pipeline for both cohorts (ny + all_states), standard and voting_soft.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$HOME/.venvs/15_7_stroke}"
LOG="${ROOT}/results/run_all_pipelines.log"

mkdir -p "${ROOT}/results/ml" "${ROOT}/results/ml_voting_soft"

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

echo "========== $(date -Iseconds) START run_all_pipelines ==========" | tee -a "$LOG"

echo "[1/2] Standard pipeline → results/ml/{ny,all_states}" | tee -a "$LOG"
python scripts/run_ml_pipeline.py --dataset both --n-jobs -1 2>&1 | tee -a "$LOG"

echo "[2/2] Voting-soft pipeline → results/ml_voting_soft/{ny,all_states}" | tee -a "$LOG"
python voting_soft_pipeline/run_ml_pipeline.py --dataset both --n-jobs -1 2>&1 | tee -a "$LOG"

echo "========== $(date -Iseconds) DONE run_all_pipelines ==========" | tee -a "$LOG"

if [[ -f scripts/validate_pipeline.py ]]; then
  python scripts/validate_pipeline.py ny all_states 2>&1 | tee -a "$LOG" || true
fi
