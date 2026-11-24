#!/usr/bin/env bash
# Run R tidymodels pipelines ONE AT A TIME (shared fixes in tidymodels_pipeline/R/).
#
# Usage (from project root):
#   bash scripts/run_tidymodels_pipelines.sh list
#   bash scripts/run_tidymodels_pipelines.sh tidymodels_single_ny
#   bash scripts/run_tidymodels_pipelines.sh all_production_ny
#
# Logs: logs/tidymodels_runs/<job>_<timestamp>.log
# Done: results/<tree>/<dataset>[/subdir]/pipeline_config.json

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"

N_JOBS="${N_JOBS:-4}"
CV_JOBS="${CV_JOBS:-1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/tidymodels_runs}"
mkdir -p "$LOG_DIR"

timestamp() { date +"%Y%m%d_%H%M%S"; }

is_done() {
  local subdir="$1" dataset="$2" subpath="${3:-}"
  local d="$ROOT/results/$subdir/$dataset"
  [[ -n "$subpath" ]] && d="$d/$subpath"
  [[ -f "$d/pipeline_config.json" ]]
}

run_r() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}_$(timestamp).log"
  echo "=== $name ==="
  echo "  log: $log"
  echo "  cmd: Rscript $*"
  Rscript "$@" 2>&1 | tee "$log"
  echo "  finished: $name"
}

# --- results/tidymodels/ (voting_soft final model) ---

tidymodels_ny() {
  run_r tidymodels_ny \
    tidymodels_pipeline/scripts/run_ml_pipeline.R \
    --dataset ny --n-jobs "$N_JOBS" --cv-jobs "$CV_JOBS"
}

tidymodels_all_states() {
  run_r tidymodels_all_states \
    tidymodels_pipeline/scripts/run_ml_pipeline.R \
    --dataset all_states --n-jobs "$N_JOBS" --cv-jobs "$CV_JOBS"
}

tidymodels_ny_light_adaboost() {
  run_r tidymodels_ny_light \
    tidymodels_pipeline/scripts/run_ml_pipeline_light_adaboost.R \
    --dataset ny --n-jobs "$N_JOBS" --cv-jobs "$CV_JOBS"
}

# --- results/tidymodels_single/ (best single final model) ---

tidymodels_single_ny() {
  run_r tidymodels_single_ny \
    tidymodels_single_pipeline/scripts/run_ml_pipeline.R \
    --dataset ny --n-jobs "$N_JOBS" --cv-jobs "$CV_JOBS"
}

tidymodels_single_all_states() {
  run_r tidymodels_single_all_states \
    tidymodels_single_pipeline/scripts/run_ml_pipeline.R \
    --dataset all_states --n-jobs "$N_JOBS" --cv-jobs "$CV_JOBS"
}

tidymodels_single_ny_light_adaboost() {
  run_r tidymodels_single_ny_light \
    tidymodels_single_pipeline/scripts/run_ml_pipeline_light_adaboost.R \
    --dataset ny --n-jobs "$N_JOBS" --cv-jobs "$CV_JOBS"
}

# Smoke test (AdaBoost mfinal=1) — validated end-to-end
tidymodels_single_ny_test() {
  run_r tidymodels_single_ny_test \
    tidymodels_single_pipeline/scripts/run_ml_pipeline_light_adaboost.R \
    --test --dataset ny --n-jobs "$N_JOBS" --cv-jobs "$CV_JOBS"
}

all_production_ny() {
  echo "NY production runs (sequential). AdaBoost full mfinal=100 is slow on each."
  is_done tidymodels_single ny || tidymodels_single_ny
  is_done tidymodels ny || tidymodels_ny
}

all_production() {
  echo "All production R runs (sequential; can take days)."
  is_done tidymodels_single ny || tidymodels_single_ny
  is_done tidymodels ny || tidymodels_ny
  is_done tidymodels_single all_states || tidymodels_single_all_states
  is_done tidymodels all_states || tidymodels_all_states
}

all_incomplete() {
  echo "Skip jobs that already have pipeline_config.json"
  is_done tidymodels ny || tidymodels_ny
  is_done tidymodels all_states || tidymodels_all_states
  is_done tidymodels_single ny || tidymodels_single_ny
  is_done tidymodels_single all_states || tidymodels_single_all_states
}

list_jobs() {
  cat <<'EOF'
Shared R core: tidymodels_pipeline/R/ (both entry scripts source this tree)

results/tidymodels/  — voting_soft final model
  tidymodels_ny                     -> results/tidymodels/ny/
  tidymodels_all_states             -> results/tidymodels/all_states/
  tidymodels_ny_light_adaboost      -> results/tidymodels/ny/light_adaboost/

results/tidymodels_single/  — best single (mirrors Python results/ml/)
  tidymodels_single_ny              -> results/tidymodels_single/ny/
  tidymodels_single_all_states      -> results/tidymodels_single/all_states/
  tidymodels_single_ny_light_adaboost -> results/tidymodels_single/ny/light_adaboost/
  tidymodels_single_ny_test         -> results/tidymodels_single/ny/test_adaboost/ (smoke)

  all_production_ny    -> single NY, then voting_soft NY
  all_production       -> all four production jobs above
  all_incomplete       -> any missing pipeline_config.json

Recommended settings:
  export PROJECT_ROOT="$(pwd)"
  N_JOBS=4 CV_JOBS=1 bash scripts/run_tidymodels_pipelines.sh tidymodels_single_ny

Monitor:
  tail -f results/tidymodels_single/ny/03_cv_model_comparison_partial.csv
EOF
}

usage() {
  echo "Usage: bash scripts/run_tidymodels_pipelines.sh <job>"
  echo ""
  list_jobs
}

cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" == "help" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript not found. Run: Rscript tidymodels_pipeline/setup.R" >&2
  exit 1
fi

if declare -f "$cmd" >/dev/null 2>&1; then
  "$cmd"
else
  echo "Unknown job: $cmd" >&2
  usage
  exit 1
fi
