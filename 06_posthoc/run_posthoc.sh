#!/usr/bin/env bash
# Regenerate all results/posthoc/ analyses after ML pipelines complete.
#
# Usage (from project root):
#   bash scripts/run_posthoc.sh
#   bash scripts/run_posthoc.sh compare scoring   # subset of steps
#
# Requires: Python venv (statsmodels), R (survey, broom via tidymodels_pipeline/setup.R)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"

VENV="${VENV:-$HOME/.venvs/15_7_stroke}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
fi

STEPS=("$@")
if [[ ${#STEPS[@]} -eq 0 ]]; then
  STEPS=(compare glm_py glm_r importance scoring)
fi

run_step() {
  local name="$1"
  shift
  echo ""
  echo "========== posthoc: $name =========="
  "$@"
}

should_run() {
  local want="$1"
  for s in "${STEPS[@]}"; do
    [[ "$s" == "$want" ]] && return 0
  done
  return 1
}

if should_run compare; then
  run_step "compare_pipelines" python scripts/compare_pipelines.py --dataset both
fi

if should_run glm_py; then
  run_step "glm_ny_unweighted" python scripts/fit_interpretable_glm.py --dataset ny
  run_step "glm_ny_weighted" python scripts/fit_interpretable_glm.py --dataset ny --weighted
  run_step "glm_all_states_unweighted" \
    python scripts/fit_interpretable_glm.py --dataset all_states --features-from results/ml/all_states
  run_step "glm_all_states_weighted" \
    python scripts/fit_interpretable_glm.py --dataset all_states --weighted \
      --features-from results/ml/all_states
fi

if should_run glm_r; then
  run_step "glm_ny_svyglm" Rscript scripts/fit_interpretable_glm.R --dataset ny
  if Rscript scripts/fit_interpretable_glm.R --dataset all_states \
      --features-from results/ml/all_states 2>/dev/null; then
    :
  else
    echo "  [warn] national svyglm failed (lonely PSU strata); writing R unweighted GLM instead"
    Rscript scripts/fit_interpretable_glm.R --dataset all_states \
      --features-from results/ml/all_states --unweighted
  fi
fi

if should_run importance; then
  run_step "export_model_importance" \
    python scripts/export_model_importance.py --dataset both --pipeline both --no-plot
fi

if should_run scoring; then
  for ds in ny all_states; do
    for pipe in ml voting_soft; do
      for split in test val; do
        run_step "score_${ds}_${split}_${pipe}" \
          python scripts/score_cohort.py --dataset "$ds" --split "$split" --pipeline "$pipe"
      done
    done
  done
fi

echo ""
echo "Post-hoc complete. See results/posthoc/ and IMPORTANCE_README.md"
