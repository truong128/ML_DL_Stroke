#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PROJECT_ROOT="$ROOT"
cd "$ROOT"
ARGS=(--dataset both)
if [[ "${1:-}" == "--quick" ]]; then ARGS+=(--quick); fi
Rscript tidymodels_pipeline/scripts/run_ml_pipeline.R "${ARGS[@]}"
