#!/usr/bin/env bash
# Build the runnable `project/` layout expected by the pipelines from this package.
#
#   project/
#   ├── scripts/                    <- 04_pipelines/shared_scripts + 06_posthoc
#   ├── voting_soft_pipeline/       <- 04_pipelines/02_python_voting_soft
#   ├── tidymodels_pipeline/        <- 04_pipelines/03_r_tidymodels_voting_soft
#   ├── tidymodels_single_pipeline/ <- 04_pipelines/04_r_tidymodels_single
#   ├── knowledge_graph_pipeline/   <- 04_pipelines/05_knowledge_graph
#   ├── knowledge_graph_data_only/  <- 04_pipelines/06_knowledge_graph_data_only
#   ├── deep_learning_pipeline/     <- 04_pipelines/07_python_deep_learning
#   ├── datasets/processed/         <- 02_datasets/processed (CSV files)
#   └── results/                    <- pipeline outputs
#
# Files are copied (works on exFAT/USB drives where symlinks fail). Re-run after editing code.
set -euo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$PKG/project}"

copy_dir() {  # src dst
  mkdir -p "$2"
  cp -R "$1"/. "$2"/
  find "$2" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find "$2" -name '._*' -delete 2>/dev/null || true
}

echo "Package: $PKG"
echo "Project: $OUT"
mkdir -p "$OUT/results" "$OUT/datasets/processed"

copy_dir "$PKG/04_pipelines/shared_scripts"              "$OUT/scripts"
copy_dir "$PKG/06_posthoc"                               "$OUT/scripts"
copy_dir "$PKG/04_pipelines/02_python_voting_soft"       "$OUT/voting_soft_pipeline"
copy_dir "$PKG/04_pipelines/03_r_tidymodels_voting_soft" "$OUT/tidymodels_pipeline"
copy_dir "$PKG/04_pipelines/04_r_tidymodels_single"      "$OUT/tidymodels_single_pipeline"
copy_dir "$PKG/04_pipelines/05_knowledge_graph"          "$OUT/knowledge_graph_pipeline"
copy_dir "$PKG/04_pipelines/06_knowledge_graph_data_only" "$OUT/knowledge_graph_data_only"
copy_dir "$PKG/04_pipelines/07_python_deep_learning"     "$OUT/deep_learning_pipeline"
# Pipeline 1 is the canonical copy of scripts/run_ml_pipeline.py
cp "$PKG/04_pipelines/01_python_standard/run_ml_pipeline.py" "$OUT/scripts/run_ml_pipeline.py"

if compgen -G "$PKG/02_datasets/processed/*.csv" > /dev/null; then
  cp "$PKG"/02_datasets/processed/*.csv "$OUT/datasets/processed/"
  echo "Copied processed CSVs."
else
  echo "[warn] No CSVs in 02_datasets/processed/ — add data before running (see 02_datasets/README.md)."
fi

echo ""
echo "Done. Next:"
echo "  cd \"$OUT\" && export PROJECT_ROOT=\"\$(pwd)\""
echo "  python scripts/run_ml_pipeline.py --dataset ny --quick"
