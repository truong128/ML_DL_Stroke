#!/usr/bin/env Rscript
# tidymodels single pipeline with configurable AdaBoost (default: light mfinal=40).
#
# Usage (from project root):
#   export PROJECT_ROOT="$(pwd)"
#
#   # Light AdaBoost (production-ish speed)
#   Rscript tidymodels_single_pipeline/scripts/run_ml_pipeline_light_adaboost.R \
#     --dataset ny --n-jobs 22 --cv-jobs 4
#   # -> results/tidymodels_single/ny/light_adaboost/
#
#   # End-to-end pipeline smoke test (AdaBoost mfinal=1; minutes not hours)
#   Rscript tidymodels_single_pipeline/scripts/run_ml_pipeline_light_adaboost.R \
#     --test --dataset ny --n-jobs 22 --cv-jobs 4
#   # -> results/tidymodels_single/ny/test_adaboost/
#
#   # Custom AdaBoost size
#   Rscript tidymodels_single_pipeline/scripts/run_ml_pipeline_light_adaboost.R \
#     --adaboost-mfinal 5 --dataset ny --n-jobs 4
#
# Done when: .../<output_subdir>/pipeline_config.json exists

args <- commandArgs(trailingOnly = TRUE)
dataset <- "ny"
quick <- FALSE
use_gpu <- FALSE
test_mode <- FALSE
n_jobs <- -1L
cv_jobs <- NULL
adaboost_mfinal <- NULL
adaboost_maxdepth <- NULL
if (length(args) > 0) {
  for (i in seq_along(args)) {
    if (args[i] == "--dataset" && i < length(args)) dataset <- args[i + 1]
    if (args[i] == "--quick") quick <- TRUE
    if (args[i] == "--test") test_mode <- TRUE
    if (args[i] == "--use-gpu") use_gpu <- TRUE
    if (args[i] %in% c("--n-jobs", "--n_jobs") && i < length(args)) {
      n_jobs <- as.integer(args[i + 1])
    }
    if (args[i] %in% c("--cv-jobs", "--cv_jobs") && i < length(args)) {
      cv_jobs <- as.integer(args[i + 1])
    }
    if (args[i] %in% c("--adaboost-mfinal", "--adaboost_mfinal") && i < length(args)) {
      adaboost_mfinal <- as.integer(args[i + 1])
    }
    if (args[i] %in% c("--adaboost-maxdepth", "--adaboost_maxdepth") && i < length(args)) {
      adaboost_maxdepth <- as.integer(args[i + 1])
    }
  }
}

if (test_mode) {
  adaboost_mfinal <- 1L
  adaboost_maxdepth <- 4L
}

script_dir <- dirname(sub(
  "--file=", "",
  commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]
))
if (length(script_dir) == 0 || script_dir == "") {
  script_dir <- file.path(getwd(), "tidymodels_single_pipeline", "scripts")
}
project_root <- normalizePath(file.path(script_dir, "..", ".."), mustWork = TRUE)
pipeline_dir <- file.path(project_root, "tidymodels_pipeline")
Sys.setenv(PROJECT_ROOT = project_root)

source_tree <- function(dir) {
  r_dir <- file.path(dir, "R")
  preload <- file.path(
    r_dir,
    c("utils_metrics.R", "utils_parallel.R", "utils_gpu.R", "xgboost_gpu_py.R")
  )
  preload <- preload[file.exists(preload)]
  invisible(lapply(preload, source, local = FALSE))
  files <- sort(list.files(r_dir, pattern = "\\.R$", full.names = TRUE))
  files <- setdiff(files, preload)
  invisible(lapply(files, source, local = FALSE))
}
source_tree(pipeline_dir)

OUTPUT_SUBDIR <- if (test_mode) {
  "test_adaboost"
} else {
  "light_adaboost"
}

if (test_mode) {
  message(
    "=== Pipeline TEST mode: AdaBoost mfinal=1, maxdepth=4 (full 16-model CV otherwise) ===\n",
    "Outputs: results/tidymodels_single/<dataset>/", OUTPUT_SUBDIR, "/\n",
    "Not for publication — use default light (mfinal=40) or run_ml_pipeline.R for real runs."
  )
} else {
  light_mfinal <- if (is.null(adaboost_mfinal)) 40L else adaboost_mfinal
  light_maxdepth <- if (is.null(adaboost_maxdepth)) 8L else adaboost_maxdepth
  message(
    "=== Full pipeline (single) with light AdaBoost: mfinal=",
    light_mfinal, ", maxdepth=", light_maxdepth, " ===\n",
    "Outputs: results/tidymodels_single/<dataset>/", OUTPUT_SUBDIR, "/\n",
    "Heavy AdaBoost (mfinal=100): results/tidymodels_single/<dataset>/\n",
    "Recommended: --n-jobs 22 --cv-jobs 4 (no --use-gpu for fairer gradient_boosting vs Python)."
  )
}

datasets <- if (dataset == "both") c("ny", "all_states") else dataset
for (ds in datasets) {
  run_pipeline(
    ds,
    quick = quick,
    n_jobs = n_jobs,
    cv_jobs = cv_jobs,
    use_gpu = use_gpu,
    light_adaboost = !test_mode && is.null(adaboost_mfinal),
    adaboost_mfinal = adaboost_mfinal,
    adaboost_maxdepth = adaboost_maxdepth,
    output_subdir = OUTPUT_SUBDIR,
    variant = "single"
  )
}
