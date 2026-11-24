#!/usr/bin/env Rscript
# BRFSS stroke ML benchmark — tidymodels (https://www.tidymodels.org)
#
# Usage:
#   Rscript scripts/run_ml_pipeline.R --dataset ny
#   Rscript scripts/run_ml_pipeline.R --dataset both
#   Rscript scripts/run_ml_pipeline.R --dataset ny --quick
#   Rscript scripts/run_ml_pipeline.R --dataset ny --n-jobs -1   # all CPUs (default)
#   Rscript scripts/run_ml_pipeline.R --dataset ny --use-gpu
#   Rscript scripts/run_ml_pipeline.R --dataset ny --light-adaboost

args <- commandArgs(trailingOnly = TRUE)
dataset <- "ny"
quick <- FALSE
use_gpu <- FALSE
light_adaboost <- FALSE
n_jobs <- -1L
cv_jobs <- NULL
if (length(args) > 0) {
  for (i in seq_along(args)) {
    if (args[i] == "--dataset" && i < length(args)) dataset <- args[i + 1]
    if (args[i] == "--quick") quick <- TRUE
    if (args[i] == "--use-gpu") use_gpu <- TRUE
    if (args[i] == "--light-adaboost") light_adaboost <- TRUE
    if (args[i] %in% c("--n-jobs", "--n_jobs") && i < length(args)) {
      n_jobs <- as.integer(args[i + 1])
    }
    if (args[i] %in% c("--cv-jobs", "--cv_jobs") && i < length(args)) {
      cv_jobs <- as.integer(args[i + 1])
    }
  }
}

script_dir <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
if (length(script_dir) == 0 || script_dir == "") {
  script_dir <- file.path(getwd(), "scripts")
}
pipeline_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
Sys.setenv(PROJECT_ROOT = normalizePath(file.path(pipeline_dir, ".."), mustWork = TRUE))

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

datasets <- if (dataset == "both") c("ny", "all_states") else dataset
for (ds in datasets) {
  run_pipeline(
    ds, quick = quick, n_jobs = n_jobs, cv_jobs = cv_jobs,
    use_gpu = use_gpu, light_adaboost = light_adaboost,
    output_subdir = if (light_adaboost) "light_adaboost" else NULL,
    variant = "voting_soft"
  )
}
