#!/usr/bin/env Rscript
# Export mean |SHAP| on validation for locked R tidymodels models (11_best_model.rds).
# Does not re-train — loads fitted workflow and uses XGBoost predcontrib (TreeSHAP).
#
# Usage (from Collaboration Package root):
#   .conda_r_score/bin/Rscript 06_posthoc/export_shap_r.R --dataset all_states --pipeline tidymodels_single
#   .conda_r_score/bin/Rscript 06_posthoc/export_shap_r.R --dataset ny --pipeline tidymodels_single

args <- commandArgs(trailingOnly = TRUE)
dataset <- "all_states"
pipeline <- "tidymodels_single"
max_rows <- 1500L
seed <- 42L
for (i in seq_along(args)) {
  if (args[i] == "--dataset" && i < length(args)) dataset <- args[i + 1]
  if (args[i] == "--pipeline" && i < length(args)) pipeline <- args[i + 1]
  if (args[i] == "--max-rows" && i < length(args)) max_rows <- as.integer(args[i + 1])
  if (args[i] == "--seed" && i < length(args)) seed <- as.integer(args[i + 1])
}

pkg_root <- normalizePath(file.path(
  dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])),
  ".."
), mustWork = TRUE)

`%||%` <- function(x, y) if (is.null(x)) y else x

suppressPackageStartupMessages({
  library(readr)
  library(jsonlite)
  library(recipes)
  library(workflows)
  library(parsnip)
  library(hardhat)
  library(xgboost)
  library(dplyr)
})

pipeline_dirs <- list(
  tidymodels_single = file.path(pkg_root, "05_results", "04_r_tidymodels_single"),
  tidymodels = file.path(pkg_root, "05_results", "03_r_tidymodels_voting_soft")
)
if (!pipeline %in% names(pipeline_dirs)) {
  stop("Unknown --pipeline. Choose: ", paste(names(pipeline_dirs), collapse = ", "))
}

r_dir <- file.path(pkg_root, "04_pipelines", "03_r_tidymodels_voting_soft", "R")
preload <- file.path(r_dir, c("utils_metrics.R", "utils_parallel.R", "utils_gpu.R", "xgboost_gpu_py.R"))
preload <- preload[file.exists(preload)]
invisible(lapply(preload, source, local = FALSE))
files <- sort(list.files(r_dir, pattern = "\\.R$", full.names = TRUE))
files <- setdiff(files, preload)
invisible(lapply(files, source, local = FALSE))

results_dir <- file.path(pipeline_dirs[[pipeline]], dataset)
model_path <- file.path(results_dir, "11_best_model.rds")
cfg_path <- file.path(results_dir, "pipeline_config.json")
if (!file.exists(model_path)) stop("Missing model: ", model_path)
cfg <- jsonlite::fromJSON(cfg_path)
selected <- cfg$selected_features
best_model <- cfg$best_model %||% cfg$final_model
obj <- readRDS(model_path)

if (!identical(obj$variant, "single")) {
  stop("SHAP export supports variant=single only (use Python export_model_importance for voting_soft).")
}

prefix <- if (dataset == "ny") "brfss_stroke_new_york" else "brfss_stroke_all_states"
proc <- file.path(pkg_root, "02_datasets", "processed")
train <- readr::read_csv(file.path(proc, paste0(prefix, "_train.csv")), show_col_types = FALSE)
val <- readr::read_csv(file.path(proc, paste0(prefix, "_val.csv")), show_col_types = FALSE)

feature_cols <- get_feature_columns(train)
xy_fit <- prepare_xy(train, feature_cols)
xy_val <- prepare_xy(val, feature_cols)
bc <- fit_batch_correction(xy_fit$x, train[[BATCH_COL]])
x_val_bc <- apply_batch_correction(xy_val$x, val[[BATCH_COL]], bc)
val_df <- matrix_to_df(
  x_val_bc[, match(selected, feature_cols), drop = FALSE],
  selected, xy_val$y, val[[BATCH_COL]]
)
val_df$stroke <- factor(val_df$stroke, levels = c(0, 1))

wf <- obj$fit
prep <- hardhat::extract_recipe(wf)
baked <- recipes::bake(prep, new_data = val_df)
pred_cols <- setdiff(names(baked), c(TARGET, BATCH_COL))
X <- as.matrix(baked[, pred_cols, drop = FALSE])

set.seed(seed)
n_sample <- min(max_rows, nrow(X))
idx <- sample.int(nrow(X), size = n_sample, replace = FALSE)
Xs <- X[idx, , drop = FALSE]

booster <- parsnip::extract_fit_engine(wf)
if (!inherits(booster, "xgb.Booster")) {
  stop("Expected xgb.Booster engine; got ", paste(class(booster), collapse = ", "))
}

contrib <- predict(booster, Xs, predcontrib = TRUE)
if (ncol(contrib) != length(pred_cols) + 1L) {
  stop("predcontrib columns mismatch: got ", ncol(contrib), ", expected ", length(pred_cols) + 1L)
}
shaps <- contrib[, seq_len(ncol(contrib) - 1L), drop = FALSE]
colnames(shaps) <- pred_cols
mean_abs <- colMeans(abs(shaps))
total <- sum(mean_abs)

out_df <- data.frame(
  feature = pred_cols,
  shap_mean_abs = as.numeric(mean_abs),
  shap_mean_abs_normalized = if (total > 0) as.numeric(mean_abs / total) else NA_real_
) |>
  dplyr::arrange(dplyr::desc(shap_mean_abs))

out_csv <- file.path(results_dir, "12_shap_mean_abs.csv")
readr::write_csv(out_df, out_csv)

meta <- list(
  model = best_model,
  dataset = dataset,
  pipeline = pipeline,
  note = sprintf(
    "SHAP on validation (n=%d; batch-corrected features; recipe bake + locked XGBoost TreeSHAP/predcontrib; uncalibrated)",
    n_sample
  ),
  n_features = length(pred_cols),
  seed = seed,
  max_rows = max_rows
)
writeLines(
  jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
  file.path(results_dir, "12_shap_meta.json")
)
writeLines(
  c(
    sprintf("SHAP exported for %s (%s) on validation, n=%d.", best_model, dataset, n_sample),
    sprintf("Top feature: %s (mean |SHAP| = %.4f)", out_df$feature[1], out_df$shap_mean_abs[1]),
    sprintf("Written: %s", out_csv)
  ),
  file.path(results_dir, "10_shap_summary.txt")
)

message("Wrote ", nrow(out_df), " features to ", out_csv)
message("  top: ", out_df$feature[1], " = ", round(out_df$shap_mean_abs[1], 4))
message("  note: ", meta$note)
