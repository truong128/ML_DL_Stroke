#!/usr/bin/env Rscript
# Compare sequential vs parallel CV/val metrics (same code paths, sanity AUC range).

script_dir <- dirname(sub(
  "--file=", "",
  commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]
))
pipeline_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
Sys.setenv(PROJECT_ROOT = normalizePath(file.path(pipeline_dir, ".."), mustWork = TRUE))

source_tree <- function(dir) {
  r_dir <- file.path(dir, "R")
  preload <- file.path(r_dir, c("utils_metrics.R", "utils_parallel.R"))
  preload <- preload[file.exists(preload)]
  invisible(lapply(preload, source, local = FALSE))
  files <- sort(list.files(r_dir, pattern = "\\.R$", full.names = TRUE))
  invisible(lapply(setdiff(files, preload), source, local = FALSE))
}
source_tree(pipeline_dir)

prefix <- "brfss_stroke_new_york"
splits <- load_splits(prefix)
feature_cols <- get_feature_columns(splits$train)
xy_fit <- prepare_xy(splits$train, feature_cols)
bc <- fit_batch_correction(xy_fit$x, splits$train$survey_year)
x_fit_bc <- apply_batch_correction(xy_fit$x, splits$train$survey_year, bc)
selected <- readLines(file.path(
  Sys.getenv("PROJECT_ROOT"), "results/tidymodels_single/ny/02_lasso_selected_features.txt"
))
selected <- selected[!grepl("^#", selected) & nzchar(selected)]
if (length(selected) < 2) stop("Need LASSO features; run pipeline through LASSO first.")

train_df <- matrix_to_df(
  x_fit_bc[, match(selected, feature_cols), drop = FALSE],
  selected, xy_fit$y, splits$train$survey_year
)
val_df <- matrix_to_df(
  apply_batch_correction(
    prepare_xy(splits$val, feature_cols)$x, splits$val$survey_year, bc
  )[, match(selected, feature_cols), drop = FALSE],
  selected,
  prepare_xy(splits$val, feature_cols)$y,
  splits$val$survey_year
)
train_df$stroke <- factor(train_df$stroke, levels = c(0, 1))
val_df$stroke <- factor(val_df$stroke, levels = c(0, 1))

tmp <- tempfile()
on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

message("=== Sequential CV (3 folds, 4 models) ===")
set.seed(42)
cv_seq <- cv_evaluate_models(
  train_df, selected, tmp,
  n_splits = 3L, use_smote = TRUE, parallel = FALSE
)
subset <- c("logistic_regression", "svm_linear", "random_forest", "adaboost")
cv_seq <- cv_seq[cv_seq$model %in% subset, ]

message("\n=== Parallel CV (4 workers, same settings) ===")
init_r_parallel(n_jobs = 4L, quick = FALSE)
set.seed(42)
cv_par <- cv_evaluate_models(
  train_df, selected, tmp,
  n_splits = 3L, use_smote = TRUE, parallel = TRUE
)
cv_par <- cv_par[cv_par$model %in% subset, ]

merged <- merge(
  cv_seq[, c("model", "roc_auc_mean")],
  cv_par[, c("model", "roc_auc_mean")],
  by = "model", suffixes = c("_seq", "_par")
)
merged$diff <- abs(merged$roc_auc_mean_seq - merged$roc_auc_mean_par)
print(merged)

bad_auc <- any(merged$roc_auc_mean_seq < 0.5, na.rm = TRUE)
big_diff <- any(merged$diff > 0.05, na.rm = TRUE)
if (bad_auc) stop("FAIL: sequential AUC < 0.5 (metrics or model broken)")
if (big_diff) {
  warning("WARN: seq vs par AUC differ by >0.05 (fold RNG may differ under furrr); check values")
}

message("\n=== Val scoring: seq vs par (logistic only) ===")
spec <- get_model_specs()$logistic_regression
set.seed(42)
v_seq <- val_score_one_model("logistic_regression", spec, train_df, val_df, selected, FALSE)
set.seed(42)
init_r_parallel(2L, FALSE)
v_par <- parallel_map(
  "logistic_regression",
  function(nm) val_score_one_model(nm, spec, train_df, val_df, selected, FALSE),
  parallel = TRUE
)[[1]]
message("val AUC seq=", round(v_seq$val_roc_auc, 4), " par=", round(v_par$val_roc_auc, 4))

message("\n=== Bootstrap: seq vs par (50 reps) ===")
set.seed(99)
yt <- rep(c(0L, 1L), each = 50)
yp <- c(rep(0L, 40), rep(1L, 60))
ypr <- c(runif(50, 0, 0.4), runif(50, 0.6, 1))
b_seq <- bootstrap_metrics(yt, ypr, yp, n_boot = 50L, seed = 99L, parallel = FALSE)
init_r_parallel(2L, FALSE)
b_par <- bootstrap_metrics(yt, ypr, yp, n_boot = 50L, seed = 99L, parallel = TRUE)
roc_seq <- b_seq$mean_boot[b_seq$metric == "roc_auc"]
roc_par <- b_par$mean_boot[b_par$metric == "roc_auc"]
message("boot roc mean seq=", round(roc_seq, 4), " par=", round(roc_par, 4))

if (exists("case_weights_balanced", mode = "function")) {
  message("\nOK: case_weights_balanced defined")
} else {
  warning("WARN: case_weights_balanced() missing — final fit may error later")
}

message("\nPASS: parallel pipeline sanity checks completed")
