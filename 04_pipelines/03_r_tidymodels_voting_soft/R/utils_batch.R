# Batch correction (ComBat-lite) and data loading — mirrors scripts/run_ml_pipeline.py

META_COLS <- c(
  "survey_year", "state_fips", "split", "stroke",
  "survey_weight", "strata", "psu", "seqno"
)
BATCH_COL <- "survey_year"
TARGET <- "stroke"

resolve_project_root <- function() {
  # scripts/run_ml_pipeline.R sets PROJECT_ROOT; fallback to parent of tidymodels_pipeline
  env <- Sys.getenv("PROJECT_ROOT", unset = "")
  if (nzchar(env)) return(normalizePath(env))
  normalizePath(file.path(getwd(), ".."), mustWork = TRUE)
}

load_splits <- function(prefix, root = resolve_project_root()) {
  proc <- file.path(root, "datasets", "processed")
  list(
    train = readr::read_csv(file.path(proc, paste0(prefix, "_train.csv")), show_col_types = FALSE),
    val = readr::read_csv(file.path(proc, paste0(prefix, "_val.csv")), show_col_types = FALSE),
    test = readr::read_csv(file.path(proc, paste0(prefix, "_test.csv")), show_col_types = FALSE)
  )
}

get_feature_columns <- function(df) {
  setdiff(names(df), META_COLS)
}

prepare_xy <- function(df, feature_cols) {
  x <- df[, feature_cols, drop = FALSE]
  x[] <- lapply(x, function(col) suppressWarnings(as.numeric(col)))
  y <- as.integer(df[[TARGET]])
  list(x = as.matrix(x), y = y, features = feature_cols)
}

fit_batch_correction <- function(x, batches) {
  imp <- apply(x, 2, stats::median, na.rm = TRUE)
  x_imp <- x
  for (j in seq_len(ncol(x_imp))) {
    na_idx <- is.na(x_imp[, j])
    x_imp[na_idx, j] <- imp[j]
  }
  global_mean <- colMeans(x_imp)
  years <- unique(batches)
  year_means <- lapply(years, function(yr) {
    colMeans(x_imp[batches == yr, , drop = FALSE])
  })
  names(year_means) <- as.character(years)
  list(impute = imp, global_mean = global_mean, year_means = year_means)
}

apply_batch_correction <- function(x, batches, bc) {
  x_imp <- x
  for (j in seq_len(ncol(x_imp))) {
    na_idx <- is.na(x_imp[, j])
    x_imp[na_idx, j] <- bc$impute[j]
  }
  out <- x_imp
  for (i in seq_len(nrow(out))) {
    yr <- as.character(batches[i])
    if (yr %in% names(bc$year_means)) {
      out[i, ] <- out[i, ] - (bc$year_means[[yr]] - bc$global_mean)
    }
  }
  out
}

matrix_to_df <- function(x, features, y = NULL, batches = NULL) {
  df <- as.data.frame(x)
  names(df) <- features
  if (!is.null(y)) df[[TARGET]] <- y
  if (!is.null(batches)) df[[BATCH_COL]] <- batches
  df
}

plot_batch_pca <- function(x, batches, out_path, title) {
  imp <- apply(x, 2, stats::median, na.rm = TRUE)
  x_imp <- x
  for (j in seq_len(ncol(x_imp))) {
    x_imp[is.na(x_imp[, j]), j] <- imp[j]
  }
  x_sc <- scale(x_imp)
  pca <- stats::prcomp(x_sc, center = FALSE, scale. = FALSE)
  pct <- summary(pca)$importance[2, 1:2] * 100
  df <- data.frame(
    PC1 = pca$x[, 1],
    PC2 = pca$x[, 2],
    survey_year = factor(batches)
  )
  p <- ggplot2::ggplot(df, ggplot2::aes(PC1, PC2, color = survey_year)) +
    ggplot2::geom_point(alpha = 0.35, size = 1.2) +
    ggplot2::labs(
      title = title,
      x = sprintf("PC1 (%.1f%%)", pct[1]),
      y = sprintf("PC2 (%.1f%%)", pct[2])
    ) +
    ggplot2::theme_minimal(base_size = 11)
  ggplot2::ggsave(out_path, p, width = 7, height = 5, dpi = 150)
}
