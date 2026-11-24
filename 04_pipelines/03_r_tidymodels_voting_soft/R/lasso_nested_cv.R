# Nested CV LASSO (glmnet) with survey_year groups — mirrors lasso_nested_cv()
#
# glmnet lambda = 1/C (sklearn-style C). The sklearn grid 0.005–1.0 maps to lambda 200–1,
# which is too strong for glmnet (all coefficients zero). We extend C with weaker penalties
# (e.g. C = 10, 100 => lambda = 0.1, 0.01) and add path-based lambda candidates.

default_lasso_C_grid <- function(quick = FALSE) {
  if (quick) {
    c(0.1, 0.5, 1, 5, 10, 50, 100)
  } else {
    c(0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100)
  }
}

lasso_prep_matrix <- function(x) {
  x_imp <- x
  for (j in seq_len(ncol(x_imp))) {
    med <- stats::median(x_imp[, j], na.rm = TRUE)
    if (is.na(med)) med <- 0
    x_imp[is.na(x_imp[, j]), j] <- med
  }
  x_sc <- scale(x_imp)
  x_sc[is.na(x_sc)] <- 0
  x_sc
}

lasso_case_weights <- function(y) {
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  ifelse(y == 1, n0 / max(n1, 1), 1)
}

lasso_lambda_grid_for_data <- function(x, y, C_grid) {
  x_sc <- lasso_prep_matrix(x)
  w <- lasso_case_weights(y)
  base_lambda <- sort(unique(1 / C_grid), decreasing = TRUE)
  path <- glmnet::glmnet(
    x_sc, y,
    family = "binomial",
    alpha = 1,
    standardize = FALSE,
    weights = w
  )
  coef_mat <- as.matrix(stats::coef(path))[-1, , drop = FALSE]
  n_nonzero <- colSums(abs(coef_mat) > 1e-6)
  viable <- path$lambda[n_nonzero >= 2L]
  if (length(viable) < 2L) {
    viable <- path$lambda[n_nonzero >= 1L]
  }
  path_lambda <- if (length(viable) >= 2L) {
    stats::quantile(viable, probs = seq(0.1, 0.9, length.out = 6), na.rm = TRUE)
  } else {
    stats::quantile(path$lambda, probs = seq(0.1, 0.9, length.out = 6), na.rm = TRUE)
  }
  sort(unique(c(base_lambda, as.numeric(path_lambda))), decreasing = TRUE)
}

lasso_select <- function(x, y, feature_names, lambda, max_features = 30L) {
  x_sc <- lasso_prep_matrix(x)
  w <- lasso_case_weights(y)
  fit <- glmnet::glmnet(
    x_sc, y,
    family = "binomial",
    alpha = 1,
    lambda = lambda,
    standardize = FALSE,
    weights = w
  )
  coefs <- as.matrix(stats::coef(fit))[-1, 1]
  names <- feature_names[abs(coefs) > 1e-6]
  if (length(names) > max_features) {
    ord <- order(abs(coefs), decreasing = TRUE)[seq_len(max_features)]
    names <- feature_names[ord][abs(coefs[ord]) > 1e-6]
  }
  names
}

lasso_select_weakest <- function(x, y, feature_names, lambda_grid, max_features = 30L,
                                 min_features = 5L) {
  for (lam in sort(lambda_grid, decreasing = FALSE)) {
    sel <- lasso_select(x, y, feature_names, lam, max_features)
    if (length(sel) >= min_features) {
      return(list(features = sel, lambda = lam, C = 1 / lam))
    }
  }
  sel <- lasso_select(x, y, feature_names, min(lambda_grid), max_features)
  list(features = sel, lambda = min(lambda_grid), C = 1 / min(lambda_grid))
}

pick_best_lambda_row <- function(grid) {
  valid <- grid[!is.na(grid$roc_auc_mean), , drop = FALSE]
  if (nrow(valid) > 0) {
    valid <- valid[order(-valid$roc_auc_mean, -valid$n_features_mean), , drop = FALSE]
    return(valid[1, , drop = FALSE])
  }
  grid <- grid[order(-grid$n_features_mean, grid$lambda, decreasing = c(TRUE, FALSE)), , drop = FALSE]
  grid[1, , drop = FALSE]
}

inner_tune_lasso_lambda <- function(x, y, groups, lambda_grid, inner_splits) {
  df <- data.frame(
    stroke = factor(y, levels = c(0, 1)),
    survey_year = groups
  )
  df <- cbind(df, as.data.frame(x))
  folds <- rsample::group_vfold_cv(df, group = survey_year, v = inner_splits)
  rows <- lapply(lambda_grid, function(lam) {
    aucs <- integer(length(folds$splits))
    n_feats <- integer(length(folds$splits))
    for (i in seq_along(folds$splits)) {
      sp <- folds$splits[[i]]
      tr <- rsample::analysis(sp)
      te <- rsample::assessment(sp)
      feats <- setdiff(names(tr), c("stroke", "survey_year"))
      sel <- lasso_select(
        as.matrix(tr[, feats]), as.integer(as.character(tr$stroke)),
        feats, lam
      )
      n_feats[i] <- length(sel)
      if (length(sel) < 2) {
        aucs[i] <- NA_real_
        next
      }
      tr2 <- tr[, c("stroke", sel), drop = FALSE]
      te2 <- te[, c("stroke", sel), drop = FALSE]
      spec <- parsnip::logistic_reg() |>
        parsnip::set_engine("glm") |>
        parsnip::set_mode("classification")
      wflow <- workflows::workflow() |>
        workflows::add_recipe(make_recipe(sel, tr2, use_smote = TRUE)) |>
        workflows::add_model(spec)
      fit <- parsnip::fit(wflow, tr2)
      p <- predict(fit, te2, type = "prob")$.pred_1
      aucs[i] <- roc_auc_pos(as.integer(as.character(te2$stroke)), p)
    }
    data.frame(
      lambda = lam,
      C = 1 / lam,
      roc_auc_mean = mean(aucs, na.rm = TRUE),
      roc_auc_std = stats::sd(aucs, na.rm = TRUE),
      n_features_mean = mean(n_feats)
    )
  })
  grid <- dplyr::bind_rows(rows)
  best <- pick_best_lambda_row(grid)
  list(best_lambda = best$lambda[[1]], grid = grid)
}

lasso_nested_cv <- function(
    x, y, groups, feature_names, out_dir,
    C_grid = NULL,
    outer_splits = 5L,
    inner_splits = 3L,
    max_features = 30L,
    stability_threshold = 0.5,
    eval_classifier = "logistic_regression") {
  if (is.null(C_grid)) {
    C_grid <- default_lasso_C_grid(quick = outer_splits < 5L)
  }
  lambda_grid <- lasso_lambda_grid_for_data(x, y, C_grid)

  df <- data.frame(
    stroke = factor(y, levels = c(0, 1)),
    survey_year = groups
  )
  df <- cbind(df, as.data.frame(x))
  colnames(df)[-(1:2)] <- feature_names

  outer <- rsample::group_vfold_cv(df, group = survey_year, v = outer_splits)
  feature_counts <- stats::setNames(rep(0L, length(feature_names)), feature_names)
  outer_rows <- list()
  inner_all <- list()

  specs <- get_model_specs()
  eval_spec <- specs[[eval_classifier]]

  for (fold in seq_along(outer$splits)) {
    sp <- outer$splits[[fold]]
    tr <- rsample::analysis(sp)
    te <- rsample::assessment(sp)
    x_otr <- as.matrix(tr[, feature_names])
    y_otr <- as.integer(as.character(tr$stroke))
    g_otr <- tr$survey_year
    y_ote <- as.integer(as.character(te$stroke))

    fold_lambda_grid <- lasso_lambda_grid_for_data(x_otr, y_otr, C_grid)
    tuned <- inner_tune_lasso_lambda(x_otr, y_otr, g_otr, fold_lambda_grid, inner_splits)
    inner_df <- tuned$grid
    inner_df$outer_fold <- fold
    inner_all[[fold]] <- inner_df
    best_lam <- tuned$best_lambda

    selected <- lasso_select(x_otr, y_otr, feature_names, best_lam, max_features)
    for (f in selected) feature_counts[f] <- feature_counts[f] + 1L

    if (length(selected) < 2) {
      rescue <- lasso_select_weakest(x_otr, y_otr, feature_names, fold_lambda_grid, max_features)
      selected <- rescue$features
      best_lam <- rescue$lambda
      for (f in selected) feature_counts[f] <- feature_counts[f] + 1L
    }

    if (length(selected) < 2) {
      outer_rows[[fold]] <- data.frame(
        outer_fold = fold, best_C = 1 / best_lam,
        n_features = length(selected), outer_roc_auc = NA_real_, note = "too_few_features"
      )
      next
    }

    tr2 <- tr[, c("stroke", selected), drop = FALSE]
    te2 <- te[, c("stroke", selected), drop = FALSE]
    fit <- fit_workflow(eval_spec, tr2, selected, use_smote = TRUE)
    p <- predict_prob_pos(fit, te2)
    outer_auc <- if (length(unique(y_ote)) > 1) {
      roc_auc_pos(y_ote, p)
    } else {
      NA_real_
    }
    outer_rows[[fold]] <- data.frame(
      outer_fold = fold, best_C = 1 / best_lam,
      n_features = length(selected), outer_roc_auc = outer_auc, note = ""
    )
  }

  outer_df <- dplyr::bind_rows(outer_rows)
  inner_all_df <- dplyr::bind_rows(inner_all)
  stability_df <- data.frame(
    feature = feature_names,
    times_selected = as.integer(feature_counts[feature_names]),
    stability = as.integer(feature_counts[feature_names]) / outer_splits
  ) |>
    dplyr::arrange(dplyr::desc(stability))

  stable <- stability_df$feature[stability_df$stability >= stability_threshold]
  selection_method <- sprintf("stability>=%s", stability_threshold)
  if (length(stable) < 5) {
    rescue <- lasso_select_weakest(x, y, feature_names, lambda_grid, max_features)
    stable <- rescue$features
    selection_method <- sprintf(
      "fallback_weakest_penalty_C=%.4g (<%s stable at threshold)",
      rescue$C, stability_threshold
    )
  }

  summary <- list(
    outer_roc_auc_mean = mean(outer_df$outer_roc_auc, na.rm = TRUE),
    outer_roc_auc_std = stats::sd(outer_df$outer_roc_auc, na.rm = TRUE),
    median_best_C = stats::median(outer_df$best_C, na.rm = TRUE),
    n_stable_features = length(stable),
    stable_features = stable,
    selection_method = selection_method,
    lambda_grid_used = lambda_grid
  )

  readr::write_csv(outer_df, file.path(out_dir, "02a_lasso_nested_cv_outer_folds.csv"))
  readr::write_csv(inner_all_df, file.path(out_dir, "02b_lasso_nested_cv_inner_C_grid.csv"))
  readr::write_csv(stability_df, file.path(out_dir, "02c_lasso_feature_stability.csv"))
  readr::write_csv(
    data.frame(
      outer_roc_auc_mean = summary$outer_roc_auc_mean,
      outer_roc_auc_std = summary$outer_roc_auc_std,
      median_best_C = summary$median_best_C,
      n_stable_features = summary$n_stable_features,
      selection_method = summary$selection_method
    ),
    file.path(out_dir, "02d_lasso_nested_cv_summary.csv")
  )

  p <- ggplot2::ggplot(outer_df, ggplot2::aes(factor(outer_fold), outer_roc_auc)) +
    ggplot2::geom_col(fill = "steelblue") +
    ggplot2::geom_hline(
      yintercept = summary$outer_roc_auc_mean,
      linetype = "dashed", color = "red"
    ) +
    ggplot2::labs(
      title = "Nested CV — outer-test performance (tidymodels)",
      x = "Outer fold", y = "ROC-AUC"
    ) +
    ggplot2::theme_minimal()
  ggplot2::ggsave(file.path(out_dir, "02e_lasso_nested_cv_plots.png"), p, width = 7, height = 4, dpi = 150)

  summary
}
