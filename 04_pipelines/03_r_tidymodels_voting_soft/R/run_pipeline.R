# Main end-to-end pipeline — variant: best single or voting_soft final model

run_pipeline <- function(
    dataset = c("ny", "all_states"),
    quick = FALSE,
    parallel = TRUE,
    n_jobs = -1L,
    cv_jobs = NULL,
    use_gpu = FALSE,
    light_adaboost = FALSE,
    adaboost_mfinal = NULL,
    adaboost_maxdepth = NULL,
    output_subdir = NULL,
    root = resolve_project_root(),
    variant = c("voting_soft", "single")) {
  dataset <- match.arg(dataset)
  variant <- match.arg(variant)
  prefix <- if (dataset == "ny") "brfss_stroke_new_york" else "brfss_stroke_all_states"
  results_subdir <- if (variant == "single") "tidymodels_single" else "tidymodels"
  base_dir <- file.path(root, "results", results_subdir, dataset)
  out_dir <- if (!is.null(output_subdir) && nzchar(output_subdir)) {
    file.path(base_dir, output_subdir)
  } else {
    base_dir
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  message("=== tidymodels pipeline (", variant, "): ", dataset, " ===")
  message("Outputs: ", out_dir)

  load_parsnip_engines()

  gpu_cfg <- configure_gpu_runtime(use_gpu, quick = quick)
  message(gpu_cfg$message)
  options(ml.quick = quick, ml.light_adaboost = isTRUE(light_adaboost))
  if (!is.null(adaboost_mfinal)) {
    options(ml.adaboost_mfinal = as.integer(adaboost_mfinal))
  }
  if (!is.null(adaboost_maxdepth)) {
    options(ml.adaboost_maxdepth = as.integer(adaboost_maxdepth))
  }
  ada_mfinal <- resolve_adaboost_mfinal(quick = quick, light_adaboost = light_adaboost)
  ada_maxdepth <- resolve_adaboost_maxdepth(quick = quick, light_adaboost = light_adaboost)
  if (!is.null(adaboost_mfinal) || !is.null(adaboost_maxdepth)) {
    message("AdaBoost: override mfinal=", ada_mfinal, ", maxdepth=", ada_maxdepth, ".")
  } else if (light_adaboost) {
    message("AdaBoost: light mode (mfinal=40, maxdepth=8).")
  }

  par_cfg <- init_r_parallel(n_jobs = if (parallel) n_jobs else 1L, quick = quick)
  # Fold-parallel AdaBoost disabled (adabag hangs under furrr/multicore); folds are sequential.
  options(ml.adaboost_fold_workers = 1L)
  message("AdaBoost CV: sequential survey-year folds (adabag; fold-parallel disabled).")
  use_parallel <- par_cfg$parallel
  cv_workers <- if (is.null(cv_jobs)) {
    resolve_cv_workers(n_jobs, quick = quick, max_cv = 1L)
  } else {
    max(1L, as.integer(cv_jobs))
  }
  par_limits <- apply_gpu_parallel_limits(gpu_cfg, cv_workers, use_parallel)
  cv_workers <- par_limits$cv_workers
  use_parallel_cv <- par_limits$use_parallel_cv
  use_parallel_val <- par_limits$use_parallel_val
  # CRAN xgboost segfaults in forked furrr/multicore workers during validation.
  if (use_parallel_val && identical(par_cfg$plan, "multicore")) {
    use_parallel_val <- FALSE
    par_limits$notes <- c(
      par_limits$notes,
      "Validation scoring sequential (xgboost unsafe under forked multicore)."
    )
  }
  for (note in par_limits$notes) {
    message(note)
  }
  message("CV parallelism capped at cv_workers=", cv_workers, " (incremental partial CSV)")

  splits <- load_splits(prefix, root)
  train <- splits$train
  val <- splits$val
  test <- splits$test

  feature_cols <- get_feature_columns(train)
  xy_fit <- prepare_xy(train, feature_cols)
  xy_val <- prepare_xy(val, feature_cols)
  xy_te <- prepare_xy(test, feature_cols)
  batches_fit <- train[[BATCH_COL]]
  batches_val <- val[[BATCH_COL]]
  batches_te <- test[[BATCH_COL]]

  bc <- fit_batch_correction(xy_fit$x, batches_fit)
  x_fit_bc <- apply_batch_correction(xy_fit$x, batches_fit, bc)
  x_val_bc <- apply_batch_correction(xy_val$x, batches_val, bc)
  x_test_bc <- apply_batch_correction(xy_te$x, batches_te, bc)

  plot_batch_pca(
    xy_fit$x, batches_fit,
    file.path(out_dir, "01_batch_pca_before.png"),
    "PCA before batch correction (train, by survey_year)"
  )
  plot_batch_pca(
    x_fit_bc, batches_fit,
    file.path(out_dir, "01_batch_pca_after.png"),
    "PCA after year-mean correction (train-fitted)"
  )

  n_boot <- if (quick) 200L else 1000L
  n_splits <- if (quick) 3L else 5L
  outer_lasso <- if (quick) 3L else 5L
  inner_lasso <- if (quick) 2L else 3L
  C_grid <- default_lasso_C_grid(quick = quick)
  n_perm <- if (quick) 5L else 15L

  message(
    "Nested CV for LASSO on train only (group_vfold_cv by survey_year, outer=",
    outer_lasso, ", inner=", inner_lasso, ")..."
  )
  lasso_summary <- lasso_nested_cv(
    x_fit_bc, xy_fit$y, batches_fit, feature_cols, out_dir,
    C_grid = C_grid,
    outer_splits = outer_lasso,
    inner_splits = inner_lasso,
    max_features = 30L,
    stability_threshold = 0.5
  )
  selected <- lasso_summary$stable_features
  writeLines(
    c(
      selected,
      "",
      paste0("# ", lasso_summary$selection_method),
      paste0(
        "# nested outer AUC = ",
        round(lasso_summary$outer_roc_auc_mean, 4),
        " +/- ",
        round(lasso_summary$outer_roc_auc_std, 4)
      )
    ),
    file.path(out_dir, "02_lasso_selected_features.txt")
  )
  readr::write_csv(data.frame(feature = selected), file.path(out_dir, "02_lasso_coefficients.csv"))
  message(
    "LASSO nested CV: outer AUC=", round(lasso_summary$outer_roc_auc_mean, 4),
    " (+/-", round(lasso_summary$outer_roc_auc_std, 4), "), ",
    length(selected), " stable features"
  )

  train_df <- matrix_to_df(
    x_fit_bc[, match(selected, feature_cols), drop = FALSE],
    selected, xy_fit$y, batches_fit
  )
  val_df <- matrix_to_df(
    x_val_bc[, match(selected, feature_cols), drop = FALSE],
    selected, xy_val$y, batches_val
  )
  test_df <- matrix_to_df(
    x_test_bc[, match(selected, feature_cols), drop = FALSE],
    selected, xy_te$y, batches_te
  )
  train_df$stroke <- factor(train_df$stroke, levels = c(0, 1))
  val_df$stroke <- factor(val_df$stroke, levels = c(0, 1))
  test_df$stroke <- factor(test_df$stroke, levels = c(0, 1))

  specs <- get_model_specs()
  classifier_names <- names(specs)
  message("Evaluating ", length(classifier_names), " classifiers: ", paste(classifier_names, collapse = ", "))

  options(ml.cv_workers = cv_workers)
  cv_df <- cv_evaluate_models(
    train_df, selected, out_dir,
    n_splits = n_splits, use_smote = TRUE,
    parallel = use_parallel_cv, cv_workers = cv_workers
  )
  val_sel <- select_best_model_on_val(
    train_df, val_df, selected, use_smote = FALSE, parallel = use_parallel_val
  )
  readr::write_csv(val_sel, file.path(out_dir, "03b_val_model_selection.csv"))
  writeLines(
    c(paste0("n_classifiers=", length(classifier_names)), classifier_names, ""),
    file.path(out_dir, "03c_classifiers_evaluated.txt")
  )
  best_single <- val_sel$model[1]
  message(
    "Best single model (val ROC-AUC): ", best_single,
    " (", round(val_sel$val_roc_auc[1], 4), ")"
  )

  top3_val <- val_sel$model[seq_len(min(3L, nrow(val_sel)))]
  readr::write_csv(
    data.frame(
      rank = seq_along(top3_val),
      model = top3_val,
      val_roc_auc = val_sel$val_roc_auc[match(top3_val, val_sel$model)]
    ),
    file.path(out_dir, "04b_voting_soft_members.csv")
  )

  # Ensemble train-CV benchmark (reference)
  top3_train <- cv_df$model[seq_len(min(3L, nrow(cv_df)))]
  ens_rows <- list()
  for (label in c("voting_soft_val_top3", "voting_soft_traincv_top3")) {
    members <- if (label == "voting_soft_val_top3") top3_val else top3_train
    folds <- rsample::group_vfold_cv(train_df, group = survey_year, v = n_splits)
    aucs <- numeric(length(folds$splits))
    for (i in seq_along(folds$splits)) {
      sp <- folds$splits[[i]]
      tr <- rsample::analysis(sp)
      te <- rsample::assessment(sp)
      fits <- lapply(members, fit_model_on_train, train_df = tr, predictors = selected, use_smote = TRUE)
      p <- voting_soft_predict(fits, te)
      yt <- as.integer(as.character(te$stroke))
      aucs[i] <- roc_auc_pos(yt, p)
    }
    ens_rows[[label]] <- data.frame(ensemble = label, roc_auc_mean_train_cv = mean(aucs))
  }
  readr::write_csv(dplyr::bind_rows(ens_rows), file.path(out_dir, "04_ensemble_comparison.csv"))

  # Val comparison: best single vs voting_soft
  fit_single <- fit_model_on_train(best_single, train_df, selected, use_smote = FALSE)
  fits_vote <- lapply(top3_val, fit_model_on_train, train_df = train_df, predictors = selected, use_smote = FALSE)
  p_single_val <- predict_prob_pos(fit_single, val_df)
  p_vote_val <- voting_soft_predict(fits_vote, val_df)
  yv <- as.integer(as.character(val_df$stroke))
  val_auc_single <- roc_auc_pos(yv, p_single_val)
  val_auc_vote <- roc_auc_pos(yv, p_vote_val)
  readr::write_csv(
    data.frame(
      model = c(best_single, "voting_soft"),
      val_roc_auc = c(val_auc_single, val_auc_vote),
      role = c(
        if (variant == "single") "final_single" else "best_single",
        if (variant == "single") "reference_ensemble" else "final_ensemble"
      ),
      members = c("", paste(top3_val, collapse = ","))
    ),
    file.path(out_dir, "04c_val_final_model_comparison.csv")
  )
  message(
    "Val ROC-AUC — single (", best_single, "): ", round(val_auc_single, 4),
    " | voting_soft (", paste(top3_val, collapse = ", "), "): ", round(val_auc_vote, 4)
  )

  # Final model: best single OR voting_soft; isotonic calibration on val, test once
  if (variant == "single") {
    fit_final <- fit_single
    predict_final <- function(df) predict_prob_pos(fit_final, df)
    final_label <- best_single
  } else {
    fit_final <- fits_vote
    predict_final <- function(df) voting_soft_predict(fits_vote, df)
    final_label <- "voting_soft"
  }

  p_val_uncal <- predict_final(val_df)
  p_val_cal <- isotonic_calibrate(yv, p_val_uncal, p_val_uncal)
  thr_res <- choose_classification_threshold(yv, p_val_cal)
  best_threshold <- thr_res$threshold

  p_test_uncal <- predict_final(test_df)
  p_test <- isotonic_calibrate(yv, p_val_uncal, p_test_uncal)
  yt <- as.integer(as.character(test_df$stroke))
  y_pred <- as.integer(p_test >= best_threshold)
  cm <- table(
    factor(yt, levels = c(0, 1)),
    factor(y_pred, levels = c(0, 1))
  )

  sens <- cm[2, 2] / max(sum(cm[2, ]), 1)
  spec <- cm[1, 1] / max(sum(cm[1, ]), 1)
  message(
    "Threshold (val, Youden J): ", round(best_threshold, 3),
    " | test pred+=", round(mean(y_pred), 3),
    " (prev=", round(mean(yt), 3), ") | sens=", round(sens, 3), " spec=", round(spec, 3)
  )

  boot_df <- bootstrap_metrics(yt, p_test, y_pred, n_boot = n_boot, parallel = use_parallel)
  boot_df$point[boot_df$metric == "roc_auc"] <- roc_auc_pos(yt, p_test)
  boot_df$point[boot_df$metric == "accuracy"] <- yardstick::accuracy_vec(
    truth = factor(yt, levels = c(0, 1)), estimate = factor(y_pred, levels = c(0, 1))
  )
  boot_df$point[boot_df$metric == "f1"] <- f_meas_pos(yt, y_pred)
  boot_df$point[boot_df$metric == "brier"] <- mean((p_test - yt)^2)
  boot_df$point[boot_df$metric == "pr_auc"] <- pr_auc_pos(yt, p_test)
  readr::write_csv(boot_df, file.path(out_dir, "05_test_metrics_bootstrap.csv"))

  # ROC + confusion matrix plot
  roc_df <- roc_curve_pos(yt, p_test)
  p_roc <- ggplot2::ggplot(roc_df, ggplot2::aes(1 - specificity, sensitivity)) +
    ggplot2::geom_line() +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ggplot2::labs(
      title = sprintf("ROC (test) AUC=%.3f", roc_auc_pos(yt, p_test)),
      x = "FPR", y = "TPR"
    ) +
    ggplot2::theme_minimal()
  cm_df <- as.data.frame(cm)
  names(cm_df) <- c("Actual", "Predicted", "n")
  p_cm <- ggplot2::ggplot(cm_df, ggplot2::aes(Predicted, Actual, fill = n)) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(ggplot2::aes(label = n), color = "white") +
    ggplot2::labs(title = sprintf("Test CM — %s (thr=%.3f)", final_label, best_threshold)) +
    ggplot2::theme_minimal()
  png(file.path(out_dir, "06_confusion_matrix_test.png"), width = 10, height = 4, units = "in", res = 150)
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(nrow = 1, ncol = 2)))
  print(p_cm, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1))
  print(p_roc, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 2))
  dev.off()

  # Calibration curve (unique breaks when many tied scores, e.g. random forest)
  cal_df <- calibration_bins_df(yt, p_test, n_bins = 10L)
  p_cal <- ggplot2::ggplot(cal_df, ggplot2::aes(mean_pred, frac_pos)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ggplot2::labs(title = "Calibration (test)", x = "Mean predicted prob", y = "Fraction positive") +
    ggplot2::theme_minimal()
  ggplot2::ggsave(file.path(out_dir, "07_calibration_test.png"), p_cal, width = 5, height = 4, dpi = 150)

  # Per-year violin
  test_plot <- test_df
  test_plot$score <- p_test
  p_violin <- ggplot2::ggplot(test_plot, ggplot2::aes(factor(survey_year), score, fill = stroke)) +
    ggplot2::geom_violin(trim = FALSE, alpha = 0.7) +
    ggplot2::geom_hline(yintercept = best_threshold, linetype = "dashed", color = "gray40") +
    ggplot2::labs(title = "Predicted P(stroke) by survey_year (test)", x = "survey_year", y = "score") +
    ggplot2::theme_minimal()
  ggplot2::ggsave(file.path(out_dir, "08_per_year_score_violin.png"), p_violin, width = 10, height = 4, dpi = 150)

  # Permutation importance on validation (uncalibrated final model)
  set.seed(42)
  base_auc <- roc_auc_pos(yv, p_val_uncal)
  perm_rows <- lapply(selected, function(feat) {
    imp <- numeric(n_perm)
    for (r in seq_len(n_perm)) {
      vd <- val_df
      vd[[feat]] <- sample(vd[[feat]])
      imp[r] <- roc_auc_pos(yv, predict_final(vd))
    }
    data.frame(
      feature = feat,
      importance_mean = base_auc - mean(imp),
      importance_std = stats::sd(imp)
    )
  })
  perm_df <- dplyr::bind_rows(perm_rows) |> dplyr::arrange(dplyr::desc(importance_mean))
  readr::write_csv(perm_df, file.path(out_dir, "09_permutation_importance.csv"))
  p_perm <- ggplot2::ggplot(head(perm_df, 20), ggplot2::aes(importance_mean, reorder(feature, importance_mean))) +
    ggplot2::geom_col() +
    ggplot2::labs(title = "Permutation importance (validation, ROC-AUC)") +
    ggplot2::theme_minimal()
  ggplot2::ggsave(file.path(out_dir, "09_permutation_importance.png"), p_perm, width = 8, height = 6, dpi = 150)

  shap_msg <- if (variant == "single") {
    paste(
      "SHAP not computed in R for", best_single, ".",
      "See Python results/ml/", dataset, "/10_shap_summary.png if available."
    )
  } else {
    paste(
      "SHAP not computed for voting_soft ensemble.",
      "See Python results/ml/", dataset, "/ or run SHAP per member:", paste(top3_val, collapse = ", ")
    )
  }
  writeLines(shap_msg, file.path(out_dir, "10_shap_summary.txt"))

  test_auc <- roc_auc_pos(yt, p_test)
  test_bal <- bal_accuracy_bin(yt, y_pred)

  config <- list(
    dataset = dataset,
    pipeline = "tidymodels",
    pipeline_variant = variant,
    results_subdir = results_subdir,
    framework = "https://www.tidymodels.org",
    n_classifiers_evaluated = length(classifier_names),
    classifiers_evaluated = classifier_names,
    selected_features = selected,
    lasso_nested_cv = lasso_summary,
    best_model = if (variant == "single") best_single else "voting_soft",
    final_model = if (variant == "single") best_single else "voting_soft",
    voting_members = if (variant == "single") NULL else top3_val,
    best_single_model = best_single,
    val_roc_auc_single = val_auc_single,
    val_roc_auc_voting_soft = val_auc_vote,
    classification_threshold = best_threshold,
    threshold_tuning = thr_res$info,
    lasso_data = "train_only",
    cv_scheme = "group_vfold_cv_by_survey_year",
    final_fit = "train_only",
    final_smote = FALSE,
    probability_calibration = "isotonic_on_val",
    interpretability_data = "validation_only",
    n_train_fit = nrow(train_df),
    n_val_threshold = nrow(val_df),
    n_test = nrow(test_df),
    test_roc_auc = test_auc,
    test_balanced_accuracy = test_bal,
    n_jobs = n_jobs,
    parallel_workers = par_cfg$workers,
    parallel_plan = par_cfg$plan,
    use_gpu_requested = gpu_cfg$requested,
    use_gpu_enabled = gpu_cfg$enabled,
    gpu_backend = gpu_cfg$backend,
    ml_python = if (!is.null(gpu_cfg$python) && nzchar(gpu_cfg$python)) gpu_cfg$python else NULL,
    gpu_parallel_notes = par_limits$notes,
    light_adaboost = isTRUE(light_adaboost),
    adaboost_mfinal = ada_mfinal,
    adaboost_maxdepth = ada_maxdepth,
    output_subdir = if (!is.null(output_subdir) && nzchar(output_subdir)) {
      output_subdir
    } else {
      NULL
    }
  )
  jsonlite::write_json(config, file.path(out_dir, "pipeline_config.json"), pretty = TRUE, auto_unbox = TRUE)

  if (variant == "single") {
    report <- sprintf(
      paste0(
        "# tidymodels Pipeline Report — %s (single)\n\n",
        "## Final model\n- Best single: **%s** (val ROC-AUC = %.4f)\n",
        "- Reference voting_soft val AUC = %.4f (%s)\n\n",
        "## Test\n- ROC-AUC: %.4f\n- Balanced accuracy: %.4f\n- Threshold: %.4f\n\n",
        "Benchmark vs `results/ml/` and `results/tidymodels/` (voting_soft).\n"
      ),
      dataset, best_single, val_auc_single, val_auc_vote, paste(top3_val, collapse = ", "),
      test_auc, test_bal, best_threshold
    )
    model_obj <- list(
      variant = "single",
      model_name = best_single,
      fit = fit_single,
      threshold = best_threshold,
      isotonic_anchor = list(p_val = p_val_uncal, y_val = yv)
    )
  } else {
    report <- sprintf(
      paste0(
        "# tidymodels Pipeline Report — %s (voting_soft)\n\n",
        "## Final model\n- Soft voting over: %s\n",
        "- Val AUC single **%s** = %.4f; voting_soft = %.4f\n\n",
        "## Test\n- ROC-AUC: %.4f\n- Balanced accuracy: %.4f\n- Threshold: %.4f\n\n",
        "Benchmark vs `results/ml/` and `results/ml_voting_soft/`.\n"
      ),
      dataset, paste(top3_val, collapse = ", "), best_single, val_auc_single, val_auc_vote,
      test_auc, test_bal, best_threshold
    )
    model_obj <- list(
      variant = "voting_soft",
      fits = fits_vote,
      members = top3_val,
      threshold = best_threshold,
      isotonic_anchor = list(p_val = p_val_uncal, y_val = yv)
    )
  }
  writeLines(report, file.path(out_dir, "WORKFLOW_REPORT.md"))

  saveRDS(model_obj, file.path(out_dir, "11_best_model.rds"))

  message("Done. Outputs: ", out_dir)
  invisible(config)
}
