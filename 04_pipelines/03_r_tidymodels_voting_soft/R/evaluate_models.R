# Cross-validated model comparison and validation selection

cv_fold_seed <- function(name) {
  # Stable folds per model name (same in sequential and parallel workers).
  sum(utf8ToInt(name)) %% .Machine$integer.max
}

cv_evaluate_one_fold <- function(i, folds, spec, predictors, use_smote, name) {
  set.seed(cv_fold_seed(name) + i)
  sp <- folds$splits[[i]]
  tr <- rsample::analysis(sp)
  te <- rsample::assessment(sp)
  fit <- fit_model_from_spec(spec, tr, predictors, use_smote = use_smote)
  p <- predict_prob_pos(fit, te)
  yt <- as.integer(as.character(te$stroke))
  pred <- as.integer(p >= 0.5)
  list(
    auc = roc_auc_pos(yt, p),
    f1 = f_meas_pos(yt, pred),
    bal = bal_accuracy_bin(yt, pred),
    pr = pr_auc_pos(yt, p)
  )
}

adabag_cv_fold_workers <- function(n_splits) {
  w <- getOption("ml.adaboost_fold_workers", 1L)
  max(1L, min(as.integer(w), as.integer(n_splits)))
}

cv_evaluate_one_model <- function(
    name, spec, train_df, predictors, n_splits, use_smote) {
  message("[CV start] ", name, " (pid ", Sys.getpid(), ") ", format(Sys.time(), "%H:%M:%S"))
  out <- tryCatch({
  set.seed(cv_fold_seed(name))
  folds <- rsample::group_vfold_cv(train_df, group = survey_year, v = n_splits)
  n_folds <- length(folds$splits)
  # adabag::boosting busy-loops indefinitely under furrr/multicore fold workers
  # (even mfinal=1). Always run AdaBoost CV folds sequentially.
  if (identical(spec_type(spec), "adabag")) {
    message("[CV] adaboost: ", n_folds, " sequential folds (adabag; no fold-parallel)")
  }
  aucs <- f1s <- bals <- prs <- numeric(n_folds)
  for (i in seq_len(n_folds)) {
    one <- cv_evaluate_one_fold(i, folds, spec, predictors, use_smote, name)
    aucs[i] <- one$auc
    f1s[i] <- one$f1
    bals[i] <- one$bal
    prs[i] <- one$pr
  }
  res <- data.frame(
    model = name,
    roc_auc_mean = mean(aucs, na.rm = TRUE),
    roc_auc_std = stats::sd(aucs, na.rm = TRUE),
    f1_mean = mean(f1s, na.rm = TRUE),
    balanced_acc_mean = mean(bals, na.rm = TRUE),
    pr_auc_mean = mean(prs, na.rm = TRUE)
  )
  message("[CV done] ", name, " AUC=", round(res$roc_auc_mean, 4))
  res
  }, error = function(e) {
    message("CV failed for ", name, ": ", conditionMessage(e))
    data.frame(
      model = name,
      roc_auc_mean = NA_real_,
      roc_auc_std = NA_real_,
      f1_mean = NA_real_,
      balanced_acc_mean = NA_real_,
      pr_auc_mean = NA_real_
    )
  })
  out
}

append_cv_partial_row <- function(row, path) {
  row <- as.data.frame(row)
  if (!file.exists(path)) {
    readr::write_csv(row, path)
  } else {
    readr::write_csv(row, path, append = TRUE)
  }
}

cv_evaluate_models <- function(
    train_df, predictors, out_dir,
    n_splits = 5L, use_smote = TRUE, parallel = TRUE, cv_workers = 4L) {
  specs <- get_model_specs()
  nms <- names(specs)
  cv_workers <- max(1L, as.integer(cv_workers))
  use_par <- isTRUE(parallel) && cv_workers > 1L
  partial_path <- file.path(out_dir, "03_cv_model_comparison_partial.csv")
  final_path <- file.path(out_dir, "03_cv_model_comparison.csv")
  if (file.exists(partial_path)) {
    unlink(partial_path)
  }

  message(
    "Cross-validating ", length(nms), " models (cv_workers=", cv_workers,
    ", incremental save -> ", basename(partial_path), ")..."
  )

  eval_one <- function(nm) {
    cv_evaluate_one_model(nm, specs[[nm]], train_df, predictors, n_splits, use_smote)
  }

  rows <- if (use_par) {
    # Batched parallel CV: append partial CSV after each batch (not only after all 16).
    n_batches <- ceiling(length(nms) / cv_workers)
    message(
      "Parallel CV in ", n_batches, " batch(es) of up to ", cv_workers,
      " models; partial CSV updates after each batch."
    )
    batch_rows <- vector("list", length(nms))
    idx <- 0L
    for (b in seq_len(n_batches)) {
      i0 <- (b - 1L) * cv_workers + 1L
      i1 <- min(b * cv_workers, length(nms))
      chunk <- nms[i0:i1]
      message(
        "[CV batch ", b, "/", n_batches, "] ",
        paste(chunk, collapse = ", "),
        " ", format(Sys.time(), "%H:%M:%S")
      )
      chunk_res <- with_parallel_plan(min(cv_workers, length(chunk)), {
        furrr::future_map(
          chunk, eval_one,
          .options = furrr::furrr_options(seed = FALSE)
        )
      })
      for (row in chunk_res) {
        idx <- idx + 1L
        batch_rows[[idx]] <- row
        append_cv_partial_row(row, partial_path)
      }
    }
    batch_rows
  } else {
    lapply(nms, function(nm) {
      row <- eval_one(nm)
      append_cv_partial_row(row, partial_path)
      row
    })
  }

  df <- dplyr::bind_rows(rows) |> dplyr::arrange(dplyr::desc(roc_auc_mean))
  readr::write_csv(df, final_path)
  df
}

val_score_one_model <- function(nm, spec, train_df, val_df, predictors, use_smote) {
  message("[val] ", nm, " ", format(Sys.time(), "%H:%M:%S"))
  tryCatch({
    fit <- fit_model_from_spec(spec, train_df, predictors, use_smote = use_smote)
    p <- predict_prob_pos(fit, val_df)
    yv <- as.integer(as.character(val_df$stroke))
    data.frame(
      model = nm,
      val_roc_auc = roc_auc_pos(yv, p),
      val_pr_auc = pr_auc_pos(yv, p)
    )
  }, error = function(e) {
    message("Val failed for ", nm, ": ", conditionMessage(e))
    data.frame(model = nm, val_roc_auc = NA_real_, val_pr_auc = NA_real_)
  })
}

select_best_model_on_val <- function(
    train_df, val_df, predictors, use_smote = FALSE, parallel = TRUE) {
  specs <- get_model_specs()
  nms <- names(specs)
  message("Validation scoring for ", length(nms), " models (parallel=", parallel, ")...")
  rows <- parallel_map(
    nms,
    function(nm) val_score_one_model(nm, specs[[nm]], train_df, val_df, predictors, use_smote),
    parallel = parallel, use_seed = FALSE
  )
  dplyr::bind_rows(rows) |> dplyr::arrange(dplyr::desc(val_roc_auc))
}

fit_model_on_train <- function(model_name, train_df, predictors, use_smote = FALSE) {
  spec <- get_model_specs()[[model_name]]
  y_tr <- as.integer(as.character(train_df$stroke))
  fit_model_from_spec(spec, train_df, predictors, use_smote = use_smote)
}

voting_soft_predict <- function(fits, new_df) {
  probs <- vapply(fits, predict_prob_pos, numeric(nrow(new_df)), new_df = new_df)
  rowMeans(probs)
}
