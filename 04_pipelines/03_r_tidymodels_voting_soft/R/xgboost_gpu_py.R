# GPU gradient boosting via Python xgboost (reticulate) when R xgboost lacks CUDA.

.py_xgb_env <- new.env(parent = emptyenv())

ensure_python_xgboost <- function() {
  if (isTRUE(.py_xgb_env$ready)) {
    return(invisible(TRUE))
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' required for GPU (--use-gpu). Install with install.packages('reticulate').")
  }
  py <- resolve_ml_python()
  reticulate::use_python(py, required = TRUE)
  .py_xgb_env$np <- reticulate::import("numpy")
  .py_xgb_env$SMOTE <- reticulate::import("imblearn.over_sampling")$SMOTE
  .py_xgb_env$XGBClassifier <- reticulate::import("xgboost")$XGBClassifier
  .py_xgb_env$ready <- TRUE
  invisible(TRUE)
}

fit_xgboost_py <- function(train_df, predictors, use_smote = FALSE, spec = list()) {
  ensure_python_xgboost()
  np <- .py_xgb_env$np
  quick <- isTRUE(getOption("ml.quick", FALSE))
  n_est <- if (!is.null(spec$n_estimators)) as.integer(spec$n_estimators) else if (quick) 100L else 200L
  max_dep <- if (!is.null(spec$max_depth)) as.integer(spec$max_depth) else 4L

  rec <- make_recipe(predictors, train_df, use_smote = FALSE)
  prep <- recipes::prep(rec, training = train_df)
  baked <- recipes::bake(prep, new_data = train_df)
  X <- as.matrix(baked[, predictors, drop = FALSE])
  storage.mode(X) <- "double"
  y <- as.integer(as.character(baked[[TARGET]]))

  if (use_smote) {
    n_pos <- sum(y == 1L)
    k <- max(1L, min(5L, n_pos - 1L))
    sm <- .py_xgb_env$SMOTE(random_state = 42L, k_neighbors = as.integer(k))
    res <- sm$fit_resample(X, y)
    X <- res[[1L]]
    y <- res[[2L]]
  }

  clf <- .py_xgb_env$XGBClassifier(
    n_estimators = as.integer(n_est),
    max_depth = as.integer(max_dep),
    learning_rate = 0.05,
    device = "cuda",
    tree_method = "hist",
    eval_metric = "logloss",
    random_state = 42L
  )
  clf$fit(X, y)
  structure(
    list(clf = clf, prep = prep, predictors = predictors),
    class = "xgboost_py_fit"
  )
}

predict_xgboost_py_prob <- function(obj, new_df) {
  ensure_python_xgboost()
  baked <- recipes::bake(obj$prep, new_data = new_df)
  X <- as.matrix(baked[, obj$predictors, drop = FALSE])
  storage.mode(X) <- "double"
  as.numeric(obj$clf$predict_proba(X)[, 2L])
}
