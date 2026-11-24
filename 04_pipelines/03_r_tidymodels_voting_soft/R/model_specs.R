# Sixteen supervised learners — mirrors get_classifiers() in run_ml_pipeline.py

resolve_adaboost_mfinal <- function(quick = FALSE, light_adaboost = FALSE) {
  ov <- getOption("ml.adaboost_mfinal", NULL)
  if (!is.null(ov)) {
    return(as.integer(ov))
  }
  if (isTRUE(quick)) {
    25L
  } else if (isTRUE(light_adaboost)) {
    40L
  } else {
    100L
  }
}

#' Register parsnip back-end packages (discrim, naivebayes, baguette, kknn).
#' Call once in the main R process before parallel CV/validation.
load_parsnip_engines <- function() {
  if (isTRUE(getOption("ml.parsnip_engines_loaded", FALSE))) {
    return(invisible(TRUE))
  }
  pkgs <- c("discrim", "naivebayes", "baguette", "kknn")
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Missing R packages for parsnip engines: ", paste(missing, collapse = ", "),
      ". Run: Rscript tidymodels_pipeline/setup.R"
    )
  }
  suppressPackageStartupMessages({
    loadNamespace("discrim")
    loadNamespace("naivebayes")
    loadNamespace("baguette")
    loadNamespace("kknn")
  })
  options(ml.parsnip_engines_loaded = TRUE)
  invisible(TRUE)
}

resolve_adaboost_maxdepth <- function(quick = FALSE, light_adaboost = FALSE) {
  ov <- getOption("ml.adaboost_maxdepth", NULL)
  if (!is.null(ov)) {
    return(as.integer(ov))
  }
  if (isTRUE(quick)) {
    6L
  } else if (isTRUE(light_adaboost)) {
    8L
  } else {
    12L
  }
}

make_recipe <- function(predictors, data_template, use_smote = FALSE) {
  cols <- c(TARGET, predictors)
  template <- data_template[, cols, drop = FALSE]
  r <- recipes::recipe(
    stats::as.formula(paste(TARGET, "~", paste(predictors, collapse = " + "))),
    data = template
  ) |>
    recipes::step_impute_median(recipes::all_numeric_predictors()) |>
    recipes::step_zv(recipes::all_numeric_predictors()) |>
    recipes::step_normalize(recipes::all_numeric_predictors())
  if (use_smote) {
    r <- r |> themis::step_smote(!!rlang::sym(TARGET), over_ratio = 0.5)
  }
  r
}

get_model_specs <- function() {
  load_parsnip_engines()
  use_gpu <- isTRUE(getOption("ml.use_gpu", FALSE))
  gpu_cfg <- getOption("ml.gpu_cfg", list(backend = "cpu"))
  quick <- isTRUE(getOption("ml.quick", FALSE))
  gb_spec <- if (use_gpu && identical(gpu_cfg$backend, "python_xgboost_cuda")) {
    list(
      type = "xgboost_gpu_py",
      n_estimators = if (quick) 100L else 200L,
      max_depth = 4L
    )
  } else if (use_gpu && identical(gpu_cfg$backend, "r_xgboost_cuda")) {
    parsnip::boost_tree(
      trees = if (quick) 100L else 200L, tree_depth = 4, learn_rate = 0.05
    ) |>
      parsnip::set_engine(
        "xgboost",
        eval_metric = "logloss",
        tree_method = "hist",
        device = "cuda",
        nthread = 1
      ) |>
      parsnip::set_mode("classification")
  } else {
    parsnip::boost_tree(
      trees = if (quick) 100L else 200L, tree_depth = 4, learn_rate = 0.05
    ) |>
      parsnip::set_engine("xgboost", eval_metric = "logloss", nthread = 1) |>
      parsnip::set_mode("classification")
  }
  light_ada <- isTRUE(getOption("ml.light_adaboost", FALSE))
  adaboost_spec <- list(
    type = "adabag",
    mfinal = resolve_adaboost_mfinal(quick = quick, light_adaboost = light_ada),
    maxdepth = resolve_adaboost_maxdepth(quick = quick, light_adaboost = light_ada)
  )
  list(
    logistic_regression = parsnip::logistic_reg() |>
      parsnip::set_engine("glm") |>
      parsnip::set_mode("classification"),

    multinom_reg = parsnip::multinom_reg(penalty = 0.01, mixture = 0) |>
      parsnip::set_engine("glmnet") |>
      parsnip::set_mode("classification"),

    svm_rbf = parsnip::svm_rbf(
      cost = 1, rbf_sigma = 0.05
    ) |>
      parsnip::set_mode("classification"),

    svm_linear = parsnip::svm_linear(
      cost = 1
    ) |>
      parsnip::set_mode("classification"),

    decision_tree = parsnip::decision_tree(
      tree_depth = 12, min_n = 5
    ) |>
      parsnip::set_engine("rpart") |>
      parsnip::set_mode("classification"),

    random_forest = parsnip::rand_forest(
      trees = 300, min_n = 5
    ) |>
      parsnip::set_engine(
        "ranger",
        probability = TRUE,
        class.weights = c("0" = 1, "1" = 1),
        max.depth = 12,
        importance = "none",
        num.threads = 1
      ) |>
      parsnip::set_mode("classification"),

    gradient_boosting = gb_spec,

    extra_trees = parsnip::rand_forest(
      trees = 300, min_n = 5
    ) |>
      parsnip::set_engine(
        "ranger",
        probability = TRUE,
        splitrule = "extratrees",
        max.depth = 12,
        class.weights = c("0" = 1, "1" = 1),
        num.threads = 1
      ) |>
      parsnip::set_mode("classification"),

    adaboost = adaboost_spec,

    # parsnip 1.6+: baguette engine removed; rpart backend (default bag count in engine).
    bag_tree = parsnip::bag_tree(
      tree_depth = 8, min_n = 10
    ) |>
      parsnip::set_engine("rpart") |>
      parsnip::set_mode("classification"),

    bagging = parsnip::rand_forest(
      trees = 100, min_n = 5
    ) |>
      parsnip::set_engine(
        "ranger",
        replace = TRUE,
        sample.fraction = 1,
        probability = TRUE,
        max.depth = 12,
        num.threads = 1
      ) |>
      parsnip::set_mode("classification"),

    knn = parsnip::nearest_neighbor(neighbors = 15) |>
      parsnip::set_engine("kknn") |>
      parsnip::set_mode("classification"),

    lda = parsnip::discrim_linear() |>
      parsnip::set_engine("MASS") |>
      parsnip::set_mode("classification"),

    # sparsediscrim shrink_cov fails on this cohort; MASS matches sklearn QDA more reliably.
    qda = parsnip::discrim_quad() |>
      parsnip::set_engine("MASS") |>
      parsnip::set_mode("classification"),

    naive_bayes = parsnip::naive_Bayes() |>
      parsnip::set_engine("naivebayes") |>
      parsnip::set_mode("classification"),

    mlp = parsnip::mlp(
      hidden_units = 32, epochs = 200
    ) |>
      parsnip::set_engine(
        "nnet", size = 32, MaxNWts = 15000, maxit = 200, trace = FALSE
      ) |>
      parsnip::set_mode("classification")
  )
}

fit_workflow <- function(spec, train_df, predictors, use_smote = FALSE) {
  rec <- make_recipe(predictors, train_df, use_smote = use_smote)
  workflows::workflow() |>
    workflows::add_recipe(rec) |>
    workflows::add_model(spec) |>
    parsnip::fit(train_df)
}

fit_model_from_spec <- function(spec, train_df, predictors, use_smote = FALSE) {
  load_parsnip_engines()
  st <- spec_type(spec)
  if (identical(st, "adabag")) {
    return(fit_adabag(train_df, predictors, spec, use_smote = use_smote))
  }
  if (identical(st, "xgboost_gpu_py")) {
    return(fit_xgboost_py(train_df, predictors, use_smote = use_smote, spec = spec))
  }
  fit_workflow(spec, train_df, predictors, use_smote = use_smote)
}

predict_prob_pos <- function(fitted, new_df) {
  if (inherits(fitted, "adabag_fit")) {
    return(predict_adabag_prob(fitted, new_df))
  }
  if (inherits(fitted, "xgboost_py_fit")) {
    return(predict_xgboost_py_prob(fitted, new_df))
  }
  p <- tryCatch(predict(fitted, new_df, type = "prob"), error = function(e) NULL)
  if (!is.null(p) && ".pred_1" %in% names(p)) {
    return(as.numeric(p$.pred_1))
  }
  cls <- predict(fitted, new_df, type = "class")
  as.numeric(as.character(cls$.pred_class))
}

# recipes::bake() returns a tibble; adabag::boosting() can busy-loop on tibbles
# when a multicore future plan is active — coerce to data.frame.
bake_data_frame <- function(prep, new_data) {
  as.data.frame(recipes::bake(prep, new_data = new_data))
}

# --- AdaBoost (adabag) — sklearn AdaBoostClassifier analogue ---
fit_adabag <- function(
    train_df, predictors, spec = list(type = "adabag"), use_smote = FALSE) {
  mfinal <- if (!is.null(spec$mfinal)) as.integer(spec$mfinal) else 100L
  maxdepth <- if (!is.null(spec$maxdepth)) as.integer(spec$maxdepth) else 12L
  form <- stats::as.formula(paste(TARGET, "~", paste(predictors, collapse = " + ")))
  imp <- make_recipe(predictors, train_df, use_smote = use_smote)
  prep <- recipes::prep(imp, training = train_df)
  baked <- bake_data_frame(prep, train_df)
  y <- factor(baked[[TARGET]], levels = c("0", "1"))
  w <- ifelse(as.integer(as.character(y)) == 1,
              sum(y == "0") / max(sum(y == "1"), 1), 1)
  fit <- adabag::boosting(
    formula = form,
    data = baked,
    mfinal = mfinal,
    coeflearn = "Breiman",
    control = rpart::rpart.control(maxdepth = maxdepth),
    weights = w
  )
  structure(
    list(fit = fit, prep = prep, predictors = predictors),
    class = "adabag_fit"
  )
}

predict_adabag_prob <- function(obj, new_df) {
  baked <- bake_data_frame(obj$prep, new_df)
  pr <- predict(obj$fit, newdata = baked, type = "prob")
  # adabag returns a list; class-1 probabilities are in $prob[, 2] (not pr[, "1"]).
  if (is.list(pr) && !is.null(pr$prob)) {
    prob <- pr$prob
    if (is.matrix(prob) && ncol(prob) >= 2L) {
      return(as.numeric(prob[, 2L]))
    }
  }
  if (is.data.frame(pr) && "1" %in% names(pr)) {
    return(as.numeric(pr[["1"]]))
  }
  stop("adabag predict: unexpected probability structure")
}
