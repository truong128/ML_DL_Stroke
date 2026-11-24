# GPU runtime helpers (XGBoost on CUDA via Python when R xgboost has no GPU build)

resolve_ml_python <- function() {
  cand <- c(
    Sys.getenv("ML_PYTHON", unset = ""),
    file.path(Sys.getenv("HOME"), ".venvs", "15_7_stroke", "bin", "python3"),
    Sys.which("python3")
  )
  cand <- unique(cand[nzchar(cand)])
  for (p in cand) {
    if (file.exists(p)) {
      return(normalizePath(p, mustWork = FALSE))
    }
  }
  stop("No Python found. Set ML_PYTHON to ~/.venvs/15_7_stroke/bin/python3")
}

nvidia_gpu_visible <- function() {
  out <- tryCatch(
    system2("nvidia-smi", "-L", stdout = TRUE, stderr = FALSE),
    error = function(e) character()
  )
  length(out) > 0L && any(grepl("GPU", out))
}

r_xgboost_cuda_available <- function() {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    return(FALSE)
  }
  ok <- tryCatch({
    d <- data.matrix(data.frame(x1 = rnorm(20), x2 = rnorm(20)))
    y <- rbinom(20, 1, 0.3)
    dm <- xgboost::xgb.DMatrix(d, label = y)
    fit <- xgboost::xgb.train(
      params = list(
        objective = "binary:logistic",
        device = "cuda",
        max_depth = 2
      ),
      data = dm,
      nrounds = 2,
      verbose = 0
    )
    params <- fit$params
    is.list(params) && identical(params$device, "cuda")
  }, error = function(e) FALSE)
  isTRUE(ok)
}

python_xgboost_cuda_available <- function() {
  py <- resolve_ml_python()
  cmd <- sprintf(
    paste0(
      "%s -c \"import xgboost as xgb; import numpy as np; ",
      "X=np.random.randn(80,5).astype('float32'); y=np.random.randint(0,2,80); ",
      "m=xgb.XGBClassifier(n_estimators=2, device='cuda', max_depth=2); ",
      "m.fit(X,y); print('ok')\""
    ),
    shQuote(py)
  )
  out <- tryCatch(system(cmd, intern = TRUE), error = function(e) character())
  any(out == "ok")
}

#' Configure GPU for this pipeline run; sets options ml.use_gpu / ml.quick / ml.gpu_cfg.
configure_gpu_runtime <- function(use_gpu_requested = FALSE, quick = FALSE) {
  cfg <- list(
    requested = isTRUE(use_gpu_requested),
    enabled = FALSE,
    backend = "cpu",
    python = NA_character_,
    nvidia_visible = nvidia_gpu_visible(),
    r_xgboost_cuda = FALSE,
    python_xgboost_cuda = FALSE,
    message = ""
  )
  if (!cfg$requested) {
    cfg$message <- "GPU: disabled (CPU-only run)."
    options(ml.use_gpu = FALSE, ml.quick = quick, ml.gpu_cfg = cfg)
    return(cfg)
  }
  if (!cfg$nvidia_visible) {
    cfg$message <- "GPU: --use-gpu set but nvidia-smi found no GPU; using CPU."
    options(ml.use_gpu = FALSE, ml.quick = quick, ml.gpu_cfg = cfg)
    return(cfg)
  }
  cfg$r_xgboost_cuda <- r_xgboost_cuda_available()
  cfg$python_xgboost_cuda <- python_xgboost_cuda_available()
  if (cfg$r_xgboost_cuda) {
    cfg$enabled <- TRUE
    cfg$backend <- "r_xgboost_cuda"
    cfg$message <- "GPU: enabled via R xgboost (device=cuda) for gradient_boosting."
  } else if (cfg$python_xgboost_cuda) {
    cfg$enabled <- TRUE
    cfg$backend <- "python_xgboost_cuda"
    cfg$python <- resolve_ml_python()
    cfg$message <- paste0(
      "GPU: enabled via Python xgboost (reticulate, ",
      cfg$python, ") for gradient_boosting. ",
      "CV/val parallelism limited to 1 worker (fork+CUDA unsafe)."
    )
  } else {
    cfg$message <- paste0(
      "GPU: CUDA GPU visible but xgboost GPU unavailable in R and Python; ",
      "using CPU. Install xgboost in ", resolve_ml_python(), " (pip install xgboost)."
    )
  }
  options(ml.use_gpu = cfg$enabled, ml.quick = quick, ml.gpu_cfg = cfg)
  cfg
}

spec_type <- function(spec) {
  if (is.list(spec) && !is.null(spec$type)) {
    as.character(spec$type[[1L]])
  } else {
    NULL
  }
}

is_custom_spec <- function(spec) {
  !is.null(spec_type(spec))
}

#' Python reticulate + CUDA must not run inside forked furrr/multicore workers (segfault).
gpu_requires_sequential_parallel <- function(gpu_cfg = getOption("ml.gpu_cfg", list())) {
  isTRUE(gpu_cfg$enabled) &&
    identical(gpu_cfg$backend, "python_xgboost_cuda")
}

#' Cap CV / val parallelism when GPU uses Python bridge.
apply_gpu_parallel_limits <- function(gpu_cfg, cv_workers, use_parallel) {
  out <- list(
    cv_workers = cv_workers,
    use_parallel_cv = isTRUE(use_parallel) && cv_workers > 1L,
    use_parallel_val = isTRUE(use_parallel),
    notes = character()
  )
  if (!gpu_requires_sequential_parallel(gpu_cfg)) {
    return(out)
  }
  if (out$cv_workers > 1L) {
    out$notes <- c(
      out$notes,
      paste0(
        "GPU (Python/reticulate): CV forced to cv_workers=1 ",
        "(forked multicore + CUDA is unsafe; was ", cv_workers, ")."
      )
    )
    out$cv_workers <- 1L
    out$use_parallel_cv <- FALSE
  }
  if (out$use_parallel_val) {
    out$notes <- c(
      out$notes,
      "GPU (Python/reticulate): validation model scoring run sequentially."
    )
    out$use_parallel_val <- FALSE
  }
  out
}
