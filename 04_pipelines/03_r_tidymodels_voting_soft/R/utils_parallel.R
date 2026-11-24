# Parallel backend for tidymodels pipeline (mirrors Python N_JOBS / --n-jobs)

resolve_n_workers <- function(n_jobs = NULL) {
  if (is.null(n_jobs)) {
    nj <- Sys.getenv("ML_N_JOBS", unset = "-1")
    n_jobs <- suppressWarnings(as.integer(nj))
  }
  if (is.na(n_jobs) || n_jobs == 0L) {
    n_jobs <- -1L
  }
  cores <- parallel::detectCores(logical = TRUE)
  workers <- if (n_jobs < 0L) {
    cores + n_jobs + 1L # e.g. -1 => all cores, -2 => cores - 1
  } else {
    as.integer(n_jobs)
  }
  max(1L, min(workers, cores))
}

#' Cap parallel CV models (default 1 = sequential + incremental CSV; use --cv-jobs 4 to speed up).
resolve_cv_workers <- function(n_jobs = NULL, quick = FALSE, max_cv = 1L) {
  if (quick) {
    return(1L)
  }
  max(1L, min(resolve_n_workers(n_jobs), as.integer(max_cv)))
}

with_parallel_plan <- function(workers, expr) {
  if (workers <= 1L || !requireNamespace("future", quietly = TRUE)) {
    return(force(expr))
  }
  oplan <- future::plan()
  on.exit(future::plan(oplan), add = TRUE)
  if (.Platform$OS.type == "windows") {
    future::plan(future::multisession, workers = workers)
  } else {
    future::plan(future::multicore, workers = workers)
  }
  force(expr)
}

set_blas_threads <- function(n = 1L) {
  n <- as.integer(n)
  Sys.setenv(
    OMP_NUM_THREADS = n,
    MKL_NUM_THREADS = n,
    OPENBLAS_NUM_THREADS = n,
    VECLIB_MAXIMUM_THREADS = n,
    NUMEXPR_NUM_THREADS = n
  )
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    try(RhpcBLASctl::blas_set_num_threads(as.integer(n)), silent = TRUE)
  }
}

#' Initialize furrr/future plan; cap BLAS threads so workers do not oversubscribe.
init_r_parallel <- function(n_jobs = NULL, quick = FALSE) {
  cores <- parallel::detectCores(logical = TRUE)
  if (quick) {
    set_blas_threads(1L)
    message("Quick mode: sequential execution (parallel disabled).")
    return(list(parallel = FALSE, workers = 1L, plan = "sequential"))
  }
  if (!requireNamespace("future", quietly = TRUE)) {
    message("Package 'future' not installed; running sequentially.")
    return(list(parallel = FALSE, workers = 1L, plan = "sequential"))
  }
  workers <- resolve_n_workers(n_jobs)
  set_blas_threads(1L)
  plan_name <- if (.Platform$OS.type == "windows") {
    future::plan(future::multisession, workers = workers)
    "multisession"
  } else {
    future::plan(future::multicore, workers = workers)
    "multicore"
  }
  if (requireNamespace("furrr", quietly = TRUE)) {
    options(future.globals.maxSize = 8 * 1024^3)
  }
  # CV/bootstrap set seeds explicitly (cv_fold_seed, bootstrap_one).
  options(future.rng.onMisuse = "ignore")
  message(
    "Parallel plan: ", plan_name, ", workers = ", workers,
    " (", cores, " CPUs detected; 1 BLAS thread per worker)"
  )
  list(parallel = workers > 1L, workers = workers, plan = plan_name)
}

parallel_map <- function(x, fn, ..., parallel = TRUE, use_seed = FALSE) {
  if (parallel && requireNamespace("furrr", quietly = TRUE)) {
    furrr::future_map(
      x, fn, ...,
      .options = furrr::furrr_options(seed = use_seed)
    )
  } else {
    lapply(x, fn, ...)
  }
}
