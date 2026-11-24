#!/usr/bin/env Rscript
# Install CRAN dependencies for the tidymodels benchmark pipeline.
pkgs <- c(
  "tidymodels", "readr", "dplyr", "tidyr", "tibble", "ggplot2", "jsonlite",
  "glmnet", "ranger", "kernlab", "xgboost", "adabag", "discrim", "naivebayes",
  "nnet", "rpart", "themis", "isotone", "future", "furrr", "scales", "purrr",
  "hardhat", "probably", "sparsediscrim", "baguette", "kknn", "MASS", "LiblineaR",
  "survey", "broom"  # post-hoc: scripts/fit_interpretable_glm.R
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing ", pkg, " ...")
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
  }
}

for (p in pkgs) install_if_missing(p)

message("Setup complete. Run: Rscript scripts/run_ml_pipeline.R --dataset ny")
