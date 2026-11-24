#!/usr/bin/env Rscript
script_dir <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
pipeline_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
for (f in sort(list.files(file.path(pipeline_dir, "R"), "\\.R$", full.names = TRUE))) {
  source(f, local = FALSE)
}
expected <- 16L
required <- c(
  "logistic_regression", "multinom_reg", "svm_rbf", "svm_linear", "decision_tree",
  "random_forest", "gradient_boosting", "extra_trees", "adaboost", "bag_tree",
  "bagging", "knn", "lda", "qda", "naive_bayes", "mlp"
)
names <- names(get_model_specs())
if (length(names) != expected) {
  stop("Expected ", expected, " classifiers, got ", length(names), ": ", paste(names, collapse = ", "))
}
if (!all(required %in% names)) {
  stop("Missing: ", paste(setdiff(required, names), collapse = ", "))
}
cat("OK:", expected, "classifiers\n")
