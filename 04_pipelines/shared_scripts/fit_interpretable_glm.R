#!/usr/bin/env Rscript
# Interpretable survey-weighted GLM on LASSO-selected features (design-based SEs).
#
# Usage:
#   Rscript scripts/fit_interpretable_glm.R --dataset ny
#   Rscript scripts/fit_interpretable_glm.R --dataset all_states --features-from results/ml/all_states
#
# Requires: survey, dplyr, readr, broom

suppressPackageStartupMessages({
  if (!requireNamespace("survey", quietly = TRUE)) {
    stop(
      "Package 'survey' not installed. Run:\n",
      "  Rscript tidymodels_pipeline/setup.R\n",
      "or: install.packages('survey', repos='https://cloud.r-project.org')"
    )
  }
  library(survey)
  library(dplyr)
  library(readr)
  library(broom)
})

args <- commandArgs(trailingOnly = TRUE)
parse_flag <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i >= length(args)) return(default)
  args[[i + 1]]
}
dataset <- parse_flag("--dataset", "ny")
split <- parse_flag("--split", "train")
features_from <- parse_flag("--features-from", NULL)
out_dir <- parse_flag("--out-dir", NULL)
weighted <- !("--unweighted" %in% args)

script_dir <- dirname(sub(
  "--file=", "",
  commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]
))
if (length(script_dir) == 0 || script_dir == "") {
  script_dir <- file.path(getwd(), "scripts")
}
root <- Sys.getenv("PROJECT_ROOT", unset = "")
if (root == "") {
  root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
} else {
  root <- normalizePath(root, mustWork = TRUE)
}

prefix <- if (dataset == "ny") "brfss_stroke_new_york" else "brfss_stroke_all_states"
proc <- file.path(root, "datasets", "processed", paste0(prefix, "_", split, ".csv"))
if (!file.exists(proc)) stop("Missing ", proc)

load_features <- function() {
  if (!is.null(features_from)) {
    p <- file.path(root, features_from)
    if (dir.exists(p)) p <- file.path(p, "02_lasso_selected_features.txt")
    if (!file.exists(p)) stop("Feature list not found: ", p)
    lines <- readLines(p)
    lines <- trimws(lines)
    lines[nzchar(lines) & !grepl("^#", lines)]
  } else {
    def <- file.path(root, "results", "ml", dataset, "02_lasso_selected_features.txt")
    if (!file.exists(def)) stop("Run ML pipeline or pass --features-from")
    lines <- readLines(def)
    lines <- trimws(lines)
    lines[nzchar(lines) & !grepl("^#", lines)]
  }
}

feats <- load_features()
dat <- read_csv(proc, show_col_types = FALSE)
for (f in feats) dat[[f]] <- as.numeric(dat[[f]])
for (f in feats) dat[[f]][is.na(dat[[f]])] <- median(dat[[f]], na.rm = TRUE)

dat$stroke <- as.integer(dat$stroke)
if (!all(c("strata", "psu", "survey_weight") %in% names(dat))) {
  stop("Processed CSV must contain strata, psu, survey_weight for svyglm")
}

n_before <- nrow(dat)
if (weighted) {
  ok <- complete.cases(dat[, c("strata", "psu", "survey_weight")])
  n_drop <- sum(!ok)
  if (n_drop > 0) {
    message(
      "Dropping ", n_drop, " rows with missing strata/psu/survey_weight ",
      "(", round(100 * n_drop / n_before, 1), "% of ", split, ")"
    )
    dat <- dat[ok, , drop = FALSE]
  }
  if (nrow(dat) < 10) stop("Too few complete survey-design rows after NA drop")
}

form <- as.formula(paste("stroke ~", paste(sprintf("`%s`", feats), collapse = " + ")))

if (weighted) {
  dsgn <- svydesign(
    ids = ~psu,
    strata = ~strata,
    weights = ~survey_weight,
    nest = TRUE,
    data = dat
  )
  fit <- svyglm(form, design = dsgn, family = quasibinomial())
  weight_note <- "svyglm_design"
} else {
  fit <- glm(form, data = dat, family = binomial())
  weight_note <- "unweighted_glm"
}

tidy_or <- broom::tidy(fit, conf.int = TRUE, exponentiate = TRUE) %>%
  mutate(
    weighting = weight_note,
    q_value_fdr_bh = ifelse(
      grepl("Intercept", term, ignore.case = TRUE),
      NA_real_,
      p.adjust(p.value, method = "BH")
    ),
    significant_fdr_05 = !is.na(q_value_fdr_bh) & q_value_fdr_bh < 0.05,
    fdr_method = "Benjamini-Hochberg",
    fdr_alpha = 0.05
  )

out <- out_dir
if (is.null(out)) out <- file.path(root, "results", "posthoc", paste0("glm_", dataset))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
suffix <- if (weighted) "svy_weighted" else "unweighted"

write_csv(tidy_or, file.path(out, paste0("glm_or_table_", split, "_", suffix, ".r.csv")))
capture.output(summary(fit), file = file.path(out, paste0("glm_summary_", split, "_", suffix, ".r.txt")))

meta <- list(
  dataset = dataset,
  split = split,
  weighted = weighted,
  n_obs = nrow(dat),
  n_obs_before_drop = n_before,
  n_dropped_design_na = if (weighted) n_before - nrow(dat) else 0L,
  n_features = length(feats),
  features = feats,
  weighting = weight_note,
  fdr_correction = "Benjamini-Hochberg across LASSO-selected predictors (intercept excluded)",
  fdr_alpha = 0.05,
  n_significant_fdr_05 = sum(tidy_or$significant_fdr_05, na.rm = TRUE),
  note = "Report q_value_fdr_bh for inference; raw p.value is unadjusted for multiplicity"
)
meta_path <- file.path(out, paste0("glm_meta_", split, "_", suffix, ".r.json"))
if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(meta, meta_path, pretty = TRUE, auto_unbox = TRUE)
} else {
  writeLines(paste(names(meta), unlist(meta), sep = ": "), meta_path)
}

message("Wrote OR table to ", out)
print(head(tidy_or, 12))
