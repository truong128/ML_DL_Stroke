#!/usr/bin/env Rscript
# Run tidymodels benchmark for NY and all_states.
args <- c("--dataset", "both")
if ("--quick" %in% commandArgs(trailingOnly = TRUE)) args <- c(args, "--quick")
script <- file.path(dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])), "run_ml_pipeline.R")
cmdargs <- c(script, args)
status <- system2("Rscript", cmdargs)
if (status != 0) quit(status = status)
