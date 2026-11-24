# yardstick metrics aligned with tidymodels .pred_1 = P(stroke = 1)
# Default yardstick event is the *first* factor level; our positive class is stroke = 1 (second).

YARDSTICK_EVENT_LEVEL <- "second"

truth_bin <- function(y) {
  factor(y, levels = c(0, 1))
}

roc_auc_pos <- function(y, estimate) {
  yardstick::roc_auc_vec(
    truth = truth_bin(y),
    estimate = estimate,
    event_level = YARDSTICK_EVENT_LEVEL
  )
}

pr_auc_pos <- function(y, estimate) {
  yardstick::pr_auc_vec(
    truth = truth_bin(y),
    estimate = estimate,
    event_level = YARDSTICK_EVENT_LEVEL
  )
}

f_meas_pos <- function(y, pred_class) {
  yardstick::f_meas_vec(
    truth = truth_bin(y),
    estimate = factor(pred_class, levels = c(0, 1)),
    event_level = YARDSTICK_EVENT_LEVEL
  )
}

bal_accuracy_bin <- function(y, pred_class) {
  yardstick::bal_accuracy_vec(
    truth = truth_bin(y),
    estimate = factor(pred_class, levels = c(0, 1))
  )
}

youden_j_bin <- function(y, pred_class) {
  cm <- table(
    factor(y, levels = c(0, 1)),
    factor(pred_class, levels = c(0, 1))
  )
  tp <- unname(cm["1", "1"])
  fn <- unname(cm["1", "0"])
  tn <- unname(cm["0", "0"])
  fp <- unname(cm["0", "1"])
  if (any(is.na(c(tp, fn, tn, fp)))) {
    return(-1)
  }
  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  sens + spec - 1
}

roc_curve_pos <- function(y, estimate) {
  # yardstick >= 1.3: roc_curve() is S3 on data.frames only (not bare factor vectors).
  df <- data.frame(
    truth = truth_bin(y),
    .pred_1 = as.numeric(estimate)
  )
  yardstick::roc_curve(
    df,
    truth,
    .pred_1,
    event_level = YARDSTICK_EVENT_LEVEL
  )
}

#' Calibration bins for reliability plot (handles tied predicted probabilities).
calibration_bins_df <- function(y, p, n_bins = 10L) {
  p <- as.numeric(p)
  y <- as.integer(y)
  ok <- is.finite(p) & !is.na(y)
  p <- p[ok]
  y <- y[ok]
  if (length(p) == 0L) {
    return(data.frame(mean_pred = numeric(), frac_pos = numeric()))
  }
  if (length(unique(p)) == 1L) {
    return(data.frame(mean_pred = mean(p), frac_pos = mean(y)))
  }
  probs <- seq(0, 1, length.out = max(3L, as.integer(n_bins) + 1L))
  br <- unique(stats::quantile(p, probs = probs, na.rm = TRUE, type = 8))
  if (length(br) < 2L) {
    br <- range(p, na.rm = TRUE)
  }
  bins <- cut(p, breaks = br, include.lowest = TRUE)
  out <- data.frame(
    mean_pred = as.numeric(tapply(p, bins, mean)),
    frac_pos = as.numeric(tapply(y, bins, mean))
  )
  stats::na.omit(out)
}
