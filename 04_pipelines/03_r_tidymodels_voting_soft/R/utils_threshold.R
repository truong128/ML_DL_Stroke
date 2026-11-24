# Threshold tuning and metrics — mirrors Python choose_classification_threshold / bootstrap

choose_classification_threshold <- function(y_true, y_prob, metric = "youden") {
  candidates <- unique(c(
    stats::quantile(y_prob, probs = seq(0.05, 0.95, length.out = 50), na.rm = TRUE),
    0.5
  ))
  best_t <- 0.5
  best_score <- -1
  for (t in candidates) {
    pred <- as.integer(y_prob >= t)
    if (length(unique(pred)) < 2) next
    score <- if (metric == "youden") {
      youden_j_bin(y_true, pred)
    } else if (metric == "balanced_accuracy") {
      bal_accuracy_bin(y_true, pred)
    } else {
      f_meas_pos(y_true, pred)
    }
    if (score > best_score) {
      best_score <- score
      best_t <- t
    }
  }
  list(
    threshold = best_t,
    info = list(
      metric = metric,
      val_score = best_score,
      n_candidates = length(candidates)
    )
  )
}

isotonic_calibrate <- function(y_val, p_val, p_new) {
  ord <- order(p_val)
  p_sorted <- p_val[ord]
  y_sorted <- y_val[ord]
  iso <- isotone::gpava(p_sorted, y_sorted)
  p_new_clamped <- pmin(pmax(p_new, min(p_sorted)), max(p_sorted))
  stats::approx(p_sorted, iso$x, xout = p_new_clamped, rule = 2)$y
}

bootstrap_one <- function(b, y_true, y_prob, y_pred, n, seed = 42L) {
  set.seed(seed + b)
  idx <- sample.int(n, n, replace = TRUE)
  yt <- y_true[idx]
  yp <- y_pred[idx]
  ypr <- y_prob[idx]
  if (length(unique(yt)) < 2) return(NULL)
  cm <- table(factor(yt, levels = c(0, 1)), factor(yp, levels = c(0, 1)))
  tn <- cm["0", "0"]; fp <- cm["0", "1"]; fn <- cm["1", "0"]; tp <- cm["1", "1"]
  if (any(is.na(c(tn, fp, fn, tp)))) return(NULL)
  c(
    roc_auc_pos(yt, ypr),
    yardstick::accuracy_vec(truth = truth_bin(yt), estimate = factor(yp, levels = c(0, 1))),
    tp / (tp + fn),
    tn / (tn + fp),
    f_meas_pos(yt, yp),
    mean((ypr - yt)^2),
    pr_auc_pos(yt, ypr)
  )
}

bootstrap_metrics <- function(
    y_true, y_prob, y_pred, n_boot = 1000, seed = 42L, parallel = FALSE) {
  set.seed(seed)
  metrics <- c("roc_auc", "accuracy", "sensitivity", "specificity", "f1", "brier", "pr_auc")
  boot_mat <- matrix(NA_real_, nrow = n_boot, ncol = length(metrics))
  colnames(boot_mat) <- metrics
  n <- length(y_true)
  boot_fn <- function(b) bootstrap_one(b, y_true, y_prob, y_pred, n, seed = seed)
  if (parallel && exists("parallel_map", mode = "function")) {
    chunks <- parallel_map(seq_len(n_boot), boot_fn, parallel = TRUE, use_seed = FALSE)
    for (i in seq_along(chunks)) {
      row <- chunks[[i]]
      if (!is.null(row)) boot_mat[i, ] <- row
    }
  } else {
    for (b in seq_len(n_boot)) {
      row <- boot_fn(b)
      if (!is.null(row)) boot_mat[b, ] <- row
    }
  }
  rows <- lapply(metrics, function(m) {
    v <- boot_mat[, m]
    v <- v[!is.na(v)]
    data.frame(
      metric = m,
      point = NA_real_,
      ci_low = stats::quantile(v, 0.025, na.rm = TRUE),
      ci_high = stats::quantile(v, 0.975, na.rm = TRUE),
      mean_boot = mean(v, na.rm = TRUE)
    )
  })
  dplyr::bind_rows(rows)
}

confusion_plot <- function(cm, thr, out_path, title_suffix = "") {
  df <- as.data.frame(cm)
  names(df) <- c("Actual", "Predicted", "n")
  p1 <- ggplot2::ggplot(df, ggplot2::aes(Predicted, Actual, fill = n)) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(ggplot2::aes(label = n), color = "white") +
    ggplot2::scale_fill_gradient(low = "#deebf7", high = "#3182bd") +
    ggplot2::labs(title = sprintf("Test confusion matrix (thr=%.3f)%s", thr, title_suffix)) +
    ggplot2::theme_minimal()
  ggplot2::ggsave(out_path, p1, width = 5, height = 4, dpi = 150)
}
