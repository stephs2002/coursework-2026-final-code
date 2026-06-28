# 09_multilayer_qap_test_v2.R
#
# Purpose:
#   Multilayer QAP regression for the current three-layer design:
#     collaboration_edge ~ genre similarity + shared label tie
#
# Why this version exists:
#   In some versions/usages of sna::netlm(), passing several predictor matrices
#   as a 3D array may trigger:
#     "Homogeneous graph orders required in netlm."
#
#   This version avoids that problem by passing multiple predictors as a LIST of
#   same-order matrices, which is the safest input format for netlm().
#
# Expected input:
#   outputs/tables/multilayer_pairwise_edge_table.csv
#
# Run:
#   Rscript 09_multilayer_qap_test_v2.R
#
# If graph mode still fails on your setup, run:
#   Rscript 09_multilayer_qap_test_v2.R outputs/tables/multilayer_pairwise_edge_table.csv 5000 digraph
#
# Args:
#   1. input path, default outputs/tables/multilayer_pairwise_edge_table.csv
#   2. number of permutations, default 5000
#   3. mode for sna::netlm, default digraph
#
# Note:
#   Default mode is set to "digraph" here because it is usually more robust in
#   sna::netlm(). Since all matrices are symmetric, substantive interpretation
#   remains the same, although dyads are represented twice internally.
#   If your setup works with mode="graph", you may use it.

# ---------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

INPUT_PATH <- ifelse(
  length(args) >= 1,
  args[1],
  "outputs/tables/multilayer_pairwise_edge_table.csv"
)

N_REPS <- ifelse(
  length(args) >= 2,
  as.integer(args[2]),
  5000
)

QAP_MODE <- ifelse(
  length(args) >= 3,
  args[3],
  "digraph"
)

OUTPUT_TABLES_DIR <- "outputs/tables"
OUTPUT_TEXT_PATH <- "outputs/qap_multilayer_results.txt"

dir.create(OUTPUT_TABLES_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(OUTPUT_TEXT_PATH), recursive = TRUE, showWarnings = FALSE)

set.seed(42)

# ---------------------------------------------------------
# 2. Dependencies
# ---------------------------------------------------------

if (!requireNamespace("sna", quietly = TRUE)) {
  stop(
    "Package 'sna' is not installed. Install it with install.packages('sna') ",
    "and run the script again."
  )
}

library(sna)

# ---------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------

check_required_columns <- function(df, required_cols) {
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop("Input file is missing required columns: ", paste(missing_cols, collapse = ", "))
  }
}

make_symmetric_matrix <- function(df, artists, value_col) {
  mat <- matrix(
    0,
    nrow = length(artists),
    ncol = length(artists),
    dimnames = list(artists, artists)
  )

  for (i in seq_len(nrow(df))) {
    a <- as.character(df$source[i])
    b <- as.character(df$target[i])

    if (!a %in% artists || !b %in% artists) {
      next
    }

    value <- df[[value_col]][i]
    if (is.na(value)) {
      value <- 0
    }

    value <- as.numeric(value)

    mat[a, b] <- value
    mat[b, a] <- value
  }

  diag(mat) <- 0
  return(mat)
}

get_unique_dyad_values <- function(mat) {
  mat[lower.tri(mat, diag = FALSE)]
}

matrix_has_variance <- function(mat) {
  vals <- get_unique_dyad_values(mat)
  stats::var(vals, na.rm = TRUE) > 0
}

matrix_summary <- function(mat, matrix_name) {
  vals <- get_unique_dyad_values(mat)

  data.frame(
    matrix = matrix_name,
    n_nodes = nrow(mat),
    n_unique_dyads = length(vals),
    min = min(vals, na.rm = TRUE),
    mean = mean(vals, na.rm = TRUE),
    median = stats::median(vals, na.rm = TRUE),
    max = max(vals, na.rm = TRUE),
    sd = stats::sd(vals, na.rm = TRUE),
    nonzero_dyads = sum(vals != 0, na.rm = TRUE),
    nonzero_share = round(mean(vals != 0, na.rm = TRUE), 4)
  )
}

run_qap_model <- function(model_name, y, predictor_mats, reps, qap_mode) {
  # Remove predictors with zero variance.
  usable_predictors <- predictor_mats[sapply(predictor_mats, matrix_has_variance)]

  if (length(usable_predictors) == 0) {
    return(list(
      model_name = model_name,
      skipped = TRUE,
      reason = "All predictors have zero variance.",
      fit = NULL,
      predictors_used = character(0)
    ))
  }

  # IMPORTANT:
  # For one predictor, pass a matrix.
  # For multiple predictors, pass a LIST of matrices, not a 3D array.
  # This avoids the "Homogeneous graph orders required" error.
  if (length(usable_predictors) == 1) {
    x_input <- usable_predictors[[1]]
  } else {
    x_input <- usable_predictors
  }

  fit <- sna::netlm(
    y = y,
    x = x_input,
    nullhyp = "qap",
    reps = reps,
    mode = qap_mode,
    diag = FALSE
  )

  return(list(
    model_name = model_name,
    skipped = FALSE,
    reason = "",
    fit = fit,
    predictors_used = names(usable_predictors)
  ))
}

capture_model_output <- function(model_result) {
  if (isTRUE(model_result$skipped)) {
    return(c(
      paste0("MODEL: ", model_result$model_name),
      paste0("SKIPPED: ", model_result$reason),
      ""
    ))
  }

  c(
    paste0("MODEL: ", model_result$model_name),
    paste0("Predictors used: ", paste(model_result$predictors_used, collapse = ", ")),
    capture.output(print(model_result$fit)),
    ""
  )
}

get_descriptive_lm_coefficients <- function(dyads, formula_obj, model_name) {
  # NOT QAP inference. This is only descriptive OLS output.
  lm_fit <- stats::lm(formula_obj, data = dyads)

  coef_table <- as.data.frame(summary(lm_fit)$coefficients)
  coef_table$term <- rownames(coef_table)
  rownames(coef_table) <- NULL

  names(coef_table) <- c(
    "estimate",
    "std_error",
    "t_value",
    "lm_p_value_not_qap",
    "term"
  )

  coef_table$model <- model_name
  coef_table$r_squared_lm_not_qap <- summary(lm_fit)$r.squared
  coef_table$adj_r_squared_lm_not_qap <- summary(lm_fit)$adj.r.squared

  coef_table <- coef_table[
    c(
      "model",
      "term",
      "estimate",
      "std_error",
      "t_value",
      "lm_p_value_not_qap",
      "r_squared_lm_not_qap",
      "adj_r_squared_lm_not_qap"
    )
  ]

  return(coef_table)
}

# ---------------------------------------------------------
# 4. Load data
# ---------------------------------------------------------

if (!file.exists(INPUT_PATH)) {
  stop("Input file not found: ", INPUT_PATH)
}

dyads <- read.csv(INPUT_PATH, stringsAsFactors = FALSE)

required_cols <- c(
  "source",
  "target",
  "collaboration_edge",
  "jaccard_similarity",
  "label_edge",
  "shared_label_count"
)

check_required_columns(dyads, required_cols)

numeric_cols <- c(
  "collaboration_edge",
  "jaccard_similarity",
  "label_edge",
  "shared_label_count"
)

for (col in numeric_cols) {
  dyads[[col]] <- as.numeric(dyads[[col]])
  dyads[[col]][is.na(dyads[[col]])] <- 0
}

if (!"genre_edge" %in% names(dyads)) {
  dyads$genre_edge <- ifelse(dyads$jaccard_similarity >= 0.5, 1, 0)
}

if (!"collaboration_weight" %in% names(dyads)) {
  dyads$collaboration_weight <- dyads$collaboration_edge
}

dyads$collaboration_weight <- as.numeric(dyads$collaboration_weight)
dyads$collaboration_weight[is.na(dyads$collaboration_weight)] <- 0

artists <- sort(unique(c(dyads$source, dyads$target)))

cat("Loaded dyadic table:", INPUT_PATH, "\n")
cat("Number of artists:", length(artists), "\n")
cat("Number of dyads:", nrow(dyads), "\n")
cat("QAP permutations:", N_REPS, "\n")
cat("QAP mode:", QAP_MODE, "\n")

# ---------------------------------------------------------
# 5. Construct matrices
# ---------------------------------------------------------

collaboration_mat <- make_symmetric_matrix(dyads, artists, "collaboration_edge")
genre_similarity_mat <- make_symmetric_matrix(dyads, artists, "jaccard_similarity")
genre_edge_mat <- make_symmetric_matrix(dyads, artists, "genre_edge")
label_edge_mat <- make_symmetric_matrix(dyads, artists, "label_edge")
shared_label_count_mat <- make_symmetric_matrix(dyads, artists, "shared_label_count")
collaboration_weight_mat <- make_symmetric_matrix(dyads, artists, "collaboration_weight")

matrix_summaries <- do.call(
  rbind,
  list(
    matrix_summary(collaboration_mat, "collaboration_edge"),
    matrix_summary(genre_similarity_mat, "jaccard_similarity"),
    matrix_summary(genre_edge_mat, "genre_edge"),
    matrix_summary(label_edge_mat, "label_edge"),
    matrix_summary(shared_label_count_mat, "shared_label_count"),
    matrix_summary(collaboration_weight_mat, "collaboration_weight")
  )
)

write.csv(
  matrix_summaries,
  file.path(OUTPUT_TABLES_DIR, "qap_multilayer_matrix_summary.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 6. Descriptive dyadic summaries
# ---------------------------------------------------------

descriptive_summary <- data.frame(
  metric = c(
    "n_artists",
    "n_unique_dyads",
    "n_collaboration_edges",
    "n_genre_edges",
    "n_label_edges",
    "mean_jaccard_collaboration_pairs",
    "mean_jaccard_non_collaboration_pairs",
    "label_edge_share_collaboration_pairs",
    "label_edge_share_non_collaboration_pairs",
    "mean_shared_label_count_collaboration_pairs",
    "mean_shared_label_count_non_collaboration_pairs"
  ),
  value = c(
    length(artists),
    nrow(dyads),
    sum(dyads$collaboration_edge == 1),
    sum(dyads$genre_edge == 1),
    sum(dyads$label_edge == 1),
    mean(dyads$jaccard_similarity[dyads$collaboration_edge == 1], na.rm = TRUE),
    mean(dyads$jaccard_similarity[dyads$collaboration_edge == 0], na.rm = TRUE),
    mean(dyads$label_edge[dyads$collaboration_edge == 1], na.rm = TRUE),
    mean(dyads$label_edge[dyads$collaboration_edge == 0], na.rm = TRUE),
    mean(dyads$shared_label_count[dyads$collaboration_edge == 1], na.rm = TRUE),
    mean(dyads$shared_label_count[dyads$collaboration_edge == 0], na.rm = TRUE)
  )
)

descriptive_summary$value <- round(descriptive_summary$value, 6)

write.csv(
  descriptive_summary,
  file.path(OUTPUT_TABLES_DIR, "qap_multilayer_descriptive_summary.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 7. QAP models
# ---------------------------------------------------------

models <- list(
  run_qap_model(
    model_name = "M1_collaboration_on_genre_similarity",
    y = collaboration_mat,
    predictor_mats = list(
      genre_similarity = genre_similarity_mat
    ),
    reps = N_REPS,
    qap_mode = QAP_MODE
  ),

  run_qap_model(
    model_name = "M2_collaboration_on_label_edge",
    y = collaboration_mat,
    predictor_mats = list(
      label_edge = label_edge_mat
    ),
    reps = N_REPS,
    qap_mode = QAP_MODE
  ),

  run_qap_model(
    model_name = "M3_collaboration_on_genre_similarity_and_label_edge",
    y = collaboration_mat,
    predictor_mats = list(
      genre_similarity = genre_similarity_mat,
      label_edge = label_edge_mat
    ),
    reps = N_REPS,
    qap_mode = QAP_MODE
  ),

  run_qap_model(
    model_name = "M4_collaboration_on_genre_similarity_and_shared_label_count",
    y = collaboration_mat,
    predictor_mats = list(
      genre_similarity = genre_similarity_mat,
      shared_label_count = shared_label_count_mat
    ),
    reps = N_REPS,
    qap_mode = QAP_MODE
  ),

  run_qap_model(
    model_name = "M5_collaboration_on_binary_genre_edge_and_label_edge",
    y = collaboration_mat,
    predictor_mats = list(
      genre_edge = genre_edge_mat,
      label_edge = label_edge_mat
    ),
    reps = N_REPS,
    qap_mode = QAP_MODE
  )
)

# ---------------------------------------------------------
# 8. Descriptive LM coefficients table
# ---------------------------------------------------------

lm_coefficients <- do.call(
  rbind,
  list(
    get_descriptive_lm_coefficients(
      dyads,
      collaboration_edge ~ jaccard_similarity,
      "M1_collaboration_on_genre_similarity"
    ),
    get_descriptive_lm_coefficients(
      dyads,
      collaboration_edge ~ label_edge,
      "M2_collaboration_on_label_edge"
    ),
    get_descriptive_lm_coefficients(
      dyads,
      collaboration_edge ~ jaccard_similarity + label_edge,
      "M3_collaboration_on_genre_similarity_and_label_edge"
    ),
    get_descriptive_lm_coefficients(
      dyads,
      collaboration_edge ~ jaccard_similarity + shared_label_count,
      "M4_collaboration_on_genre_similarity_and_shared_label_count"
    ),
    get_descriptive_lm_coefficients(
      dyads,
      collaboration_edge ~ genre_edge + label_edge,
      "M5_collaboration_on_binary_genre_edge_and_label_edge"
    )
  )
)

write.csv(
  lm_coefficients,
  file.path(OUTPUT_TABLES_DIR, "qap_multilayer_lm_coefficients_descriptive.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 9. Save QAP text output
# ---------------------------------------------------------

output_lines <- c(
  "MULTILAYER QAP REGRESSION RESULTS",
  "=================================",
  "",
  paste0("Input file: ", INPUT_PATH),
  paste0("Number of artists: ", length(artists)),
  paste0("Number of unique dyads: ", nrow(dyads)),
  paste0("QAP permutations: ", N_REPS),
  paste0("QAP mode: ", QAP_MODE),
  "",
  "IMPORTANT NOTE:",
  "The printed p-values in the model outputs below are QAP permutation p-values.",
  "The CSV file qap_multilayer_lm_coefficients_descriptive.csv is only descriptive",
  "and should not be used for inferential p-values.",
  "",
  "DESCRIPTIVE SUMMARY",
  "-------------------",
  capture.output(print(descriptive_summary)),
  "",
  "MATRIX SUMMARY",
  "--------------",
  capture.output(print(matrix_summaries)),
  "",
  "QAP MODELS",
  "----------"
)

for (model_result in models) {
  output_lines <- c(
    output_lines,
    capture_model_output(model_result),
    "----------------------------------------",
    ""
  )
}

writeLines(output_lines, OUTPUT_TEXT_PATH)

# ---------------------------------------------------------
# 10. Print final console output
# ---------------------------------------------------------

cat("\nMultilayer QAP completed.\n")
cat("Results text file:", OUTPUT_TEXT_PATH, "\n")
cat("Descriptive summary:", file.path(OUTPUT_TABLES_DIR, "qap_multilayer_descriptive_summary.csv"), "\n")
cat("Matrix summary:", file.path(OUTPUT_TABLES_DIR, "qap_multilayer_matrix_summary.csv"), "\n")
cat("Descriptive LM coefficients:", file.path(OUTPUT_TABLES_DIR, "qap_multilayer_lm_coefficients_descriptive.csv"), "\n\n")

cat("Descriptive summary:\n")
print(descriptive_summary)

cat("\nMatrix summary:\n")
print(matrix_summaries)

cat("\nDone.\n")
