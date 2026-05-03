# ==============================================================================
# Latent Class Growth Analysis (LCGA) on EARS Screen Time Data
# Input:  Wide-format TSV with one row per participant × session
# Output: Class assignments CSV, model fit table CSV, trajectory plots
# ==============================================================================

# --- 1. Install & Load Packages -------------------------------------------------

required_packages <- c("tidyverse", "lcmm", "ggplot2", "gridExtra", "corrr")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}
invisible(lapply(required_packages, install_if_missing))

library(tidyverse)
library(lcmm)
library(ggplot2)
library(gridExtra)
library(corrr)   # tidy correlation matrix for the redundancy check

# --- 2. Configuration ---------------------------------------------------------

DATA_PATH   <- "/Users/jasmine/Downloads/thesis_lab/dataset.tsv"
OUTPUT_DIR  <- file.path(getwd(), "ears_lcga_output")

ID_COL      <- "participant_id"
SESSION_COL <- "session_id"

# Ordered waves → converted to numeric time index (0–5)
TARGET_WAVES <- c("ses-00A", "ses-01A", "ses-02A", "ses-03A", "ses-04A", "ses-05A")

# Five screen-time dimensions
SCREEN_VARS <- c(
  total_screen  = "nt_y_earsapp__mins_mean",
  social_media  = "nt_y_earsplt__mins__abcd__social_mean",
  streaming     = "nt_y_earsplt__mins__abcd__stream_mean",
  gaming        = "nt_y_earsapp__mins__andr__gameaction_mean",
  active_typing = "nt_y_earskey__mins_sum"
)

# LCGA settings
MAX_CLASSES    <- 6     # Test 1–6 latent classes
SET_SEED       <- 2024
CORR_THRESHOLD <- 0.70  # Drop one of any pair of features correlated above this

# --- 3. Load & Reshape Data ---------------------------------------------------
# Expected input: wide TSV where each row = one participant × one session.
# Required columns: participant_id, session_id, and the five SCREEN_VARS values.

cat("Loading data...\n")
raw <- read_tsv(DATA_PATH, show_col_types = FALSE)

# Keep only target waves and relevant columns
keep_cols <- c(ID_COL, SESSION_COL, unname(SCREEN_VARS))
missing_cols <- setdiff(keep_cols, colnames(raw))
if (length(missing_cols) > 0)
  stop("Missing columns in input file: ", paste(missing_cols, collapse = ", "))

long_data <- raw %>%
  select(all_of(keep_cols)) %>%
  filter(.data[[SESSION_COL]] %in% TARGET_WAVES) %>%
  # Map session labels to a numeric time index (0, 1, 2, 3, 4, 5)
  mutate(
    time_index = as.integer(factor(.data[[SESSION_COL]],
                                   levels = TARGET_WAVES)) - 1L,
    subject_int = as.integer(factor(.data[[ID_COL]]))
  )

cat(sprintf("Participants: %d | Sessions retained: %d\n",
            n_distinct(long_data[[ID_COL]]),
            n_distinct(long_data[[SESSION_COL]])))

# ── Diagnostic: report missingness per screen variable ───────────────────────
cat("\nMissingness per screen variable (across all retained rows):\n")
long_data %>%
  summarise(across(all_of(unname(SCREEN_VARS)), \(x) mean(is.na(x)), .names = "{.col}__pct_na")) %>%
  tidyr::pivot_longer(everything(), names_to = "variable", values_to = "pct_na") %>%
  mutate(variable = gsub("__pct_na$", "", variable),
         pct_na   = sprintf("%.1f%%", pct_na * 100)) %>%
  print()
# ─────────────────────────────────────────────────────────────────────────────

# --- 4. Correlation Check on Feature Matrix -----------------------------------
# We build a person-level feature matrix (OLS slope + intercept per variable)
# then drop any variable whose slope is correlated > CORR_THRESHOLD with another.
# This guards against near-collinear features inflating class solutions.

cat("\nStep 1 — Per-participant OLS slopes & intercepts...\n")

ols_features <- long_data %>%
  group_by(.data[[ID_COL]], subject_int) %>%
  summarise(
    across(
      all_of(unname(SCREEN_VARS)),
      list(
        slope     = \(x) if (sum(!is.na(x)) >= 2) coef(lm(x ~ time_index))[2] else NA_real_,
        intercept = \(x) if (sum(!is.na(x)) >= 2) coef(lm(x ~ time_index))[1] else NA_real_
      ),
      .names = "{.col}__{.fn}"
    ),
    .groups = "drop"
  )

# Pull just slope columns for the correlation check
slope_cols <- grep("__slope$", colnames(ols_features), value = TRUE)

cat("\nStep 2 — Correlation matrix of slopes (checking for redundancy)...\n")
slope_corr <- ols_features %>%
  select(all_of(slope_cols)) %>%
  corrr::correlate(quiet = TRUE)

print(slope_corr)

# Greedy drop: if any pair exceeds threshold, drop the second variable
drop_vars <- character(0)
corr_mat  <- as.matrix(slope_corr[, -1])
rownames(corr_mat) <- slope_cols
for (i in seq_along(slope_cols)) {
  if (slope_cols[i] %in% drop_vars) next
  for (j in seq_along(slope_cols)) {
    if (i >= j || slope_cols[j] %in% drop_vars) next
    if (!is.na(corr_mat[i, j]) && abs(corr_mat[i, j]) > CORR_THRESHOLD) {
      cat(sprintf("  Dropping '%s' (|r| = %.2f with '%s')\n",
                  slope_cols[j], corr_mat[i, j], slope_cols[i]))
      drop_vars <- c(drop_vars, slope_cols[j])
    }
  }
}

# Identify which screen variables survive
surviving_base_vars <- slope_cols[!slope_cols %in% drop_vars] %>%
  gsub("__slope$", "", .)

# Surviving SCREEN_VARS (short names → original column names)
surviving_screen_vars <- SCREEN_VARS[unname(SCREEN_VARS) %in% surviving_base_vars]
cat(sprintf("\nVariables retained after correlation check: %s\n",
            paste(names(surviving_screen_vars), collapse = ", ")))

# --- 5. Standardize Outcome Variables -----------------------------------------
# lcmm works better with standardized outcomes; we z-score each variable
# across the whole long dataset.

cat("\nStep 3 — Standardizing outcome variables...\n")

long_std <- long_data %>%
  mutate(across(
    all_of(unname(surviving_screen_vars)),
    ~ as.numeric(scale(.x)),
    .names = "{.col}__z"
  ))

z_vars <- paste0(unname(surviving_screen_vars), "__z")
names(z_vars) <- names(surviving_screen_vars)

cat("Standardized variables:\n")
long_std %>%
  select(all_of(unname(z_vars))) %>%
  summarise(across(everything(), list(
    mean = \(x) mean(x, na.rm = TRUE),
    sd   = \(x) sd(x,   na.rm = TRUE)
  ))) %>%
  print()

# --- 6. Fit Separate LCGA per Screen Dimension --------------------------------
# For each surviving screen variable we fit 1..MAX_CLASSES LCGA models,
# then select the optimal class count by BIC (same logic as the ABCD template).

set.seed(SET_SEED)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Helper: fit one hlme model
fit_one_lcga <- function(data, outcome_z, n_classes, model_k1 = NULL) {
  lcga_data <- as.data.frame(data)
  f_fixed   <- as.formula(sprintf("%s ~ time_index", outcome_z))

  if (n_classes == 1) {
    hlme(fixed = f_fixed, random = ~1, subject = "subject_int",
         ng = 1, data = lcga_data, verbose = FALSE)
  } else {
    tryCatch({
      # gridsearch in lcmm 2.x requires minit — pass the k=1 model every
      # time (not the previous k) to avoid the parameter-length mismatch crash
      gridsearch(
        hlme(fixed   = f_fixed, mixture = f_fixed, random = ~1,
             subject = "subject_int", ng = n_classes,
             nwg     = FALSE, data = lcga_data, verbose = FALSE),
        rep     = 50,
        maxiter = 30,
        minit   = model_k1
      )
    }, error = function(e) {
      cat(sprintf("  [ERROR] ng=%d: %s\n", n_classes, e$message))
      NULL
    })
  }
}

# Helper: extract fit statistics — compatible with lcmm >= 2.2.2
# In lcmm 2.x, $npm was renamed; we derive it from length(m$best) instead.
extract_fit <- function(m, var_name) {
  safe <- function(x) tryCatch({ v <- as.numeric(x); v[!is.na(v)][1] }, error = function(e) NA_real_)
  ng     <- m$ng
  loglik <- safe(m$loglik)
  aic    <- safe(m$AIC)
  bic    <- safe(m$BIC)

  # ns = number of subjects; npm = number of parameters
  # lcmm 2.x stores ns directly; npm is length of the $best vector
  ns  <- tryCatch(as.integer(m$ns)[1],          error = function(e) NA_integer_)
  npm <- tryCatch(as.integer(length(m$best))[1], error = function(e) NA_integer_)

  # Fallback: try $npm in case an older lcmm version is used
  if (is.na(npm)) npm <- tryCatch(as.integer(m$npm)[1], error = function(e) NA_integer_)

  sabic <- tryCatch(round(-2 * loglik + log((ns + 2) / 24) * npm, 2),
                    error = function(e) NA_real_, warning = function(w) NA_real_)

  entropy <- NA_real_; min_cs <- NA_integer_
  if (!is.null(m$pprob) && ng > 1) {
    pc <- grep("^prob", colnames(m$pprob), value = TRUE)
    if (length(pc) >= 2) {
      probs   <- as.matrix(m$pprob[, pc, drop = FALSE])
      cls     <- apply(probs, 1, which.max)
      min_cs  <- as.integer(min(table(cls)))
      H       <- -sum(probs * log(probs + 1e-10))
      entropy <- round(1 - H / (nrow(probs) * log(ncol(probs))), 3)
    }
  }
  tibble(Variable = var_name, Classes = ng, LogLik = round(loglik, 2),
         AIC = round(aic, 2), BIC = round(bic, 2), SABIC = sabic,
         Entropy = entropy, Min_ClassSize = min_cs)
}

# Main loop over surviving screen dimensions
all_fit_tables   <- list()
all_assignments  <- list()
all_best_models  <- list()
all_traj_plots   <- list()

id_map <- long_std %>% distinct(subject_int, .data[[ID_COL]])

for (var_short in names(z_vars)) {
  outcome_z <- z_vars[[var_short]]
  cat(sprintf("\n========== %s ==========\n", var_short))

  # Drop rows where this outcome is NA
  df_var <- long_std %>%
    filter(!is.na(.data[[outcome_z]])) %>%
    select(all_of(c(ID_COL, "subject_int", "time_index", outcome_z)))

  # ── Guard: skip variable if too few non-NA observations ──────────────────
  n_obs   <- nrow(df_var)
  n_subj_var <- n_distinct(df_var$subject_int)
  if (n_obs == 0 || n_subj_var < 10) {
    cat(sprintf("  SKIPPING: only %d observations / %d subjects with non-NA values.\n",
                n_obs, n_subj_var))
    cat("  → Check that the column name is correct and the variable has data.\n")
    next
  }
  cat(sprintf("  Non-NA observations: %d across %d subjects\n", n_obs, n_subj_var))
  # Per-variable 5% threshold (the root cause of the k=1 bug — was using full
  # sample of 11,863 instead of the variable-specific sample)
  min_size_thr <- ceiling(0.05 * n_subj_var)
  cat(sprintf("  Min class size threshold (5%%): %d subjects\n", min_size_thr))
  # ─────────────────────────────────────────────────────────────────────────

  # Fit k=1 first — used as minit seed for all gridsearch calls
  cat("  Fitting 1 class(es)...\n")
  model_k1 <- fit_one_lcga(df_var, outcome_z, 1)
  models_k  <- list()
  if (!is.null(model_k1)) models_k[["1"]] <- model_k1

  for (k in 2:MAX_CLASSES) {
    cat(sprintf("  Fitting %d class(es)...\n", k))
    result <- fit_one_lcga(df_var, outcome_z, k, model_k1 = model_k1)
    if (!is.null(result)) models_k[[as.character(k)]] <- result
  }
  # ── Guard: skip if ALL models failed ─────────────────────────────────────
  if (length(models_k) == 0) {
    cat(sprintf("  SKIPPING %s: all models failed to converge.\n", var_short))
    next
  }
  # ─────────────────────────────────────────────────────────────────────────

  fit_tbl <- map_dfr(models_k, extract_fit, var_name = var_short)
  all_fit_tables[[var_short]] <- fit_tbl

  cat(sprintf("\n  Fit table for %s:\n", var_short))
  print(fit_tbl)

  # Select optimal k
  # 1. Ensure Min_ClassSize column exists
  if (!"Min_ClassSize" %in% colnames(fit_tbl)) fit_tbl$Min_ClassSize <- NA_integer_

  # 2. Filter to models where every class has >= 5% of THIS variable's subjects
  valid <- fit_tbl %>% filter(is.na(Min_ClassSize) | Min_ClassSize >= min_size_thr)
  if (nrow(valid) == 0) {
    cat("  WARNING: no model meets the 5% class size threshold — using all models.\n")
    valid <- fit_tbl
  }

  # 3. Pick lowest BIC among valid models
  optimal_k <- valid$Classes[which.min(valid$BIC)]

  # 4. Sanity-check: report BIC improvement over k=1
  bic_k1 <- fit_tbl$BIC[fit_tbl$Classes == 1]
  bic_opt <- fit_tbl$BIC[fit_tbl$Classes == optimal_k]
  delta   <- round(bic_k1 - bic_opt, 1)
  cat(sprintf("  → Optimal k = %d  |  BIC improvement over k=1: %.1f points\n",
              optimal_k, delta))
  if (delta < 10)
    cat("  NOTE: BIC improvement < 10 — 1-class solution may be adequate.\n")

  best_model <- models_k[[as.character(optimal_k)]]
  all_best_models[[var_short]] <- best_model

  # --- Class assignments for this variable ---
  pprob <- best_model$pprob
  if (colnames(pprob)[1] != "subject_int") colnames(pprob)[1] <- "subject_int"

  if ("class" %in% colnames(pprob)) {
    assigned <- pprob$class
  } else {
    pc <- grep("^prob", colnames(pprob), value = TRUE)
    assigned <- apply(as.matrix(pprob[, pc, drop = FALSE]), 1, which.max)
  }

  assign_df <- pprob %>%
    mutate(!!paste0("class_", var_short) := assigned) %>%
    left_join(id_map, by = "subject_int") %>%
    select(all_of(ID_COL), starts_with("prob"), !!paste0("class_", var_short))

  all_assignments[[var_short]] <- assign_df

  cat(sprintf("  Class sizes:\n"))
  print(table(assigned))

  # --- Trajectory plot for this variable ---
  df_class <- df_var %>%
    left_join(
      tibble(subject_int = pprob$subject_int,
             latent_class = factor(assigned)),
      by = "subject_int"
    )

  traj_means <- df_class %>%
    group_by(latent_class, time_index) %>%
    summarise(
      mean_z = mean(.data[[outcome_z]], na.rm = TRUE),
      se_z   = sd(.data[[outcome_z]], na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  class_sizes <- df_class %>%
    distinct(subject_int, latent_class) %>%
    count(latent_class, name = "n_subj")

  traj_means <- traj_means %>%
    left_join(class_sizes, by = "latent_class") %>%
    mutate(
      class_label = sprintf("Class %s (n=%d)", latent_class, n_subj),
      time_label  = TARGET_WAVES[time_index + 1]
    )

  p <- ggplot(traj_means,
              aes(x = time_index, y = mean_z,
                  color = class_label, group = class_label)) +
    geom_ribbon(aes(ymin = mean_z - se_z, ymax = mean_z + se_z,
                    fill = class_label), alpha = 0.15, color = NA) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    scale_x_continuous(breaks = 0:5, labels = TARGET_WAVES) +
    labs(
      title    = sprintf("LCGA: %s (%d-class solution)", var_short, optimal_k),
      subtitle = "Mean ± SE of z-scored screen time by latent class",
      x        = "Session",
      y        = "Z-scored screen time",
      color    = "Latent Class",
      fill     = "Latent Class"
    ) +
    theme_bw(base_size = 11) +
    theme(
      legend.position  = "right",
      axis.text.x      = element_text(angle = 30, hjust = 1)
    )

  all_traj_plots[[var_short]] <- p

  ggsave(
    file.path(OUTPUT_DIR, sprintf("lcga_trajectories_%s.png", var_short)),
    p, width = 9, height = 5, dpi = 150
  )
}

# --- 7. Combined BIC Plot (all variables in one panel) ------------------------

all_fit_df <- bind_rows(all_fit_tables)

p_bic_all <- ggplot(all_fit_df, aes(x = Classes, y = BIC,
                                     color = Variable, group = Variable)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = seq_len(MAX_CLASSES)) +
  labs(
    title    = "Model Selection: BIC by Number of Classes (all screen dimensions)",
    subtitle = "Lower BIC = better fit",
    x        = "Number of Latent Classes",
    y        = "BIC"
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "right")

ggsave(file.path(OUTPUT_DIR, "lcga_bic_all_variables.png"),
       p_bic_all, width = 9, height = 5, dpi = 150)

# --- 8. Combined Trajectory Grid ----------------------------------------------

if (length(all_traj_plots) > 0) {
  n_plots <- length(all_traj_plots)
  ncols   <- min(2L, n_plots)
  nrows   <- ceiling(n_plots / ncols)

  combined_grid <- gridExtra::arrangeGrob(
    grobs  = all_traj_plots,
    ncol   = ncols,
    nrow   = nrows,
    top    = "EARS Screen Time — LCGA Trajectories (z-scored)"
  )
  ggsave(file.path(OUTPUT_DIR, "lcga_trajectories_combined.png"),
         combined_grid, width = ncols * 9, height = nrows * 5, dpi = 150)
  cat("\nCombined trajectory grid saved.\n")
}

# --- 9. Save Outputs ----------------------------------------------------------

# 9a. Per-variable class assignments → merge into one wide file by participant
cat("\nMerging class assignments across variables...\n")
assignment_wide <- id_map %>% select(all_of(ID_COL))

for (var_short in names(all_assignments)) {
  cols_to_keep <- c(ID_COL, paste0("class_", var_short))
  existing     <- intersect(cols_to_keep, colnames(all_assignments[[var_short]]))
  assignment_wide <- assignment_wide %>%
    left_join(all_assignments[[var_short]][, existing, drop = FALSE],
              by = ID_COL)
}

write_csv(assignment_wide,
          file.path(OUTPUT_DIR, "ears_lcga_class_assignments.csv"))

# 9b. Combined model fit table
write_csv(all_fit_df,
          file.path(OUTPUT_DIR, "ears_lcga_model_fit_table.csv"))

cat(sprintf("\nOutputs written to: %s\n", OUTPUT_DIR))
cat("  ears_lcga_class_assignments.csv\n")
cat("  ears_lcga_model_fit_table.csv\n")
cat("  lcga_bic_all_variables.png\n")
cat("  lcga_trajectories_combined.png\n")
cat("  lcga_trajectories_<variable>.png  (one per screen dimension)\n")

# --- 10. Summary --------------------------------------------------------------

cat("\n===== OPTIMAL CLASS SOLUTIONS PER VARIABLE =====\n")
all_fit_df %>%
  group_by(Variable) %>%
  slice_min(BIC, n = 1) %>%
  select(Variable, Classes, BIC, AIC, Entropy, Min_ClassSize) %>%
  print()

cat("\n===== SCRIPT COMPLETE =====\n")
