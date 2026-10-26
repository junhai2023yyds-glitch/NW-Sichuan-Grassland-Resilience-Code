# =============================================================================
# 04_RI_SerialDependence_LinearSensitivity.R
# Reviewer 2, Comment 9 (3.5) -- combined Phase 02A+02B, portable
# Regional trend (OLS and HAC lags 1..6); all-pixel linear slope, HAC(4)/HAC(6),
# and pixel-wise FDR (BH and BY).  NO change-point reclassification in this file.
# Portable public repository edition; derived from tested scripts 02A and 02B
#
# Scientific precautions:
#   - OLS slope describes linear rate; HAC changes inference, not the slope.
#   - Explicit calendar-year lags: missing observations DO NOT become neighbors.
#   - Bartlett kernel with finite-sample n/(n-2) adjustment; t(df=n-2).
#   - A 5-year centered window entails overlap but not rho=0.8.
#   - Multiple-testing family: ALL eligible pixels tested for ONE linear slope.
#   - "Linear not significant" is NOT equivalent to "no change point".
#   - Rasters and CSVs go into a NEW run directory, never overwrite old results.
#   - Formal HAC inference requires >=25 valid years (original >=10 still kept
#     as the common descriptive extent). Insufficient inference is coded as 3.
# =============================================================================

# =========================== 1. USER CONFIG ==================================
# Run from the repository root: Rscript R_analysis/04_RI_SerialDependence_LinearSensitivity.R
# No private path appears in this source. If large inputs live elsewhere,
# set RI_ANNUAL_DIR and RI_GRASS_MASK in your local environment (NOT in Git).
project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
var_folder <- Sys.getenv("RI_ANNUAL_DIR",
  unset = file.path(project_root, "data", "RI_annual"))
grass_shp_path <- Sys.getenv("RI_GRASS_MASK",
  unset = file.path(project_root, "data", "mask", "grassland.shp"))
output_parent <- Sys.getenv("RI_SENSITIVITY_OUT",
  unset = file.path(project_root, "output", "reviewer2_comment9", "04_sensitivity"))

start_year <- 1985L
end_year <- 2020L
min_valid_years <- 10L       # Original descriptive/common-valid mask
min_hac_years <- 25L         # New inference QC; flagged, NOT called "no trend"
cv_mean_eps <- 1e-12
alpha <- 0.05
primary_hac_lag <- 4L
sensitivity_hac_lag <- 6L
max_diagnostic_lag <- 6L
min_diagnostic_pairs <- 8L
use_cores <- 1L             # Custom terra::app functions: safest on Windows

if (!requireNamespace("terra", quietly = TRUE)) {
  stop("Install terra first: install.packages('terra')")
}
suppressPackageStartupMessages(library(terra))
options(scipen = 999)

run_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(output_parent, paste0("Run_", run_tag))
if (dir.exists(out_dir)) out_dir <- paste0(out_dir, "_", Sys.getpid())
if (!dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)) {
  stop("Could not create output directory: ", out_dir)
}
cat("Phase 2A output directory:\n", out_dir, "\n", sep = "")

# ======================== 2. READ ANNUAL RI ==================================
annual_pattern <- "^RI([12][0-9]{3})_n_VG\\.(tif|tiff)$"
files <- list.files(var_folder, pattern = annual_pattern,
                    full.names = TRUE, ignore.case = TRUE)
if (!length(files)) stop("No annual RI rasters found in: ", var_folder)
years <- as.integer(sub(annual_pattern, "\\1", basename(files),
                        ignore.case = TRUE))
keep <- is.finite(years) & years >= start_year & years <= end_year
files <- files[keep]
years <- years[keep]
ord <- order(years)
files <- files[ord]
years <- years[ord]
if (anyDuplicated(years)) stop("Duplicate yearly RI raster files.")
if (!identical(years, seq.int(start_year, end_year))) {
  stop("Exactly 36 consecutive calendar-year files (1985..2020) are required.")
}
if (!file.exists(grass_shp_path)) stop("Missing grassland vector: ", grass_shp_path)

r_grass <- terra::rast(files)
names(r_grass) <- paste0("RI_", years)
grass <- terra::vect(grass_shp_path)
if (!terra::same.crs(r_grass, grass)) {
  grass <- terra::project(grass, terra::crs(r_grass))
}
r_grass <- terra::mask(terra::crop(r_grass, grass), grass)

# Replicate original mask from 02_RI_spatial_analysis(2).R.
r_n <- terra::app(r_grass, fun = function(v) sum(is.finite(v)), cores = use_cores)
r_mean <- terra::app(r_grass, fun = function(v) {
  v <- v[is.finite(v)]
  if (!length(v)) return(NA_real_)
  mean(v)
}, cores = use_cores)
r_sd <- terra::app(r_grass, fun = function(v) {
  v <- v[is.finite(v)]
  if (length(v) < 2L) return(NA_real_)
  stats::sd(v)
}, cores = use_cores)
valid_condition <- (r_n >= min_valid_years & !is.na(r_mean) &
                      !is.na(r_sd) & abs(r_mean) > cv_mean_eps)
r_common_mask <- terra::ifel(valid_condition, 1, NA)
names(r_common_mask) <- "common_analysis_mask"
r_stack <- terra::mask(r_grass, r_common_mask)
terra::writeRaster(r_common_mask,
  file.path(out_dir, "01_Common_Valid_Analysis_Mask.tif"),
  overwrite = FALSE, datatype = "INT1U", NAflag = 255,
  gdal = c("COMPRESS=LZW"))

r_cell_area <- terra::cellSize(r_grass[[1L]], unit = "km")
area_vals <- as.numeric(terra::values(r_cell_area, mat = FALSE))
mask_vals <- as.numeric(terra::values(r_common_mask, mat = FALSE))
in_common <- is.finite(mask_vals)
common_area <- sum(area_vals[in_common], na.rm = TRUE)
common_count <- sum(in_common)
if (!is.finite(common_area) || common_count <= 0L) {
  stop("No eligible pixels: check input raster/mask.")
}
cat("Common extent: ", common_count, " pixels, ",
    round(common_area, 4), " km2\n", sep = "")

# ===================== 3. VECTORIZED HAC FUNCTIONS ==========================
# For OLS with intercept and calendar-time slope b, slope score at year i is
# u_i = e_i * (t_i - mean(t_observed)) / Sxx, where e_i is the OLS residual.
# Bartlett HAC slope variance = n/(n-2) * [sum u_i^2 +
#     2 * sum_{k=1}^L (1-k/(L+1)) * sum_i u_i*u_{i+k}].
# Missing year-pairs contribute zero to the cross-products rather than being
# falsely compressed. The correction is equivalent to sandwich::NeweyWest(...,
# lag=L, prewhite=FALSE, adjust=TRUE) for COMPLETE annual series.

t <- as.numeric(years) - mean(as.numeric(years))
Tlen <- length(t)
if (Tlen != 36L || sensitivity_hac_lag >= Tlen) {
  stop("Unexpected year count / HAC lag range")
}

pixel_hac_batch <- function(X) {
  if (is.null(dim(X))) X <- matrix(as.numeric(X), nrow = 1L)
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (ncol(X) != Tlen) stop("terra::app input does not have 36 year columns")
  nrow_X <- nrow(X)
  obs <- is.finite(X)
  X[!obs] <- NA_real_
  n <- rowSums(obs)
  n_safe <- pmax(n, 1L)
  df <- n - 2L

  # Fit calendar-time OLS slopes, even when years are missing.
  st <- as.numeric(obs %*% t)
  st2 <- as.numeric(obs %*% (t * t))
  sy <- rowSums(X, na.rm = TRUE)
  sty <- rowSums(sweep(X, 2L, t, "*"), na.rm = TRUE)
  sxx <- st2 - st * st / n_safe
  sxy <- sty - st * sy / n_safe
  slope <- sxy / sxx
  intercept <- (sy - slope * st) / n_safe
  usable_slope <- (n >= min_valid_years) & is.finite(slope) &
                  is.finite(sxx) & sxx > 1e-10
  slope[!usable_slope] <- NA_real_

  E <- X - tcrossprod(slope, t) - intercept
  E[!is.finite(E)] <- 0
  rss <- rowSums(E * E)

  W <- matrix(rep(t, times = nrow_X), nrow = nrow_X, byrow = TRUE)
  W <- sweep(W, 1L, st / n_safe, "-")
  W <- sweep(W, 1L, sxx, "/")
  W[!obs | !is.finite(W)] <- 0
  U <- E * W
  gamma0 <- rowSums(U * U)

  hac_var <- function(max_lag) {
    variance <- gamma0
    for (k in seq_len(max_lag)) {
      before <- seq_len(Tlen - k)
      after <- before + k
      c_k <- rowSums(U[, before, drop = FALSE] * U[, after, drop = FALSE])
      variance <- variance + 2 * (1 - k / (max_lag + 1)) * c_k
    }
    variance <- variance * (n / pmax(df, 1L))
    # Bartlett weighting is nonnegative definite; protect round-off only.
    variance[is.finite(variance) & variance >= -1e-18 & variance < 0] <- 0
    variance[!is.finite(variance) | variance < 0] <- NA_real_
    variance
  }
  v4 <- hac_var(primary_hac_lag)
  v6 <- hac_var(sensitivity_hac_lag)
  v_ols <- (rss / pmax(df, 1L)) / sxx

  p_from_variance <- function(v, minimum_years) {
    p <- rep(NA_real_, nrow_X)
    ok <- n >= minimum_years & usable_slope & is.finite(v) & v >= 0 & df > 0
    positive <- ok & v > 1e-24
    p[positive] <- 2 * stats::pt(abs(slope[positive] / sqrt(v[positive])),
                                 df = df[positive], lower.tail = FALSE)
    deterministic <- ok & !positive
    p[deterministic] <- ifelse(abs(slope[deterministic]) < 1e-12, 1, 0)
    p
  }

  p_ols <- p_from_variance(v_ols, min_valid_years)
  p_hac4 <- p_from_variance(v4, min_hac_years)
  p_hac6 <- p_from_variance(v6, min_hac_years)
  se4 <- sqrt(v4)
  se6 <- sqrt(v6)
  se4[n < min_hac_years | !usable_slope] <- NA_real_
  se6[n < min_hac_years | !usable_slope] <- NA_real_

  cbind(n_valid = as.numeric(n), OLS_slope = slope,
        p_OLS = p_ols,
        p_HAC_L4 = p_hac4,
        p_HAC_L6 = p_hac6,
        se_HAC_L4 = se4,
        se_HAC_L6 = se6)
}

# Mathematical self-tests before processing real rasters.
set.seed(20261026L)
y0 <- 0.45 - 0.0024 * t + stats::rnorm(Tlen, 0, 0.02)
y_missing <- y0
y_missing[c(4L, 9L, 20L)] <- NA_real_
test_matrix <- rbind(y0, y_missing, rep(0.5, Tlen), rep(NA_real_, Tlen))
test_result <- pixel_hac_batch(test_matrix)
for (i in 1:2) {
  yy <- test_matrix[i, ]
  m <- stats::lm(yy ~ t)
  stopifnot(abs(test_result[i, "OLS_slope"] - stats::coef(m)[2L]) < 1e-10,
            abs(test_result[i, "p_OLS"] - summary(m)$coefficients[2L, 4L]) < 1e-9)
}
stopifnot(is.na(test_result[4L, "p_HAC_L4"]))
if (requireNamespace("sandwich", quietly = TRUE)) {
  m <- stats::lm(y0 ~ t)
  V4 <- sandwich::NeweyWest(m, lag = primary_hac_lag,
                             prewhite = FALSE, adjust = TRUE)
  stopifnot(abs(test_result[1L, "se_HAC_L4"] - sqrt(V4[2L, 2L])) < 1e-10)
  cat("Optional package comparison with sandwich::NeweyWest: PASSED\n")
}
cat("Synthetic self-tests passed (OLS slope, OLS p, empty series).\n")

# ================ 3A. TEMPORAL AUTOCORRELATION DIAGNOSTICS ==================
# Retain lag identities even when observations are missing.
# These are descriptive diagnostics, not corrected change-point p-values.
paired_cor_matrix <- function(A, B, required_pairs = min_diagnostic_pairs) {
  matched <- is.finite(A) & is.finite(B)
  n_pairs_by_row <- rowSums(matched)
  A[!matched] <- NA_real_
  B[!matched] <- NA_real_

  mean_A <- rowSums(A, na.rm = TRUE) / pmax(n_pairs_by_row, 1L)
  mean_B <- rowSums(B, na.rm = TRUE) / pmax(n_pairs_by_row, 1L)
  centered_A <- sweep(A, 1L, mean_A, "-")
  centered_B <- sweep(B, 1L, mean_B, "-")
  cross_sum <- rowSums(centered_A * centered_B, na.rm = TRUE)
  sum_A2 <- rowSums(centered_A * centered_A, na.rm = TRUE)
  sum_B2 <- rowSums(centered_B * centered_B, na.rm = TRUE)
  denominator <- sqrt(sum_A2 * sum_B2)
  rho <- cross_sum / denominator
  rho[n_pairs_by_row < required_pairs | !is.finite(rho) |
        denominator <= 1e-20] <- NA_real_
  rho <- pmax(-1, pmin(1, rho))
  list(rho = rho, pairs = as.numeric(n_pairs_by_row))
}

lag_diagnostic_batch <- function(X) {
  if (is.null(dim(X))) X <- matrix(as.numeric(X), nrow = 1L)
  X <- as.matrix(X)
  if (ncol(X) != length(years)) {
    stop("Unexpected number of annual layers in terra::app input.")
  }
  storage.mode(X) <- "double"
  X[!is.finite(X)] <- NA_real_
  nrows <- nrow(X)
  n <- rowSums(is.finite(X))

  # Fit one least-squares straight line to EACH pixel's available years.
  # It is used ONLY for residual autocorrelation diagnostics.
  ty <- as.numeric(years) - mean(as.numeric(years))
  valid <- is.finite(X)
  sum_y <- rowSums(X, na.rm = TRUE)
  sum_t <- as.numeric(valid %*% ty)
  sum_t2 <- as.numeric(valid %*% (ty * ty))
  sum_ty <- rowSums(sweep(X, 2L, ty, "*"), na.rm = TRUE)
  denom <- sum_t2 - (sum_t * sum_t) / pmax(n, 1L)
  slope <- (sum_ty - sum_t * sum_y / pmax(n, 1L)) / denom
  slope[n < 3L | !is.finite(denom) | denom <= 1e-12] <- NA_real_
  intercept <- (sum_y - slope * sum_t) / pmax(n, 1L)
  residuals <- X - tcrossprod(slope, ty) - intercept

  # Columns: raw_lag1..6, detrended_lag1..6, pairs_lag1..6
  result <- matrix(NA_real_, nrow = nrows, ncol = 3L * max_diagnostic_lag)
  for (k in seq_len(max_diagnostic_lag)) {
    first <- seq_len(ncol(X) - k)
    second <- first + k
    rc <- paired_cor_matrix(X[, first, drop = FALSE],
                            X[, second, drop = FALSE])
    dc <- paired_cor_matrix(residuals[, first, drop = FALSE],
                            residuals[, second, drop = FALSE])
    result[, k] <- rc$rho
    result[, max_diagnostic_lag + k] <- dc$rho
    result[, 2L * max_diagnostic_lag + k] <- rc$pairs
  }
  result
}


cat("Computing lag-1..6 raw/detrended RI correlations for eligible pixels...\n")
r_lags <- terra::app(r_stack, fun = lag_diagnostic_batch, cores = use_cores)
lag_names <- c(paste0("raw_lag", 1:max_diagnostic_lag),
               paste0("detrended_lag", 1:max_diagnostic_lag),
               paste0("pairs_lag", 1:max_diagnostic_lag))
stopifnot(terra::nlyr(r_lags) == length(lag_names))
names(r_lags) <- lag_names
r_lags <- terra::mask(r_lags, r_common_mask)
r_eligible_area <- terra::mask(r_cell_area, r_common_mask)
names(r_eligible_area) <- "cell_area_km2"
df_lags <- terra::as.data.frame(c(r_lags, r_eligible_area), na.rm = FALSE)
df_lags <- df_lags[is.finite(df_lags$cell_area_km2), , drop = FALSE]
if (!nrow(df_lags)) stop("No valid lag-diagnostic pixels")
lag_summary <- function(nm, kind, k) {
  rho <- df_lags[[nm]]
  good <- is.finite(rho)
  aa <- df_lags$cell_area_km2[good]
  z <- rho[good]
  tested_area <- sum(aa)
  data.frame(series = kind, lag_years = k,
             valid_pixel_count = sum(good),
             valid_area_km2 = tested_area,
             valid_area_percent_of_common = 100 * tested_area / common_area,
             area_weighted_mean_rho = if(length(z)) weighted.mean(z, aa) else NA_real_,
             median_rho = if(length(z)) median(z) else NA_real_,
             pct_estimable_area_rho_gt_0_3 = if(length(z)) 100*sum(aa[z>0.3])/tested_area else NA_real_,
             pct_estimable_area_rho_gt_0_5 = if(length(z)) 100*sum(aa[z>0.5])/tested_area else NA_real_)
}
acf_sum <- do.call(rbind, c(
  lapply(1:max_diagnostic_lag, function(k)
    lag_summary(paste0("raw_lag",k), "raw", k)),
  lapply(1:max_diagnostic_lag, function(k)
    lag_summary(paste0("detrended_lag",k), "detrended_residual", k))))
utils::write.csv(acf_sum, file.path(out_dir, "00_Pixel_ACF_Summary.csv"),
                 row.names = FALSE)

# ===================== 4. REGIONAL HAC 1..6 ================================

# Recompute the EXACT region arithmetic means used in Phase 1.
regional_df <- data.frame(
  year = years,
  mean_RI = as.numeric(terra::global(r_stack, "mean", na.rm = TRUE)[, 1L]),
  n_pixels = as.numeric(terra::global(r_stack, "notNA")[, 1L])
)
utils::write.csv(regional_df,
  file.path(out_dir, "02_Regional_Annual_Mean_RI_Recomputed.csv"),
  row.names = FALSE, fileEncoding = "UTF-8")
if (any(!is.finite(regional_df$mean_RI))) stop("Regional annual mean contains NA")

reg_y <- regional_df$mean_RI
reg_fit <- stats::lm(reg_y ~ t)
reg_slope <- unname(stats::coef(reg_fit)[2L])
reg_ols_p <- summary(reg_fit)$coefficients[2L, 4L]
reg_res <- as.numeric(stats::residuals(reg_fit))

# Regional lag-1..6 correlation diagnostics (same annual calendar lags).
reg_pair_cor <- function(z,k) {
  left <- z[seq_len(length(z)-k)]
  right <- z[(k+1L):length(z)]
  good <- is.finite(left) & is.finite(right)
  if (sum(good) < min_diagnostic_pairs ||
      stats::sd(left[good]) <= 1e-12 || stats::sd(right[good]) <= 1e-12)
    return(NA_real_)
  unname(stats::cor(left[good], right[good]))
}
regional_acf <- data.frame(lag_years = 1:max_diagnostic_lag,
  raw_rho = vapply(1:max_diagnostic_lag,
    function(k) reg_pair_cor(reg_y,k), numeric(1)),
  detrended_residual_rho = vapply(1:max_diagnostic_lag,
    function(k) reg_pair_cor(reg_res,k), numeric(1)))
utils::write.csv(regional_acf,
  file.path(out_dir, "00_Regional_ACF_Summary.csv"), row.names = FALSE)

reg_u <- reg_res * t / sum(t^2)
reg_df <- Tlen - 2L
reg_hac_test <- function(L) {
  v <- sum(reg_u^2)
  for (k in seq_len(L)) {
    v <- v + 2 * (1 - k / (L + 1)) *
      sum(reg_u[seq_len(Tlen - k)] * reg_u[(k + 1L):Tlen])
  }
  v <- max(0, v * Tlen / reg_df)
  se <- sqrt(v)
  tt <- if (se > 0) reg_slope / se else NA_real_
  pp <- if (is.finite(tt)) 2 * stats::pt(abs(tt), df = reg_df,
                                         lower.tail = FALSE) else NA_real_
  crit <- stats::qt(0.975, df = reg_df)
  data.frame(method = paste0("HAC_Bartlett_L", L), lag_years = L,
             n_years = Tlen, slope_per_year = reg_slope,
             se_slope = se, p_value = pp,
             ci95_lower = reg_slope - crit * se,
             ci95_upper = reg_slope + crit * se)
}
reg_ols_se <- summary(reg_fit)$coefficients[2L, 2L]
reg_crit <- stats::qt(0.975, df = reg_df)
reg_tests <- rbind(
  data.frame(method = "Ordinary_OLS", lag_years = NA_integer_,
             n_years = Tlen, slope_per_year = reg_slope,
             se_slope = reg_ols_se, p_value = reg_ols_p,
             ci95_lower = reg_slope - reg_crit * reg_ols_se,
             ci95_upper = reg_slope + reg_crit * reg_ols_se),
  do.call(rbind, lapply(seq_len(sensitivity_hac_lag), reg_hac_test))
)
utils::write.csv(reg_tests, file.path(out_dir, "03_Regional_OLS_and_HAC_L1to6.csv"),
                 row.names = FALSE, fileEncoding = "UTF-8")
print(reg_tests)

# ========================= 5. ALL-PIXEL HAC ================================
cat("Computing all-pixel calendar-time OLS and HAC tests...\n")
r_stats <- terra::app(r_stack, fun = pixel_hac_batch, cores = use_cores)
stopifnot(terra::nlyr(r_stats) == 7L)
names(r_stats) <- c("n_valid", "OLS_slope", "p_OLS", "p_HAC_L4",
                    "p_HAC_L6", "se_HAC_L4", "se_HAC_L6")
r_stats <- terra::mask(r_stats, r_common_mask)
terra::writeRaster(r_stats,
  file.path(out_dir, "04_Pixel_Linear_OLS_HAC_L4_L6.tif"),
  overwrite = FALSE, datatype = "FLT8S", NAflag = -9999,
  gdal = c("COMPRESS=LZW"))

# ====================== 6. ALL-PIXEL FDR ADJUST =============================
# One FDR family per independent linear-slope specification. Never pool p_OLS,
# p_HAC_L4 and p_HAC_L6 in one family. BH is primary; BY is conservative check.
read_layer <- function(layer) as.numeric(terra::values(layer, mat = FALSE))
slope_vals <- read_layer(r_stats[["OLS_slope"]])
n_vals <- read_layer(r_stats[["n_valid"]])
p_ols <- read_layer(r_stats[["p_OLS"]])
p_ols_matched <- p_ols
p_ols_matched[!in_common | n_vals < min_hac_years] <- NA_real_
p4 <- read_layer(r_stats[["p_HAC_L4"]])
p6 <- read_layer(r_stats[["p_HAC_L6"]])
if (length(p4) != length(in_common)) stop("Raster length mismatch")

adjust_global <- function(p, method) {
  out <- rep(NA_real_, length(p))
  eligible <- in_common & is.finite(p) & p >= 0 & p <= 1
  if (any(eligible)) out[eligible] <- stats::p.adjust(p[eligible], method = method)
  out
}
q_ols_matched_bh <- adjust_global(p_ols_matched, "BH")
q4_bh <- adjust_global(p4, "BH")
q4_by <- adjust_global(p4, "BY")
q6_bh <- adjust_global(p6, "BH")

to_raster <- function(vals, layer_name) {
  ans <- terra::setValues(r_stats[[1L]], vals)
  names(ans) <- layer_name
  ans
}
r_q <- c(to_raster(q_ols_matched_bh, "q_OLS_matched_BH"),
         to_raster(q4_bh, "q_HAC_L4_BH"),
         to_raster(q4_by, "q_HAC_L4_BY"),
         to_raster(q6_bh, "q_HAC_L6_BH"))
terra::writeRaster(r_q,
  file.path(out_dir, "05_Pixel_Global_FDR_qvalues.tif"),
  overwrite = FALSE, datatype = "FLT8S", NAflag = -9999,
  gdal = c("COMPRESS=LZW"))

# Status 0 = not significant linear slope; 1 = significant positive;
#        2 = significant negative; 3 = not formally testable (<25 yr/etc).
# Outside the common descriptive extent remains NoData.
make_status <- function(p_or_q) {
  stat <- rep(NA_real_, length(p_or_q))
  stat[in_common] <- 3
  testable <- in_common & is.finite(p_or_q) & is.finite(slope_vals)
  stat[testable] <- 0
  stat[testable & p_or_q < alpha & slope_vals > 0] <- 1
  stat[testable & p_or_q < alpha & slope_vals < 0] <- 2
  stat
}
status_ols <- make_status(p_ols)
status_ols_matched <- make_status(p_ols_matched)
status_ols_matched_bh <- make_status(q_ols_matched_bh)
status_hac4_raw <- make_status(p4)
status_hac4_bh <- make_status(q4_bh)
status_hac4_by <- make_status(q4_by)
status_hac6_bh <- make_status(q6_bh)

r_status <- c(to_raster(status_hac4_bh, "linear_HAC_L4_BH_status"),
              to_raster(status_hac4_by, "linear_HAC_L4_BY_status"),
              to_raster(status_hac6_bh, "linear_HAC_L6_BH_status"))
terra::writeRaster(r_status,
  file.path(out_dir, "06_Pixel_Linear_Significance_Status.tif"),
  overwrite = FALSE, datatype = "INT1U", NAflag = 255,
  gdal = c("COMPRESS=LZW"))

# ====================== 7. AREA SUMMARIES ===================================
status_labels <- c("Linear slope not significant (NOT overall no-trend)",
                   "Significant increasing linear slope",
                   "Significant decreasing linear slope",
                   "Not testable under this specification")
area_summary <- function(z, method_name) {
  testable <- in_common & z %in% c(0, 1, 2)
  area_testable <- sum(area_vals[testable], na.rm = TRUE)
  do.call(rbind, lapply(0:3, function(code) {
    sel <- in_common & !is.na(z) & z == code
    area_here <- sum(area_vals[sel], na.rm = TRUE)
    data.frame(method = method_name, status_code = code,
               status_description = status_labels[code + 1L],
               pixel_count = sum(sel), area_km2 = area_here,
               pct_common_area = 100 * area_here / common_area,
               pct_testable_area = if (code == 3 || area_testable == 0) NA_real_
                                   else 100 * area_here / area_testable)
  }))
}
summary_table <- do.call(rbind, list(
  area_summary(status_ols, "OLS_raw_original_10plus"),
  area_summary(status_ols_matched, "OLS_raw_matched_25plus"),
  area_summary(status_ols_matched_bh, "OLS_BH_matched_25plus"),
  area_summary(status_hac4_raw, "HAC_L4_raw_p"),
  area_summary(status_hac4_bh, "HAC_L4_BH_primary"),
  area_summary(status_hac4_by, "HAC_L4_BY_conservative"),
  area_summary(status_hac6_bh, "HAC_L6_BH_sensitivity")
))
utils::write.csv(summary_table,
  file.path(out_dir, "07_Linear_Significance_Comparison_Area.csv"),
  row.names = FALSE, fileEncoding = "UTF-8")

# Report how p and q-values are distributed, even when few pass BH.
p_summary <- function(x, name) {
  vals <- x[in_common & is.finite(x)]
  data.frame(item = name, n = length(vals),
             min = if (length(vals)) min(vals) else NA_real_,
             q05 = if (length(vals)) unname(quantile(vals, 0.05)) else NA_real_,
             median = if (length(vals)) median(vals) else NA_real_,
             q95 = if (length(vals)) unname(quantile(vals, 0.95)) else NA_real_,
             fraction_below_0_05 = if (length(vals)) mean(vals < 0.05) else NA_real_)
}
p_table <- do.call(rbind, list(p_summary(p_ols, "p_OLS"),
  p_summary(p_ols_matched, "p_OLS_matched_25plus"),
  p_summary(q_ols_matched_bh, "q_OLS_BH_matched_25plus"),
  p_summary(p4, "p_HAC_L4"), p_summary(q4_bh, "q_HAC_L4_BH"),
  p_summary(q4_by, "q_HAC_L4_BY"), p_summary(p6, "p_HAC_L6"),
  p_summary(q6_bh, "q_HAC_L6_BH")))
utils::write.csv(p_table,
  file.path(out_dir, "08_P_and_Q_Distribution.csv"),
  row.names = FALSE, fileEncoding = "UTF-8")

metadata <- data.frame(
  item = c("Purpose", "Original_script", "Raster_folder", "Grass_mask",
           "Start_year", "End_year", "Input_year_count", "Original_min_years",
           "Formal_HAC_min_years", "Primary_lag", "Sensitivity_lag",
           "HAC_kernel", "HAC_finite_sample_adjust", "HAC_p_reference",
           "FDR_primary", "FDR_sensitivity", "Significance_alpha",
           "Common_pixel_count", "Common_area_km2", "HAC4_testable_pixels",
           "HAC4_testable_area_km2", "Regional_OLS_slope_per_year",
           "Change_point_tests_completed", "Final_trend_reclassification_completed",
           "Output_directory"),
  value = c("Regional + pixel LINEAR inference only", "02_RI_spatial_analysis.R",
            "data/RI_annual (or RI_ANNUAL_DIR)", "data/mask/grassland.shp (or RI_GRASS_MASK)", start_year, end_year, length(files),
            min_valid_years, min_hac_years, primary_hac_lag, sensitivity_hac_lag,
            "Bartlett", "n/(n-2)", "t(df=n-2)", "BH", "BY and lag 6 BH",
            alpha, common_count, format(common_area, digits = 13),
            sum(in_common & is.finite(p4)),
            format(sum(area_vals[in_common & is.finite(p4)], na.rm = TRUE), digits = 13),
            format(reg_slope, digits = 13), "NO", "NO", "output/reviewer2_comment9/04_sensitivity/Run_*" )
)
utils::write.csv(metadata,
  file.path(out_dir, "09_Phase2A_Metadata.csv"),
  row.names = FALSE, fileEncoding = "UTF-8")

# =========================== 8. QC AND README ===============================
expected_common <- common_count
for (method in unique(summary_table$method)) {
  z <- summary_table[summary_table$method == method, ]
  stopifnot(sum(z$pixel_count) == expected_common,
            abs(sum(z$area_km2) - common_area) < max(1e-6, common_area * 1e-9))
}
if (any(q4_bh[in_common & is.finite(q4_bh)] <
        p4[in_common & is.finite(q4_bh)] - 1e-12)) {
  stop("FDR q-value must not be smaller than the corresponding p-value")
}
readme_lines <- c(
  "Reviewer 2 Comment 9 (3.5): autocorrelation diagnostics + OLS/HAC/FDR sensitivity",
  "00_Pixel_ACF_Summary.csv and 00_Regional_ACF_Summary.csv: raw and detrended time correlations.",
  "",
  "02: 36-year area-unweighted regional RI arithmetic means; matches Phase 1.",
  "03: Regional ordinary OLS and HAC(1..6) slope/SE/p/95% CI.",
  "04: Seven-band all-pixel OLS and HAC raster (calendar-year lags 4/6).",
  "05: Four q-value bands: matched OLS BH, HAC4 BH, HAC4 BY, HAC6 BH.",
  "06: Three-band status rasters. 0=linear nonsignificant; 1=increasing;",
  "    2=decreasing; 3=not formally testable. Outside original mask=NoData.",
  "07: OLS raw (10+ and matched 25+), OLS BH, HAC4/HAC6: area tables.",
  "08: Distributions of linear p-values and q-values.",
  "09: Run settings and coverage metadata.",
  "",
  "HAC: OLS + Bartlett/Newey-West with finite sample n/(n-2), t(n-2).",
  "Missing calendar years are not compressed to adjacent time points.",
  "Primary HAC lag 4 covers the direct overlap period of the 5-year window.",
  "Lag 6 and conservative BY-FDR are sensitivity analyses, not separate confirmations.",
  "Original valid mask uses at least 10 RI years; inference needs at least 25.",
  "A non-significant linear trend DOES NOT rule out a piecewise turning point.",
  "No Pettitt test, break-point search, final AIC selection, final trend classes,",
  "nor new Figure 4/5 and Table 2 values have been produced here.",
  "DO NOT substitute these linear status maps for the old three-class trend map.",
  "",
  "References:",
  "Newey & West (1987), Econometrica 55(3), 703-708, DOI 10.2307/1913610.",
  "Benjamini & Hochberg (1995), JRSS B 57(1), 289-300,",
  "DOI 10.1111/j.2517-6161.1995.tb02031.x.",
  "Benjamini & Yekutieli (2001), Annals of Statistics 29(4), 1165-1188.",
  "",
  "CAUTION: Linear HAC+FDR is a sensitivity analysis, not a fully validated turning-point significance result.",
  "Caution: HAC is a short-series asymptotic procedure. Check dependence,",
  "coverage and breakpoint model structure before scientific interpretation."
)
writeLines(readme_lines, con = file.path(out_dir, "README_Phase2A.txt"), useBytes = TRUE)
cat("\nCompleted Phase 2A. Important files: 03, 07, 08, 09.\n")
cat("Output: ", out_dir, "\n", sep = "")
print(summary_table)
