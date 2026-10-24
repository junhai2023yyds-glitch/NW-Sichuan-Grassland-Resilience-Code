# ============================================================
# Grassland Ecosystem Resilience Index (RI) Calculation
# Northwestern Sichuan, China | 1985-2020
# Public/reproducible repository version
# ============================================================
#
# Method:
#   Monthly NDVI, temperature, and precipitation data from 1983-2022
#   are analyzed using a 5-year moving window with a 1-year step.
#
#   Within each 5-year window (60 months), each variable is standardized
#   pixel by pixel using z-scores. The following AR(1) multiple regression
#   is then fitted for each pixel:
#
#     NDVI_t = alpha * Temp_t + beta * Pre_t
#              + gamma * NDVI_(t-1) + epsilon_t
#
#   The lag-1 coefficient gamma represents vegetation memory.
#   IMPORTANT: The raw gamma rasters are computed for ALL 36 midpoint years FIRST.
#   Then ONE common gamma_min and ONE common gamma_max are obtained from all
#   valid grassland pixels and all midpoint years (1985-2020).
#   The SAME two bounds are used for every year (NOT per year/window/pixel):
#
#     gamma_norm(i,t) = (gamma(i,t) - gamma_min_global) /
#                       (gamma_max_global - gamma_min_global)
#
#   The Resilience Index is:
#
#     RI = 1 - gamma_norm
#
#   The midpoint year of each 5-year window is used as the RI year:
#     1983-1987 -> 1985
#     ...
#     2018-2022 -> 2020
#
# Repository structure:
#
#   project_root/
#   ├── 01_RI_calculation.R
#   ├── data/
#   │   ├── NDVI/              # monthly NDVI GeoTIFFs
#   │   ├── TEMP/              # monthly temperature GeoTIFFs
#   │   ├── PRE/               # monthly precipitation GeoTIFFs
#   │   └── mask/
#   │       └── grassland.shp  # grassland mask (+ .dbf/.shx/.prj etc.)
#   └── output/
#       └── RI_annual/         # created automatically
#
# File naming:
#   Each monthly GeoTIFF filename must contain a recognizable year-month,
#   e.g. NDVI_198301.tif, TEMP_1983_01.tif, PRE_1983-01.tif.
#
# IMPORTANT:
# - No local absolute paths are hard-coded.
# - Input rasters are expected to have identical CRS, extent, resolution,
#   rows, and columns. The script stops rather than silently resampling.
# - The grassland mask is applied before fitting/normalization; one global
#   min-max domain covers all valid grassland pixel-year gamma estimates.
# - Existing RI GeoTIFFs are NEVER silently overwritten (default FALSE).
# - Two passes are essential: 36 raw gamma rasters -> shared bounds -> 36 RI.
# - Monthly z-scoring *within* a five-year fit window is DISTINCT from
#   global gamma min-max normalization *across* windows.
# ============================================================

rm(list = ls())
gc()

library(terra)

options(scipen = 999)

# ===================== 1. Project paths =====================

# Run this script with the repository root as the working directory.
project_root <- normalizePath(".", winslash = "/", mustWork = FALSE)

data_root <- file.path(project_root, "data")

ndvi_dir <- file.path(data_root, "NDVI")
temp_dir <- file.path(data_root, "TEMP")
pre_dir  <- file.path(data_root, "PRE")

grass_shp_path <- file.path(data_root, "mask", "grassland.shp")

output_root <- file.path(project_root, "output")
ri_output_dir <- file.path(output_root, "RI_annual")
diagnostic_dir <- file.path(output_root, "RI_diagnostics")

dir.create(ri_output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(diagnostic_dir, recursive = TRUE, showWarnings = FALSE)

# ===================== 2. Analysis settings =====================

start_year <- 1983
end_year   <- 2022

window_years <- 5
window_months <- window_years * 12
step_years <- 1

# A complete 5-year window provides 59 paired observations after lagging NDVI.
# To reproduce the stated method exactly, all 59 paired observations are required.
min_valid_pairs <- window_months - 1

# Number of cores used by terra::app().
# Use 1 for maximum stability; increase only after testing on your system.
use_cores <- 1

# FALSE protects existing annual RI rasters.
# Set TRUE only when you intentionally want to replace existing outputs.
overwrite_outputs <- FALSE

# Keep all 36 raw-gamma rasters and additionally save normalized-gamma
# rasters for independent audit. FALSE uses temporary raw-gamma files which
# are deleted only after the WHOLE run finishes successfully.
# Recommended TRUE for transparent response to the journal reviewer.
write_gamma_rasters <- TRUE

# ===================== 3. Input checks =====================

required_dirs <- c(ndvi_dir, temp_dir, pre_dir)

for (d in required_dirs) {
  if (!dir.exists(d)) {
    stop(
      paste0(
        "Required input directory not found: ", d, "\n",
        "Please create the directory and place monthly GeoTIFF files inside."
      )
    )
  }
}

if (!file.exists(grass_shp_path)) {
  stop(
    paste0(
      "Grassland mask not found: ", grass_shp_path, "\n",
      "Please place the shapefile and its companion files in data/mask/."
    )
  )
}

# ===================== 4. Monthly file indexing =====================

# Extract YYYYMM from filenames.
# Supported examples:
#   NDVI_198301.tif
#   NDVI_1983_01.tif
#   NDVI_1983-01.tif
extract_yyyymm <- function(x) {

  nm <- tools::file_path_sans_ext(basename(x))

  m <- regexpr(
    "(19|20)[0-9]{2}[-_]?(0[1-9]|1[0-2])",
    nm,
    perl = TRUE
  )

  if (m[1] == -1) {
    return(NA_character_)
  }

  token <- regmatches(nm, m)
  token <- gsub("[-_]", "", token)

  if (nchar(token) != 6) {
    return(NA_character_)
  }

  token
}

index_monthly_files <- function(folder, variable_name) {

  files <- list.files(
    folder,
    pattern = "\\.tif(f)?$",
    full.names = TRUE,
    ignore.case = TRUE
  )

  if (length(files) == 0) {
    stop(paste0("No GeoTIFF files found for ", variable_name, ": ", folder))
  }

  yyyymm <- vapply(files, extract_yyyymm, character(1))

  if (any(is.na(yyyymm))) {
    bad <- basename(files[is.na(yyyymm)])
    stop(
      paste0(
        "Could not identify YYYYMM in the following ", variable_name,
        " filename(s):\n",
        paste(bad, collapse = "\n"),
        "\nUse filenames containing YYYYMM, YYYY_MM, or YYYY-MM."
      )
    )
  }

  dates <- as.Date(
    paste0(substr(yyyymm, 1, 4), "-", substr(yyyymm, 5, 6), "-01")
  )

  out <- data.frame(
    date = dates,
    yyyymm = yyyymm,
    file = files,
    stringsAsFactors = FALSE
  )

  out <- out[
    as.integer(format(out$date, "%Y")) >= start_year &
      as.integer(format(out$date, "%Y")) <= end_year,
  ]

  out <- out[order(out$date), ]

  if (anyDuplicated(out$date)) {
    dup_dates <- unique(out$date[duplicated(out$date)])
    stop(
      paste0(
        "Duplicate monthly files detected for ", variable_name, ": ",
        paste(format(dup_dates, "%Y-%m"), collapse = ", ")
      )
    )
  }

  expected_dates <- seq(
    as.Date(sprintf("%d-01-01", start_year)),
    as.Date(sprintf("%d-12-01", end_year)),
    by = "month"
  )

  missing_dates <- setdiff(expected_dates, out$date)

  if (length(missing_dates) > 0) {
    stop(
      paste0(
        "Missing monthly ", variable_name, " files for:\n",
        paste(format(missing_dates, "%Y-%m"), collapse = ", ")
      )
    )
  }

  if (nrow(out) != length(expected_dates)) {
    stop(
      paste0(
        variable_name, " should contain ",
        length(expected_dates), " monthly files for ",
        start_year, "-", end_year,
        ", but ", nrow(out), " were indexed."
      )
    )
  }

  out
}

cat("\nIndexing monthly input files...\n")

ndvi_index <- index_monthly_files(ndvi_dir, "NDVI")
temp_index <- index_monthly_files(temp_dir, "temperature")
pre_index  <- index_monthly_files(pre_dir, "precipitation")

if (!identical(ndvi_index$date, temp_index$date) ||
    !identical(ndvi_index$date, pre_index$date)) {
  stop("NDVI, temperature, and precipitation monthly dates do not match exactly.")
}

all_dates <- ndvi_index$date

cat(
  "Monthly period:",
  format(min(all_dates), "%Y-%m"),
  "to",
  format(max(all_dates), "%Y-%m"),
  "\n"
)

cat("Number of months:", length(all_dates), "\n")

# ===================== 5. Raster geometry check =====================

cat("\nChecking raster geometry...\n")

reference_raster <- rast(ndvi_index$file[1])

check_geometry <- function(file, reference, variable_name) {

  r <- rast(file)

  ok <- compareGeom(
    reference,
    r,
    crs = TRUE,
    ext = TRUE,
    rowcol = TRUE,
    res = TRUE,
    stopOnError = FALSE
  )

  if (!isTRUE(ok)) {
    stop(
      paste0(
        "Raster geometry mismatch detected in ", variable_name, ":\n",
        file,
        "\nAll monthly rasters must share the same CRS, extent, resolution, rows, and columns."
      )
    )
  }

  invisible(TRUE)
}

for (f in ndvi_index$file) {
  check_geometry(f, reference_raster, "NDVI")
}

for (f in temp_index$file) {
  check_geometry(f, reference_raster, "temperature")
}

for (f in pre_index$file) {
  check_geometry(f, reference_raster, "precipitation")
}

cat("Raster geometry check passed.\n")

# ===================== 6. Grassland analysis mask =====================

grass_shp <- vect(grass_shp_path)

if (!same.crs(reference_raster, grass_shp)) {
  grass_shp <- project(grass_shp, crs(reference_raster))
}

# Create a raster mask aligned exactly with the input raster template.
grass_mask <- rasterize(
  grass_shp,
  reference_raster,
  field = 1,
  background = NA
)

grass_mask <- ifel(!is.na(grass_mask), 1, NA)

# ===================== 7. Pixel-wise helper functions =====================

zscore_vec <- function(v) {

  if (anyNA(v)) {
    return(rep(NA_real_, length(v)))
  }

  s <- sd(v)

  if (is.na(s) || s <= 0) {
    return(rep(NA_real_, length(v)))
  }

  (v - mean(v)) / s
}

# Input vector layout for each pixel:
#   [NDVI 60 months] [TEMP 60 months] [PRE 60 months]
#
# Returns:
#   alpha, beta, gamma, n_pairs
fit_ar1_pixel <- function(x) {

  n <- window_months

  ndvi <- x[1:n]
  temp <- x[(n + 1):(2 * n)]
  pre  <- x[(2 * n + 1):(3 * n)]

  # Require complete monthly values for exact reproduction of the 60-month window.
  if (anyNA(ndvi) || anyNA(temp) || anyNA(pre)) {
    return(c(NA_real_, NA_real_, NA_real_, NA_real_))
  }

  ndvi_z <- zscore_vec(ndvi)
  temp_z <- zscore_vec(temp)
  pre_z  <- zscore_vec(pre)

  if (all(is.na(ndvi_z)) ||
      all(is.na(temp_z)) ||
      all(is.na(pre_z))) {
    return(c(NA_real_, NA_real_, NA_real_, NA_real_))
  }

  # Current month t = months 2...60
  y <- ndvi_z[2:n]

  # Predictors at time t and lagged NDVI at t-1.
  X <- cbind(
    temp_z[2:n],
    pre_z[2:n],
    ndvi_z[1:(n - 1)]
  )

  ok <- complete.cases(y, X)
  n_pairs <- sum(ok)

  if (n_pairs < min_valid_pairs) {
    return(c(NA_real_, NA_real_, NA_real_, n_pairs))
  }

  y <- y[ok]
  X <- X[ok, , drop = FALSE]

  # Equation in the stated method does not include an intercept.
  # lm.fit() fits exactly the supplied design matrix.
  fit <- try(
    lm.fit(
      x = X,
      y = y
    ),
    silent = TRUE
  )

  if (inherits(fit, "try-error")) {
    return(c(NA_real_, NA_real_, NA_real_, n_pairs))
  }

  coef <- fit$coefficients

  if (length(coef) != 3 || any(!is.finite(coef))) {
    return(c(NA_real_, NA_real_, NA_real_, n_pairs))
  }

  # alpha = temperature coefficient
  # beta  = precipitation coefficient
  # gamma = lagged NDVI coefficient
  c(coef[1], coef[2], coef[3], n_pairs)
}

# ===================== 8. Set up 36 five-year windows =====================
# DO NOT use a separate min-max normalization for each window.
# First calculate all 36 original gamma rasters, then determine a SINGLE
# gamma_min_global / gamma_max_global from their combined valid pixels.

window_start_years <- seq(
  start_year,
  end_year - window_years + 1,
  by = step_years
)

midpoint_years <- window_start_years + floor(window_years / 2)
expected_mid_years <- 1985:2020

if (!identical(as.integer(midpoint_years), as.integer(expected_mid_years))) {
  stop("Midpoint years are not exactly 1985-2020. Check analysis settings.")
}

n_windows <- length(window_start_years)
cat("\nNumber of 5-year windows:", n_windows, "\n")
cat("RI midpoint years:", min(midpoint_years), "to", max(midpoint_years), "\n")

# Raw gamma rasters must exist on disk between pass 1 and pass 2.
# When write_gamma_rasters is FALSE, the working folder is removed only
# after successful completion. When TRUE, raw gamma rasters are retained.
gamma_work_dir <- file.path(
  diagnostic_dir,
  if (write_gamma_rasters) "gamma_raw_global" else "gamma_raw_temporary"
)

raw_gamma_files <- file.path(
  gamma_work_dir,
  paste0("gamma_raw_", midpoint_years, ".tif")
)

ri_files <- file.path(
  ri_output_dir,
  paste0("RI_", midpoint_years, ".tif")
)

gamma_norm_files <- file.path(
  diagnostic_dir,
  paste0("gamma_norm_", midpoint_years, ".tif")
)

bounds_csv <- file.path(
  diagnostic_dir,
  "RI_global_gamma_bounds_1985_2020.csv"
)
diagnostic_csv <- file.path(
  diagnostic_dir,
  "RI_calculation_log_1985_2020.csv"
)

# Preflight: fail BEFORE calculating anything if old files would be overwritten.
protected_files <- c(raw_gamma_files, ri_files, bounds_csv, diagnostic_csv)
if (write_gamma_rasters) {
  protected_files <- c(protected_files, gamma_norm_files)
}

if (!overwrite_outputs && any(file.exists(protected_files))) {
  existing <- protected_files[file.exists(protected_files)]
  stop(paste0(
    "One or more output files already exist. No existing results were changed.\n",
    "Example(s):\n", paste(head(existing, 10), collapse = "\n"),
    "\nUse a fresh output directory, archive old results, or explicitly set ",
    "overwrite_outputs <- TRUE only after confirming replacement."
  ))
}

dir.create(gamma_work_dir, recursive = TRUE, showWarnings = FALSE)

# ===================== 9. PASS 1: fit raw gamma for all years =====================

raw_gamma_stats <- vector("list", n_windows)

for (i in seq_len(n_windows)) {

  ws <- window_start_years[i]
  we <- ws + window_years - 1
  mid_year <- midpoint_years[i]

  cat(sprintf(
    "[PASS 1: %02d/%02d] Fitting raw gamma %d-%d -> %d\n",
    i, n_windows, ws, we, mid_year
  ))

  window_start_date <- as.Date(sprintf("%d-01-01", ws))
  window_end_date <- as.Date(sprintf("%d-12-01", we))
  idx <- which(all_dates >= window_start_date & all_dates <= window_end_date)

  if (length(idx) != window_months) {
    stop(sprintf(
      "Window %d-%d has %d months; expected %d.",
      ws, we, length(idx), window_months
    ))
  }

  # Retain the original pixelwise, within-window AR(1) fitting procedure.
  ndvi_window <- rast(ndvi_index$file[idx])
  temp_window <- rast(temp_index$file[idx])
  pre_window <- rast(pre_index$file[idx])

  names(ndvi_window) <- paste0("NDVI_", format(all_dates[idx], "%Y%m"))
  names(temp_window) <- paste0("TEMP_", format(all_dates[idx], "%Y%m"))
  names(pre_window) <- paste0("PRE_", format(all_dates[idx], "%Y%m"))

  ndvi_window <- mask(ndvi_window, grass_mask)
  temp_window <- mask(temp_window, grass_mask)
  pre_window <- mask(pre_window, grass_mask)

  model_stack <- c(ndvi_window, temp_window, pre_window)
  coef_rasters <- app(model_stack, fun = fit_ar1_pixel, cores = use_cores)
  names(coef_rasters) <- c("alpha", "beta", "gamma", "n_pairs")
  gamma_raw <- coef_rasters[["gamma"]]

  # Important: save RAW gamma, not per-window-normalized gamma.
  # All global extrema below are obtained from these actual saved rasters.
  writeRaster(
    gamma_raw,
    raw_gamma_files[i],
    overwrite = overwrite_outputs,
    datatype = "FLT4S",
    NAflag = -9999,
    gdal = c("COMPRESS=LZW")
  )

  saved_gamma <- rast(raw_gamma_files[i])
  gamma_mm <- global(saved_gamma, fun = c("min", "max"), na.rm = TRUE)
  y_min <- gamma_mm[1, "min"]
  y_max <- gamma_mm[1, "max"]

  if (!is.finite(y_min) || !is.finite(y_max)) {
    stop(sprintf("No valid fitted gamma values for midpoint year %d.", mid_year))
  }

  n_valid <- global(
    ifel(!is.na(saved_gamma), 1, NA_real_),
    fun = "sum", na.rm = TRUE
  )[1, 1]

  raw_gamma_stats[[i]] <- data.frame(
    window_start = ws,
    window_end = we,
    midpoint_year = mid_year,
    n_months = length(idx),
    gamma_min_raw = as.numeric(y_min),
    gamma_max_raw = as.numeric(y_max),
    valid_cells = as.numeric(n_valid),
    raw_gamma_file = basename(raw_gamma_files[i]),
    stringsAsFactors = FALSE
  )

  rm(ndvi_window, temp_window, pre_window, model_stack,
     coef_rasters, gamma_raw, saved_gamma)
  gc()
}

raw_stats <- do.call(rbind, raw_gamma_stats)

# The ONLY normalization extrema, obtained across ALL valid pixels/years.
gamma_min_global <- min(raw_stats$gamma_min_raw)
gamma_max_global <- max(raw_stats$gamma_max_raw)

if (!is.finite(gamma_min_global) || !is.finite(gamma_max_global) ||
    gamma_max_global <= gamma_min_global) {
  stop("Global gamma min-max normalization is undefined (invalid bounds).")
}

cat(sprintf(
  "\nGLOBAL gamma bounds over ALL 36 years: min = %.12g; max = %.12g\n",
  gamma_min_global, gamma_max_global
))
cat("These SAME two values will be used for EVERY RI raster.\n\n")

bounds_table <- data.frame(
  normalization_method = "global_min_max_all_valid_pixels_all_years",
  midpoint_start_year = min(midpoint_years),
  midpoint_end_year = max(midpoint_years),
  n_midpoint_years = n_windows,
  n_valid_pixel_year_values = sum(raw_stats$valid_cells),
  gamma_min_global = gamma_min_global,
  gamma_max_global = gamma_max_global,
  gamma_range_global = gamma_max_global - gamma_min_global,
  stringsAsFactors = FALSE
)
write.csv(bounds_table, bounds_csv, row.names = FALSE, fileEncoding = "UTF-8")

# ===================== 10. PASS 2: one global scale -> RI =====================

diagnostic_rows <- vector("list", n_windows)

for (i in seq_len(n_windows)) {
  mid_year <- midpoint_years[i]
  cat(sprintf(
    "[PASS 2: %02d/%02d] Applying SHARED gamma bounds -> RI %d\n",
    i, n_windows, mid_year
  ))

  gamma_raw <- rast(raw_gamma_files[i])
  gamma_norm <- (
    gamma_raw - gamma_min_global
  ) / (
    gamma_max_global - gamma_min_global
  )

  # Global min-max is already mathematically bounded on the fitting domain.
  # Clamping only prevents possible floating-point epsilon deviations.
  gamma_norm <- clamp(gamma_norm, lower = 0, upper = 1, values = TRUE)
  names(gamma_norm) <- "gamma_norm"

  ri <- 1 - gamma_norm
  names(ri) <- paste0("RI_", mid_year)

  writeRaster(
    ri,
    ri_files[i],
    overwrite = overwrite_outputs,
    datatype = "FLT4S",
    NAflag = -9999,
    gdal = c("COMPRESS=LZW")
  )

  if (write_gamma_rasters) {
    writeRaster(
      gamma_norm,
      gamma_norm_files[i],
      overwrite = overwrite_outputs,
      datatype = "FLT4S",
      NAflag = -9999,
      gdal = c("COMPRESS=LZW")
    )
  }

  ri_saved <- rast(ri_files[i])
  ri_mm <- global(ri_saved, fun = c("min", "max", "mean"), na.rm = TRUE)

  if (!all(is.finite(as.numeric(ri_mm[1, ])))) {
    stop(sprintf("RI stats are invalid for year %d.", mid_year))
  }

  # Numerical guard to ensure the computed index remains in [0, 1].
  if (ri_mm[1, "min"] < -1e-6 || ri_mm[1, "max"] > 1 + 1e-6) {
    stop(sprintf("RI outside [0,1] detected for year %d.", mid_year))
  }

  diagnostic_rows[[i]] <- data.frame(
    raw_stats[i, ],
    gamma_min_global = gamma_min_global,
    gamma_max_global = gamma_max_global,
    RI_min = as.numeric(ri_mm[1, "min"]),
    RI_max = as.numeric(ri_mm[1, "max"]),
    RI_mean = as.numeric(ri_mm[1, "mean"]),
    output_file = basename(ri_files[i]),
    normalization_scope = "ALL_PIXELS_ALL_YEARS_1985_2020",
    stringsAsFactors = FALSE
  )

  rm(gamma_raw, gamma_norm, ri, ri_saved)
  gc()
}

diagnostic_table <- do.call(rbind, diagnostic_rows)
write.csv(diagnostic_table, diagnostic_csv, row.names = FALSE,
          fileEncoding = "UTF-8")

# ===================== 11. Final checks & cleanup =====================

missing_outputs <- ri_files[!file.exists(ri_files)]
if (length(missing_outputs) > 0L) {
  stop(paste("Missing RI rasters:", paste(missing_outputs, collapse = "\n")))
}

if (!write_gamma_rasters) {
  # Delete temporary gamma ONLY after both passes and diagnostics succeed.
  unlink(gamma_work_dir, recursive = TRUE)
}

cat("\nAll 36 RI rasters (1985-2020) have been generated.\n")
cat("RI output directory: ", ri_output_dir, "\n")
cat("Global normalization bounds: ", bounds_csv, "\n")
cat("Annual diagnostic log: ", diagnostic_csv, "\n")
if (write_gamma_rasters) {
  cat("Retained raw gamma rasters: ", gamma_work_dir, "\n")
  cat("Retained gamma_norm rasters: ", diagnostic_dir, "\n")
}

cat("\nMethod summary:\n")
cat("1. Monthly NDVI, temperature, precipitation: 1983-2022.\n")
cat("2. Five-year centered windows, one-year step: midpoint 1985-2020.\n")
cat("3. Pixelwise z-score standardization within each 60-month window.\n")
cat("4. AR(1) coefficient gamma estimated separately in each window.\n")
cat("5. ONE GLOBAL min-max pair of gamma over ALL pixels and 36 years.\n")
cat("6. The SAME gamma bounds are applied to every year; RI = 1-gamma_norm.\n")
cat("7. No within-year, within-window, or within-pixel gamma min-max scaling.\n")
