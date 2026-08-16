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
#   Gamma is min-max normalized to [0, 1] within the analysis domain:
#
#     gamma_norm = (gamma - gamma_min) / (gamma_max - gamma_min)
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
# - The grassland mask is applied before coefficient normalization, so the
#   min-max normalization domain is the grassland analysis domain.
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

# Optional diagnostic outputs.
# TRUE writes raw gamma and normalized gamma rasters for each window.
write_gamma_rasters <- FALSE

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

# Min-max normalize a coefficient raster to [0, 1].
minmax_normalize_raster <- function(r) {

  mm <- global(
    r,
    fun = c("min", "max"),
    na.rm = TRUE
  )

  r_min <- mm[1, "min"]
  r_max <- mm[1, "max"]

  if (!is.finite(r_min) || !is.finite(r_max)) {
    return(
      list(
        raster = r * NA_real_,
        min = NA_real_,
        max = NA_real_
      )
    )
  }

  if (abs(r_max - r_min) < .Machine$double.eps) {
    warning(
      "Gamma has no spatial variation in this window; min-max normalization is undefined."
    )

    return(
      list(
        raster = r * NA_real_,
        min = r_min,
        max = r_max
      )
    )
  }

  r_norm <- (r - r_min) / (r_max - r_min)
  r_norm <- clamp(r_norm, lower = 0, upper = 1, values = TRUE)

  list(
    raster = r_norm,
    min = r_min,
    max = r_max
  )
}

# ===================== 8. Five-year moving-window RI calculation =====================

window_start_years <- seq(
  start_year,
  end_year - window_years + 1,
  by = step_years
)

expected_n_windows <- end_year - start_year - window_years + 2

if (length(window_start_years) != expected_n_windows) {
  stop("Unexpected number of moving windows.")
}

cat("\nNumber of 5-year windows:", length(window_start_years), "\n")
cat(
  "RI midpoint years:",
  min(window_start_years + floor(window_years / 2)),
  "to",
  max(window_start_years + floor(window_years / 2)),
  "\n\n"
)

diagnostic_rows <- vector("list", length(window_start_years))

for (i in seq_along(window_start_years)) {

  ws <- window_start_years[i]
  we <- ws + window_years - 1
  mid_year <- ws + floor(window_years / 2)

  cat(
    sprintf(
      "[%02d/%02d] Window %d-%d -> RI %d\n",
      i,
      length(window_start_years),
      ws,
      we,
      mid_year
    )
  )

  window_start_date <- as.Date(sprintf("%d-01-01", ws))
  window_end_date   <- as.Date(sprintf("%d-12-01", we))

  idx <- which(
    all_dates >= window_start_date &
      all_dates <= window_end_date
  )

  if (length(idx) != window_months) {
    stop(
      paste0(
        "Window ", ws, "-", we,
        " contains ", length(idx),
        " months; expected ", window_months, "."
      )
    )
  }

  # Read the 60 monthly layers for each variable.
  ndvi_window <- rast(ndvi_index$file[idx])
  temp_window <- rast(temp_index$file[idx])
  pre_window  <- rast(pre_index$file[idx])

  names(ndvi_window) <- paste0("NDVI_", format(all_dates[idx], "%Y%m"))
  names(temp_window) <- paste0("TEMP_", format(all_dates[idx], "%Y%m"))
  names(pre_window)  <- paste0("PRE_", format(all_dates[idx], "%Y%m"))

  # Restrict calculations to the grassland analysis domain.
  ndvi_window <- mask(ndvi_window, grass_mask)
  temp_window <- mask(temp_window, grass_mask)
  pre_window  <- mask(pre_window, grass_mask)

  model_stack <- c(ndvi_window, temp_window, pre_window)

  coef_rasters <- app(
    model_stack,
    fun = fit_ar1_pixel,
    cores = use_cores
  )

  names(coef_rasters) <- c(
    "alpha",
    "beta",
    "gamma",
    "n_pairs"
  )

  gamma_raw <- coef_rasters[["gamma"]]

  # Spatial min-max normalization of gamma within this 5-year window.
  gamma_norm_result <- minmax_normalize_raster(gamma_raw)

  gamma_norm <- gamma_norm_result$raster
  names(gamma_norm) <- "gamma_norm"

  # RI = 1 - normalized gamma
  ri <- 1 - gamma_norm
  ri <- clamp(ri, lower = 0, upper = 1, values = TRUE)
  names(ri) <- paste0("RI_", mid_year)

  out_ri <- file.path(
    ri_output_dir,
    paste0("RI_", mid_year, ".tif")
  )

  if (file.exists(out_ri) && !overwrite_outputs) {
    stop(
      paste0(
        "Output already exists: ", out_ri, "\n",
        "Set overwrite_outputs <- TRUE only if you intentionally want to replace it."
      )
    )
  }

  writeRaster(
    ri,
    out_ri,
    overwrite = overwrite_outputs,
    datatype = "FLT4S",
    NAflag = -9999,
    gdal = c("COMPRESS=LZW")
  )

  if (write_gamma_rasters) {

    gamma_raw_file <- file.path(
      diagnostic_dir,
      paste0("gamma_raw_", mid_year, ".tif")
    )

    gamma_norm_file <- file.path(
      diagnostic_dir,
      paste0("gamma_norm_", mid_year, ".tif")
    )

    writeRaster(
      gamma_raw,
      gamma_raw_file,
      overwrite = overwrite_outputs,
      datatype = "FLT4S",
      NAflag = -9999,
      gdal = c("COMPRESS=LZW")
    )

    writeRaster(
      gamma_norm,
      gamma_norm_file,
      overwrite = overwrite_outputs,
      datatype = "FLT4S",
      NAflag = -9999,
      gdal = c("COMPRESS=LZW")
    )
  }

  # Diagnostics for reproducibility.
  ri_mm <- global(
    ri,
    fun = c("min", "max", "mean"),
    na.rm = TRUE
  )

  valid_cells <- global(
    ifel(!is.na(ri), 1, NA),
    fun = "sum",
    na.rm = TRUE
  )[1, 1]

  diagnostic_rows[[i]] <- data.frame(
    window_start = ws,
    window_end = we,
    midpoint_year = mid_year,
    n_months = length(idx),
    gamma_min_raw = gamma_norm_result$min,
    gamma_max_raw = gamma_norm_result$max,
    RI_min = ri_mm[1, "min"],
    RI_max = ri_mm[1, "max"],
    RI_mean = ri_mm[1, "mean"],
    valid_cells = valid_cells,
    output_file = basename(out_ri),
    stringsAsFactors = FALSE
  )

  # Release large objects before the next window.
  rm(
    ndvi_window,
    temp_window,
    pre_window,
    model_stack,
    coef_rasters,
    gamma_raw,
    gamma_norm,
    ri
  )

  gc()
}

# ===================== 9. Write calculation log =====================

diagnostic_table <- do.call(
  rbind,
  diagnostic_rows
)

diagnostic_csv <- file.path(
  diagnostic_dir,
  "RI_calculation_log_1985_2020.csv"
)

write.csv(
  diagnostic_table,
  diagnostic_csv,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 10. Final checks =====================

expected_mid_years <- 1985:2020

created_files <- file.path(
  ri_output_dir,
  paste0("RI_", expected_mid_years, ".tif")
)

missing_outputs <- created_files[!file.exists(created_files)]

if (length(missing_outputs) > 0) {
  warning(
    paste0(
      "The following expected RI outputs are missing:\n",
      paste(missing_outputs, collapse = "\n")
    )
  )
} else {
  cat("\nAll expected RI rasters (1985-2020) are present.\n")
}

cat("\nRI calculation completed.\n")
cat("RI output directory:\n", ri_output_dir, "\n")
cat("Diagnostic log:\n", diagnostic_csv, "\n\n")

cat("Method summary:\n")
cat("1. Monthly NDVI, temperature, and precipitation: 1983-2022.\n")
cat("2. Five-year moving window with a one-year step.\n")
cat("3. Pixel-wise z-score standardization within each 60-month window.\n")
cat("4. AR(1): NDVI_t = alpha*Temp_t + beta*Pre_t + gamma*NDVI_(t-1) + epsilon_t.\n")
cat("5. Spatial min-max normalization of gamma within each window.\n")
cat("6. RI = 1 - gamma_norm.\n")
cat("7. Window midpoint year used as RI year: 1985-2020.\n")
