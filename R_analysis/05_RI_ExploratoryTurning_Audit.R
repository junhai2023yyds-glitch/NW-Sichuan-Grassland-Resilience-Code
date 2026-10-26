# =============================================================================
# 05_RI_ExploratoryTurning_Audit.R
# Reviewer 2, Comment 9 (3.5) - EXPLORATORY RI pattern reporting
# Based on the user's original 02_RI_spatial_analysis(2).R and completed 02B.
#
# IMPORTANT
# * DOES NOT re-estimate trends, Pettitt, or breakpoints.
# * DOES NOT assert pixel-level statistically significant turning points.
# * Old p<0.05 / AIC-based classes are DESCRIPTIVE, algorithm-selected patterns.
# * Reuses 02B HAC(4)+BH result as an APPROXIMATE LINEAR-only sensitivity check.
# * 02B is NOT a validated final turning-point test; its non-significance does
#   NOT prove the absence of a turning point.
# * Writes to a NEW run folder; NEVER overwrites the original source outputs.
# =============================================================================

# ========================= 1. CONFIGURATION ==================================
# Run from repository root: Rscript R_analysis/05_RI_ExploratoryTurning_Audit.R
# Previous baseline 02 runs may live outside this repository; set environment
# RI_ORIGINAL_RUN to its Run_timestamp folder (without committing the path).
# Or copy the baseline run into output/RI_spatial/Run_timestamp/.
project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
original_raster_root <- Sys.getenv("RI_SPATIAL_OUT",
  unset = file.path(project_root, "output", "RI_spatial"))
phase2b_root <- Sys.getenv("RI_SENSITIVITY_OUT",
  unset = file.path(project_root, "output", "reviewer2_comment9", "04_sensitivity"))
output_parent <- Sys.getenv("RI_EXPLORATORY_OUT",
  unset = file.path(project_root, "output", "reviewer2_comment9", "05_exploratory_audit"))

# If you know the exact previous Run_ folder, paste it here.
# Leave empty to use the latest matching Run_ folder automatically.
original_run_dir <- Sys.getenv("RI_ORIGINAL_RUN", unset = "")
phase2b_run_dir <- Sys.getenv("RI_SENSITIVITY_RUN", unset = "")

# Paper-compatible integer area reporting (km2), preserving original rules.
area_reporting_unit_km2 <- 1

if (!requireNamespace("terra", quietly = TRUE)) {
  stop("Please install terra: install.packages('terra')")
}
suppressPackageStartupMessages(library(terra))
options(scipen = 999)

latest_run <- function(root, explicitly_selected, required_relative_path) {
  if (nzchar(explicitly_selected)) {
    candidates <- explicitly_selected
  } else {
    if (!dir.exists(root)) stop("Input parent folder missing: ", root)
    candidates <- list.dirs(root, recursive = FALSE, full.names = TRUE)
    candidates <- candidates[grepl("^Run_", basename(candidates))]
    candidates <- sort(candidates, decreasing = TRUE)
  }
  if (!length(candidates)) stop("No Run_* folders under: ", root)
  for (d in candidates) {
    if (file.exists(file.path(d, required_relative_path))) return(d)
  }
  stop("No completed run with expected file: ", required_relative_path,
       "\nPlease set its full directory in the CONFIGURATION section.")
}

old_rel_type <- file.path("01_趋势类型",
  "RI_Trend_Type_B_1985_2020_CommonValid.tif")
old_dir <- latest_run(original_raster_root, original_run_dir, old_rel_type)
new_dir <- latest_run(phase2b_root, phase2b_run_dir,
  "06_Pixel_Linear_Significance_Status.tif")

stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(output_parent, paste0("Run_", stamp))
if (dir.exists(out_dir)) out_dir <- paste0(out_dir, "_", Sys.getpid())
if (!dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)) {
  stop("Cannot create output folder: ", out_dir)
}

cat("Original analysis: ", old_dir, "\n", sep = "")
cat("02B linear sensitivity: ", new_dir, "\n", sep = "")
cat("New exploratory output: ", out_dir, "\n", sep = "")

# ====================== 2. LOAD EXISTING OUTPUTS ============================
load_one <- function(folder, relative_path) {
  p <- file.path(folder, relative_path)
  if (!file.exists(p)) stop("Required input not found: ", p)
  x <- terra::rast(p)
  if (terra::nlyr(x) != 1L) stop("Expected one raster band: ", p)
  x
}

r_type <- load_one(old_dir, old_rel_type)
r_turn_dir <- load_one(old_dir, file.path("02_ID_DI转折类型",
  "RI_Turning_Direction_C_ID_DI_1985_2020_CommonValid.tif"))
r_turn_year <- load_one(old_dir, file.path("03_转折年份",
  "RI_Turning_Year_D_1985_2020_CommonValid.tif"))
r_linear_dir <- load_one(old_dir, file.path("04_最终线性趋势方向",
  "RI_Final_Linear_Direction_1985_2020_CommonValid.tif"))
r_cv <- load_one(old_dir, file.path("06_CV",
  "RI_CV_Class_1985_2020_CommonValid.tif"))
r_original_mask <- load_one(old_dir, file.path("07_质控",
  "RI_Common_Analysis_Mask_1985_2020.tif"))
r_hac_status <- terra::rast(file.path(new_dir,
  "06_Pixel_Linear_Significance_Status.tif"))[[1L]]

for (other in list(r_turn_dir, r_turn_year, r_linear_dir,
                   r_cv, r_original_mask, r_hac_status)) {
  if (!isTRUE(terra::compareGeom(r_type, other, stopOnError = FALSE))) {
    stop("Raster grids do NOT align. Choose matching original and 02B runs; ",
         "do not resample silently.")
  }
}

get_values <- function(x) as.numeric(terra::values(x, mat = FALSE))
v_type <- get_values(r_type)
v_turn_dir <- get_values(r_turn_dir)
v_turn_year <- get_values(r_turn_year)
v_linear_dir <- get_values(r_linear_dir)
v_cv <- get_values(r_cv)
v_mask <- get_values(r_original_mask)
v_hac <- get_values(r_hac_status)

r_cell_area <- terra::cellSize(r_type, unit = "km")
a <- get_values(r_cell_area)
in_common <- is.finite(v_mask)

# Verify the original analysis denominator and all code definitions BEFORE
# exporting anything for the manuscript. No silent interpolation or relabeling.
checks <- data.frame(check = character(), pass = logical(), detail = character())
add_check <- function(name, ok, detail = "") {
  checks <<- rbind(checks, data.frame(check = name,
    pass = isTRUE(ok), detail = as.character(detail)))
}
add_check("Analysis mask = classified cell locations",
  identical(in_common, is.finite(v_type)))
add_check("Trend classes within 0/1/2",
  all(v_type[in_common] %in% c(0, 1, 2)))
add_check("CV classes within 1/2/3",
  all(v_cv[in_common] %in% c(1, 2, 3)))
add_check("Original candidate turns have direction 1/2",
  all(v_turn_dir[in_common & v_type == 2] %in% c(1, 2)))
add_check("Original candidate turns have valid year 1990-2015",
  all(v_turn_year[in_common & v_type == 2] %in% 1990:2015))
add_check("Original selected linear class has direction 1/2",
  all(v_linear_dir[in_common & v_type == 1] %in% c(1, 2)))
add_check("No original turn direction in other classes",
  all(is.na(v_turn_dir[in_common & v_type != 2])))
add_check("No original turning year in other classes",
  all(is.na(v_turn_year[in_common & v_type != 2])))
add_check("02B all original cells present (0,1,2,3)",
  all(v_hac[in_common] %in% 0:3))
add_check("02B no cells outside original common extent",
  all(is.na(v_hac[!in_common])))
add_check("Original cell areas positive/finite",
  all(is.finite(a[in_common]) & a[in_common] > 0))

write_csv <- function(x, filename) utils::write.csv(x,
  file.path(out_dir, filename), row.names = FALSE,
  fileEncoding = "UTF-8")
write_csv(checks, "10_Quality_Control_Checks.csv")
if (!all(checks$pass)) {
  stop("INPUT CHECKS FAILED. Read: ",
       file.path(out_dir, "10_Quality_Control_Checks.csv"),
       "\nDo not use outputs until mismatches are resolved.")
}

area_exact_total <- sum(a[in_common])
area_report_total <- round(area_exact_total / area_reporting_unit_km2) *
  area_reporting_unit_km2
n_valid <- sum(in_common)
cat("Original common extent: ", n_valid, " cells, ",
    format(area_exact_total, digits = 12), " km2\n", sep = "")

# =================== 3. PAPER-COMPATIBLE AREA TABLES =========================
# Original: cellSize for exact area; 1-km2 integer report uses Hamilton's
# largest-remainder allocation, preserving closure within a parent category.
balanced_area <- function(exact, parent_reported, unit = 1) {
  x <- as.numeric(exact)
  if (any(!is.finite(x)) || any(x < 0)) stop("Bad category area")
  scaled <- x / unit
  base <- floor(scaled)
  fractional <- scaled - base
  needed <- as.integer(round(parent_reported / unit) - sum(base))
  if (needed < 0L || needed > length(x)) {
    stop("Integer area closure error: check chosen runs")
  }
  if (needed > 0L) {
    index <- order(-fractional, seq_along(fractional))[seq_len(needed)]
    base[index] <- base[index] + 1L
  }
  result <- base * unit
  if (abs(sum(result) - parent_reported) > 1e-8) {
    stop("Rounded class areas do not sum to parent")
  }
  result
}

area_table <- function(groups, codes, labels, reported_parent,
                       share_parent = NA_real_) {
  if (length(groups) != length(v_type)) stop("Group length mismatch")
  area_exact <- vapply(codes, function(z)
    sum(a[in_common & !is.na(groups) & groups == z]), numeric(1))
  counts <- vapply(codes, function(z)
    sum(in_common & !is.na(groups) & groups == z), integer(1))
  reported <- balanced_area(area_exact, reported_parent,
                            area_reporting_unit_km2)
  data.frame(code = codes, label = labels, pixel_count = counts,
    area_km2_exact = area_exact,
    area_km2_report_1km2 = reported,
    pct_common_area = 100 * reported / area_report_total,
    pct_within_parent = if (is.finite(share_parent) && share_parent > 0)
      100 * reported / share_parent else NA_real_,
    stringsAsFactors = FALSE)
}

trend_tbl <- area_table(v_type, 0:2, c(
  "No pattern selected by original uncorrected rules",
  "Exploratory linear pattern (original rules)",
  "Exploratory candidate turning pattern (original rules)"
), reported_parent = area_report_total)
write_csv(trend_tbl, "01_Exploratory_Trend_Classes_Area.csv")

turning_report <- trend_tbl$area_km2_report_1km2[trend_tbl$code == 2]
linear_report <- trend_tbl$area_km2_report_1km2[trend_tbl$code == 1]

turning_dirs <- area_table(v_turn_dir, 1:2, c(
  "Candidate increase-to-decrease (I-D)",
  "Candidate decrease-to-increase (D-I)"
), reported_parent = turning_report, share_parent = turning_report)
write_csv(turning_dirs, "02_Exploratory_Turning_Directions_Area.csv")

# Original paper uses 1990-1995 inclusive (six years), and then 5-year
# right-closed bins; boundary years are not counted twice.
period_breaks <- c(1989, 1995, 2000, 2005, 2010, 2015)
period_labels <- c("1990-1995", "1995-2000", "2000-2005",
                   "2005-2010", "2010-2015")
period_codes <- rep(NA_integer_, length(v_type))
idx_turn <- which(in_common & v_type == 2)
if (length(idx_turn)) {
  period_codes[idx_turn] <- as.integer(cut(v_turn_year[idx_turn],
    breaks = period_breaks, include.lowest = TRUE, right = TRUE))
}
period_tbl <- area_table(period_codes, 1:5,
  paste0("Exploratory turning period ", period_labels),
  reported_parent = turning_report, share_parent = turning_report)
period_tbl$display_label <- period_labels
period_tbl$actual_included_years <- c("1990-1995", "1996-2000",
  "2001-2005", "2006-2010", "2011-2015")
write_csv(period_tbl, "03_Exploratory_Turning_Periods_Area.csv")

linear_tbl <- area_table(v_linear_dir, 1:2, c(
  "Exploratory original-rule linear increase",
  "Exploratory original-rule linear decrease"
), reported_parent = linear_report, share_parent = linear_report)
write_csv(linear_tbl, "04_Exploratory_Linear_Directions_Area.csv")

cv_tbl <- area_table(v_cv, 1:3,
  c("CV <20%", "CV 20-30%", "CV >30%"),
  reported_parent = area_report_total)
write_csv(cv_tbl, "05_Original_CV_Unchanged_Area.csv")

# ================ 4. REUSE 02B LINEAR SENSITIVITY ===========================
# A HAC slope test is not a turning point test. This cross-tab is diagnostic
# only: do NOT merge 02B status into the original turning classification.
cross <- expand.grid(original_code = 0:2, hac_status = 0:3)
cross$original_label <- trend_tbl$label[match(cross$original_code,
                                             trend_tbl$code)]
cross$hac_label <- c(
  "HAC4+BH: linear slope not selected (NOT no change)",
  "HAC4+BH: positive linear slope selected",
  "HAC4+BH: negative linear slope selected",
  "HAC4+BH: not formally testable"
)[cross$hac_status + 1L]
cross$pixel_count <- mapply(function(i, j)
  sum(in_common & v_type == i & v_hac == j),
  cross$original_code, cross$hac_status)
cross$area_km2_exact <- mapply(function(i, j)
  sum(a[in_common & v_type == i & v_hac == j]),
  cross$original_code, cross$hac_status)
cross$pct_common_area_exact <- 100 * cross$area_km2_exact / area_exact_total
cross$pct_within_original_class_exact <- mapply(function(i, z) {
  denom <- sum(a[in_common & v_type == i])
  if (denom > 0) 100 * z / denom else NA_real_
}, cross$original_code, cross$area_km2_exact)
write_csv(cross, "06_Exploratory_Class_vs_02B_HAC4_BH_Crosstab.csv")

linear_comparison_file <- file.path(new_dir,
  "07_Linear_Significance_Comparison_Area.csv")
regional_test_file <- file.path(new_dir,
  "03_Regional_OLS_and_HAC_L1to6.csv")
if (!file.exists(linear_comparison_file) || !file.exists(regional_test_file)) {
  stop("02B tabular output missing. Select a completed 02B Run folder.")
}
linear_sensitivity <- utils::read.csv(linear_comparison_file,
  fileEncoding = "UTF-8", check.names = FALSE)
regional_sensitivity <- utils::read.csv(regional_test_file,
  fileEncoding = "UTF-8", check.names = FALSE)
write_csv(linear_sensitivity, "07_02B_Linear_HAC_FDR_Sensitivity_REUSED.csv")
write_csv(regional_sensitivity, "08_Regional_OLS_HAC_REUSED.csv")

# ===================== 5. MANUSCRIPT VALUE AUDIT ============================
# Reference values quoted from manuscript as of 2026-10-26.
# This is an AUDIT ONLY, never a silent rewrite of journal text.
actual <- c(
  pct_no_pattern = trend_tbl$pct_common_area[1L],
  pct_linear_pattern = trend_tbl$pct_common_area[2L],
  pct_turning_pattern = trend_tbl$pct_common_area[3L],
  pct_linear_plus_turning = sum(trend_tbl$pct_common_area[2:3]),
  pct_I_D_within_turning = turning_dirs$pct_within_parent[1L],
  pct_D_I_within_turning = turning_dirs$pct_within_parent[2L],
  pct_1990_1995_within_turning = period_tbl$pct_within_parent[1L]
)
previous <- c(45.99, 34.70, 19.31, 54.01, 75.56, 24.44, 46.85)
number_audit <- data.frame(
  metric = names(actual),
  original_manuscript_percent = previous,
  exploratory_original_rule_percent = round(actual, 6),
  difference_percentage_points = round(actual - previous, 6),
  within_0_03_percentage_points = abs(actual - previous) <= 0.03,
  stringsAsFactors = FALSE
)
write_csv(number_audit, "09_Original_Manuscript_Number_Audit.csv")
if (any(!number_audit$within_0_03_percentage_points)) {
  warning("One or more old manuscript percentages differ from the selected run. ",
          "Check file 09 and select the correct original Run directory.")
}

# =================== 6. FINAL CHECKS AND OPTIONAL EXPORT ===================
checks_final <- data.frame(
  check = c("Original categories close to total area",
            "Candidate turning directions close to turning area",
            "Candidate turning years close to turning area",
            "Selected linear directions close to linear area",
            "CV categories close to total area",
            "Cross-tab closes to total area",
            "Cross-tab closes to original cell count"),
  pass = c(
    abs(sum(trend_tbl$area_km2_exact) - area_exact_total) < 1e-5,
    abs(sum(turning_dirs$area_km2_exact) -
        sum(a[in_common & v_type == 2])) < 1e-5,
    abs(sum(period_tbl$area_km2_exact) -
        sum(a[in_common & v_type == 2])) < 1e-5,
    abs(sum(linear_tbl$area_km2_exact) -
        sum(a[in_common & v_type == 1])) < 1e-5,
    abs(sum(cv_tbl$area_km2_exact) - area_exact_total) < 1e-5,
    abs(sum(cross$area_km2_exact) - area_exact_total) < 1e-5,
    sum(cross$pixel_count) == n_valid
  ),
  detail = ""
)
write_csv(rbind(checks, checks_final), "10_Quality_Control_Checks.csv")
if (!all(checks_final$pass)) stop("Area closure error. Check QA table.")

# Only re-export existing categorical data with explicit exploratory names.
# These are exact value copies, NOT new statistical inference or final maps.
export_copy <- function(x, filename, datatype, naflag) {
  terra::writeRaster(x, file.path(out_dir, filename), overwrite = FALSE,
    datatype = datatype, NAflag = naflag, gdal = c("COMPRESS=LZW"))
}
export_copy(r_type, "12_Exploratory_OriginalRule_TrendType.tif",
            "INT1U", 255)
export_copy(r_turn_dir, "13_Exploratory_OriginalRule_TurningDirection.tif",
            "INT1U", 255)
export_copy(r_turn_year, "14_Exploratory_OriginalRule_TurningYear.tif",
            "INT2S", -9999)

metadata <- data.frame(item = c(
  "Script", "Purpose", "Original_Run_Dir", "02B_Run_Dir", "Output_Dir",
  "Common_Analysis_Pixels", "Common_Analysis_Area_km2_Exact",
  "Common_Analysis_Area_km2_Reported", "Area_Rounding_km2",
  "Candidate_Turning_Pixels", "Candidate_Turning_Area_km2_Reported",
  "Original_Rule_Significance_Validated", "Turning_Serial_Dependence_FDR_Completed",
  "02B_HAC4_BH_Used_For", "Original_Trend_Reestimated",
  "Warning"
), value = c(
  "05_RI_ExploratoryTurning_Audit.R",
  "Audit/relabel original patterns as exploratory; do not re-fit RI trends",
  "original spatial Run_* (path omitted)", "04_sensitivity/Run_* (path omitted)",
  "05_exploratory_audit/Run_* (path omitted)", n_valid,
  format(area_exact_total, digits = 14), area_report_total,
  area_reporting_unit_km2, length(idx_turn), turning_report,
  "NO", "NO", "LINEAR-only sensitivity / cross-tab; NOT turning confirmation",
  "NO", "Do not claim original turning areas are FDR-confirmed"
), stringsAsFactors = FALSE)
write_csv(metadata, "11_Run_Metadata.csv")

readme <- c(
  "Reviewer 2, Comment 9 -- EXPLORATORY interpretation and area audit",
  "",
  "Purpose: re-report ORIGINAL p<0.05/AIC-selected classifications as",
  "exploratory patterns. These are the UNCORRECTED original-rule classifications.",
  "The script performs NO new turning-point significance test and NO new FDR.",
  "The original three raster copies contain unchanged original numeric codes:",
  "12: 0=No pattern selected by original rules; 1=exploratory linear pattern;",
  "    2=exploratory candidate turning pattern.",
  "13: 1=candidate increase-to-decrease; 2=candidate decrease-to-increase.",
  "14: candidate turning year, NOT a formally confirmed break date.",
  "",
  "Files 01-05: original classification/area audit, with exact area and",
  "manuscript-consistent 1-km2 integer reporting using largest remainders.",
  "File 06: original classes vs 04 HAC(4)+BH LINEAR slope status.",
  "Files 07-08: 04 LINEAR sensitivity and regional HAC tables, reused.",
  "File 09: compares re-reported original-rule percentages with original paper.",
  "File 10: geometry and area-closure checks.",
  "File 11: provenance and statistical warning.",
  "",
  "IMPORTANT LIMITATIONS:",
  "* The old 'No trend' category should be renamed 'No original-rule pattern",
  "  selected'. Absence of an original-rule pattern does not prove no change.",
  "* Old linear/turning categories were screened using uncorrected p-values.",
  "  Their shares must NOT be described as FDR-confirmed significant areas.",
  "* Five-year-centered RI windows overlap, creating serial dependence.",
  "* The 02B HAC/linear FDR is an APPROXIMATE sensitivity check only. Previous",
  "  null-model tests found that several procedures do not reliably control",
  "  type-I error across all simulated dependence structures.",
  "* HAC of a single linear slope is NOT a test for a slope reversal.",
  "* First time bin 1990-1995 has SIX calendar years; later bins have five.",
  "* Changing captions/narrative to exploratory DOES NOT satisfy the reviewer's",
  "  request for statistically corrected pixel-level turning-point inference.",
  "  The response letter must disclose what was and was not validated.",
  "* Keep historical original percentages visible only with a clear uncorrected",
  "  exploratory qualifier. Do NOT claim confirmed ecological thresholds or breaks.",
  "* Regionwide trend p-values were copied from the existing 02B HAC analysis.",
  "",
  "No source files were overwritten. No graphics were generated."
)
writeLines(readme, file.path(out_dir, "README_Exploratory_Reporting.txt"),
           useBytes = TRUE)

cat("\nExploratory audit completed.\n")
cat("Output directory: ", out_dir, "\n", sep = "")
cat("Original reported area (km2): ", area_report_total, "\n", sep = "")
cat("Review the number audit and README before revising any paper text.\n")
