# Standalone R script; no configuration or helper script is required.
# Dependency: install.packages("terra")
# Edit ri_dir, grass_shp and report_root below before running in RStudio.
# Relative paths are resolved against getwd(), not the script location.
# The default baseline values apply to the original 1985-2020 study only.
# Input rasters and the complete shapefile sidecar set are not bundled.
# ================================================================
# Diversity Reviewer 2 - Comment 12 (original 3.8)
# Boundary-rule sensitivity + spatially synchronized AR(1) null model
# Based on the published 02_RI_spatial_analysis.R analysis logic.
# Authoring date: 2026-10-24
#
# Does NOT recompute the RI from NDVI. Reads existing annual RI GeoTIFFs.
# Does NOT overwrite existing tables/maps. Uses a separate result folder.
# Main analysis: 3-, 5-, 7-year endpoint restrictions, all valid pixels.
# Null simulation: complete-case, approximately 8-km spatially stratified
# sample; fitted linear trend + AR(1) residual process; spatially
# synchronized resampling of observed innovation fields (exploratory).
# Note: simulations do not replace formal serial-dependence/FDR checks.
# ================================================================

# -------------------- USER CONFIGURATION -------------------------
ri_dir <- "data/ri"  # Set to the folder containing the 36 annual RI GeoTIFFs.
# Original shapefile used for the manuscript's grassland mask:
grass_shp <- "data/mask/grassland.shp"  # Set to the original study grassland mask.
report_root <- "results"  # Set to the output parent directory.
run_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(report_root, "10_Comments12_边界规则敏感性检验",
                     paste0("Run_", run_tag))

start_year <- 1985L
end_year <- 2020L
boundary_gaps <- c(5L, 3L, 7L) # reproduce the original (5) first
alpha <- 0.05
aic_delta <- 2
min_valid_years <- 10L
cv_mean_eps <- 1e-12

# Null-model settings. Set run_null_model <- FALSE for a rapid first run.
run_null_model <- TRUE
null_repetitions <- 199L   # increase to 499 for final analyses if feasible
null_max_tiles <- 350L     # sample at most this many spatial blocks
null_block_cells <- 8L    # approx. 8x8 of the 1-km analysis cells
random_seed <- 20261024L

# Baseline quality-control tolerance; stop if the data are inconsistent.
expected_area_km2 <- 142244
expected_abrupt_km2 <- 27464
expected_early_km2 <- 12867
expected_early_share_pct <- 46.85
max_area_relative_error <- 0.015  # 1.5% for area metrics
max_share_absolute_error <- 0.60  # percentage points
# ---------------------------------------------------------------

if (!requireNamespace("terra", quietly = TRUE)) {
  stop("缺少 terra 包。请先执行 install.packages('terra')")
}
options(scipen = 999)
stopifnot(start_year < end_year, min_valid_years >= 10L)
if (!dir.exists(ri_dir)) stop("RI 输入目录不存在: ", ri_dir)
if (!file.exists(grass_shp)) {
  stop("草地掩膜矢量不存在: ", grass_shp,
       "\n请将 grass_shp 修改为与原始论文分析一致的真实路径。")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tmp_dir <- file.path(out_dir, "_terra_tmp")
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
terra::terraOptions(tempdir = tmp_dir, memfrac = 0.65, progress = 1)
log_file <- file.path(out_dir, "运行日志.txt")
cat("Comments12 开始: ", as.character(Sys.time()), "\n", file = log_file)
log_msg <- function(...) {
  s <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ",
              paste0(..., collapse = ""))
  message(s)
  cat(s, "\n", file = log_file, append = TRUE)
}
write_csv <- function(x, filename) {
  utils::write.csv(x, file.path(out_dir, filename), row.names = FALSE,
                   fileEncoding = "UTF-8")
}

# -------------------- 1. READ AND VALIDATE RI -------------------
# The naming convention matches the original public R script.
pattern <- "^RI([12][0-9]{3})_n_VG\\.(tif|tiff)$"
files <- list.files(ri_dir, pattern = pattern, full.names = TRUE, ignore.case = TRUE)
if (length(files) == 0L) {
  listed <- head(list.files(ri_dir, pattern = "\\.tif(f)?$", ignore.case = TRUE), 12L)
  stop("没有匹配 RI1985_n_VG.tif 等格式的年度影像。\n现有影像名称示例: ",
       paste(listed, collapse = "; "),
       "\n如实际文件名不同，请修改脚本顶部 pattern 和年份提取规则。")
}
file_year <- as.integer(sub(pattern, "\\1", basename(files), ignore.case = TRUE))
stopifnot(!anyNA(file_year), !anyDuplicated(file_year))
ord <- order(file_year)
files <- files[ord]
file_year <- file_year[ord]
need_year <- start_year:end_year
if (!identical(file_year, as.integer(need_year))) {
  stop("需要全部36期1985-2020 RI影像。缺失: ",
       paste(setdiff(need_year, file_year), collapse = ", "),
       "；额外: ", paste(setdiff(file_year, need_year), collapse = ", "))
}
log_msg("输入识别成功，共 ", length(files), " 期RI影像。")
ri_raw <- terra::rast(files)
names(ri_raw) <- paste0("RI", file_year)
mask_vect <- terra::vect(grass_shp)
if (!terra::same.crs(ri_raw, mask_vect)) {
  mask_vect <- terra::project(mask_vect, terra::crs(ri_raw))
}
ri_grass <- terra::mask(terra::crop(ri_raw, mask_vect), mask_vect)
rm(ri_raw)
gc(verbose = FALSE)

# Same common-valid-area rule as original source code.
valid_n <- terra::app(ri_grass, fun = function(z) sum(is.finite(z)))
valid_mean <- terra::app(ri_grass, fun = function(z) {
  z <- z[is.finite(z)]
  if (length(z) == 0L) NA_real_ else mean(z)
})
valid_sd <- terra::app(ri_grass, fun = function(z) {
  z <- z[is.finite(z)]
  if (length(z) < 2L) NA_real_ else stats::sd(z)
})
common_ok <- valid_n >= min_valid_years &
  !is.na(valid_mean) & !is.na(valid_sd) & abs(valid_mean) > cv_mean_eps
common_mask <- terra::ifel(common_ok, 1, NA)
ri <- terra::mask(ri_grass, common_mask)
area_r <- terra::mask(terra::cellSize(ri_grass[[1]], unit = "km"), common_mask)
full_area_km2 <- as.numeric(terra::global(area_r, "sum", na.rm = TRUE)[1, 1])
log_msg("统一有效分析范围: ", round(full_area_km2, 3), " km²")
if (!is.finite(full_area_km2) || full_area_km2 <= 0) {
  stop("有效面积异常；请检查草地掩膜、RI有效值和影像地理信息。")
}
log_msg("输入RI目录: ", ri_dir)
log_msg("本次结果目录: ", out_dir)
if (abs(full_area_km2 / expected_area_km2 - 1) > max_area_relative_error) {
  stop("有效面积预检失败：请先核对RI输入目录和原论文分析范围。",
       " 实际面积=", full_area_km2, "；预期面积=", expected_area_km2)
}
log_msg("有效面积预检通过。")

# -------------------- 2. REPRODUCE ORIGINAL DETECTOR -----------
# Equivalent to the three-way model selection in the public source:
# (1) OLS trend; (2) segmented OLS with best breakpoint by AIC;
# (3) Pettitt test, with AIC of the corresponding two-level step fit.
# Only a qualifying significant I-D/D-I transition is classified abrupt.
# Output: trend_class 0/1/2, turning_year, selected_model 0/1/2/3.
calc_aic <- function(rss, n, k) {
  if (!is.finite(rss) || rss <= 0 || n <= k) return(NA_real_)
  n * log(rss / n) + 2 * k
}
fit_lm_fast <- function(X, y) {
  n <- length(y)
  k <- ncol(X)
  bad <- list(coef = rep(NA_real_, k), p = rep(NA_real_, k), aic = NA_real_)
  if (n <= k + 1) return(bad)
  fm <- try(stats::lm.fit(x = X, y = y), silent = TRUE)
  if (inherits(fm, "try-error")) return(bad)
  rss <- sum(fm$residuals^2)
  df <- n - k
  out <- list(coef = fm$coefficients, p = rep(NA_real_, k),
              aic = calc_aic(rss, n, k))
  if (df <= 0 || !is.finite(rss) || rss <= 0 || fm$rank < k) return(out)
  rr <- try(qr.R(fm$qr), silent = TRUE)
  if (inherits(rr, "try-error")) return(out)
  xx_inv <- try(chol2inv(rr), silent = TRUE)
  if (inherits(xx_inv, "try-error")) return(out)
  se <- sqrt(diag(xx_inv) * rss / df)
  if (any(!is.finite(se) | se <= 0)) return(out)
  t_stat <- out$coef / se
  out$p <- 2 * stats::pt(abs(t_stat), df = df, lower.tail = FALSE)
  out
}
pettitt_original_equivalent <- function(y) {
  n <- length(y)
  if (n < 8L) return(list(p = NA_real_, index = NA_integer_))
  # Algebraically equivalent to sum_{i<=t,j>t} sign(y_i-y_j).
  rnk <- rank(y, ties.method = "average")
  ind <- seq_len(n - 1L)
  U <- 2 * cumsum(rnk)[ind] - ind * (n + 1)
  k <- which.max(abs(U))
  K <- abs(U[k])
  p <- min(1, 2 * exp(-6 * K^2 / (n^3 + n^2)))
  list(p = p, index = k)
}
turn_direction <- function(y, yy, bp) {
  left <- which(yy <= bp)
  right <- which(yy > bp)
  if (length(left) < 4L || length(right) < 4L) return(NA_real_)
  fit1 <- fit_lm_fast(cbind(1, yy[left] - min(yy[left])), y[left])
  fit2 <- fit_lm_fast(cbind(1, yy[right] - min(yy[right])), y[right])
  sl1 <- fit1$coef[2]; sl2 <- fit2$coef[2]
  p1 <- fit1$p[2]; p2 <- fit2$p[2]
  if (!all(is.finite(c(sl1, sl2, p1, p2)))) return(NA_real_)
  if (p1 < alpha && p2 < alpha && sl1 > 0 && sl2 < 0) return(1)
  if (p1 < alpha && p2 < alpha && sl1 < 0 && sl2 > 0) return(2)
  NA_real_
}
detect_one <- function(x, all_years, min_gap) {
  ok <- is.finite(x)
  y <- x[ok]
  yy <- all_years[ok]
  if (length(y) < min_valid_years) return(c(NA_real_, NA_real_, NA_real_))
  if (stats::sd(y) < 1e-12) return(c(0, NA_real_, 0))
  tt <- yy - min(yy)
  linear <- fit_lm_fast(cbind(1, tt), y)
  linear_sig <- is.finite(linear$p[2]) && linear$p[2] < alpha

  # Select the lowest-AIC breakpoint first, as done in original R code.
  cand <- unique(yy[yy >= start_year + min_gap & yy <= end_year - min_gap])
  seg_aic <- NA_real_; seg_p <- NA_real_; seg_yr <- NA_real_
  for (bp in cand) {
    if (sum(yy <= bp) < 4L || sum(yy > bp) < 4L) next
    hinge <- pmax(0, tt - (bp - min(yy)))
    fm <- fit_lm_fast(cbind(1, tt, hinge), y)
    if (!is.finite(fm$aic)) next
    if (!is.finite(seg_aic) || fm$aic < seg_aic) {
      seg_aic <- fm$aic
      seg_p <- fm$p[3]
      seg_yr <- bp
    }
  }
  seg_valid <- is.finite(seg_p) && seg_p < alpha &&
    is.finite(turn_direction(y, yy, seg_yr))

  pet <- pettitt_original_equivalent(y)
  pet_valid <- FALSE
  pet_aic <- NA_real_; pet_yr <- NA_real_
  if (is.finite(pet$p) && is.finite(pet$index)) {
    pet_yr <- yy[pet$index]
    inside <- pet_yr >= start_year + min_gap && pet_yr <= end_year - min_gap
    if (inside) {
      fm_pet <- fit_lm_fast(cbind(1, as.numeric(yy > pet_yr)), y)
      pet_aic <- fm_pet$aic
      pet_valid <- pet$p < alpha && is.finite(turn_direction(y, yy, pet_yr))
    }
  }

  model_names <- character()
  model_aics <- numeric()
  model_ranks <- numeric()
  if (linear_sig && is.finite(linear$aic)) {
    model_names <- c(model_names, "linear")
    model_aics <- c(model_aics, linear$aic)
    model_ranks <- c(model_ranks, 1)
  }
  if (pet_valid && is.finite(pet_aic)) {
    model_names <- c(model_names, "pettitt")
    model_aics <- c(model_aics, pet_aic)
    model_ranks <- c(model_ranks, 2)
  }
  if (seg_valid && is.finite(seg_aic)) {
    model_names <- c(model_names, "segmented")
    model_aics <- c(model_aics, seg_aic)
    model_ranks <- c(model_ranks, 3)
  }
  if (length(model_names) == 0L) return(c(0, NA_real_, 0))
  within2 <- which(model_aics <= min(model_aics) + aic_delta)
  selected <- model_names[within2[which.min(model_ranks[within2])]]
  if (selected == "linear") return(c(1, NA_real_, 1))
  if (selected == "pettitt") return(c(2, pet_yr, 2))
  c(2, seg_yr, 3)
}

# -------------------- 3. SPATIAL FULL-AREA SENSITIVITY -----------
ras_by_gap <- list()
value_by_gap <- list()
area_values <- as.numeric(terra::values(area_r, mat = FALSE))
for (gap in boundary_gaps) {
  log_msg("开始运行边界敏感性：min_gap = ", gap,
          "（候选年份 ", start_year + gap, "–", end_year - gap, "）")
  output_tif <- file.path(out_dir, sprintf("MinGap_%d_Trend_Year_Model.tif", gap))
  fun_local <- function(z) detect_one(z, file_year, gap)
  result_r <- terra::app(ri, fun = fun_local, filename = output_tif,
                         overwrite = TRUE, cores = 1)
  names(result_r) <- c("trend_class", "turning_year", "selected_model")
  ras_by_gap[[as.character(gap)]] <- result_r
  value_by_gap[[as.character(gap)]] <- as.matrix(terra::values(result_r, mat = TRUE))
  log_msg("已写入: ", output_tif)
  # Validate the baseline immediately, before spending time on other gaps.
  if (gap == 5L) {
    baseline_values <- value_by_gap[["5"]]
    baseline_abrupt <- baseline_values[, 1] == 2 &
      is.finite(baseline_values[, 2])
    baseline_area <- sum(area_values[which(baseline_abrupt)], na.rm = TRUE)
    baseline_early <- sum(area_values[which(baseline_abrupt &
      baseline_values[, 2] >= 1990 & baseline_values[, 2] <= 1995)], na.rm = TRUE)
    baseline_share <- if (baseline_area > 0) 100 * baseline_early / baseline_area else NA_real_
    baseline_qc <- data.frame(
      Metric = c("Valid area, km2", "Abrupt area, km2",
                 "1990-1995 turning area, km2", "1990-1995 share, %"),
      Expected = c(expected_area_km2, expected_abrupt_km2,
                   expected_early_km2, expected_early_share_pct),
      Recomputed = c(full_area_km2, baseline_area, baseline_early, baseline_share)
    )
    baseline_qc$Difference <- baseline_qc$Recomputed - baseline_qc$Expected
    baseline_qc$Relative_error_pct <- 100 * abs(baseline_qc$Difference) / baseline_qc$Expected
    write_csv(baseline_qc, "00_原始结果复现核查.csv")
    if (any(!is.finite(baseline_qc$Recomputed)) ||
        any(baseline_qc$Relative_error_pct[1:3] > 100 * max_area_relative_error) ||
        abs(baseline_qc$Difference[4]) > max_share_absolute_error) {
      stop("5年规则复现核查失败；详见00_原始结果复现核查.csv。")
    }
    log_msg("5年规则复现核查通过，继续其他边界规则。")
  }
}

# Study-period groups are non-overlapping in actual calendar years.
periods <- data.frame(
  period = c("1990-1995", "1996-2000", "2001-2005", "2006-2010", "2011-2015"),
  from = c(1990, 1996, 2001, 2006, 2011),
  to   = c(1995, 2000, 2005, 2010, 2015),
  stringsAsFactors = FALSE
)
periods$N_years <- periods$to - periods$from + 1L
sum_area <- function(flag) {
  flag[is.na(flag)] <- FALSE
  sum(area_values[flag & is.finite(area_values)], na.rm = TRUE)
}
all_yearly <- list()
all_period <- list()
all_summary <- list()
for (gap in boundary_gaps) {
  v <- value_by_gap[[as.character(gap)]]
  cls <- v[, 1]
  yr <- v[, 2]
  abrupt <- cls == 2 & is.finite(yr)
  total_abrupt <- sum_area(abrupt)
  early <- sum_area(abrupt & yr >= 1990 & yr <= 1995)
  common <- sum_area(abrupt & yr >= 1992 & yr <= 2013)
  common_early <- sum_area(abrupt & yr >= 1992 & yr <= 1995)
  first_6 <- sum_area(abrupt & yr >= start_year + gap & yr <= start_year + gap + 5L)
  yr_table <- data.frame(
    Edge_gap = gap,
    Year = need_year,
    Area_km2_exact = vapply(need_year, function(z) sum_area(abrupt & yr == z), numeric(1))
  )
  yr_table$Share_in_all_turning_pct <- if (total_abrupt > 0) {
    100 * yr_table$Area_km2_exact / total_abrupt
  } else NA_real_
  all_yearly[[as.character(gap)]] <- yr_table

  ptab <- periods
  ptab$Edge_gap <- gap
  ptab$Area_km2_exact <- mapply(function(a, b) sum_area(abrupt & yr >= a & yr <= b),
                                ptab$from, ptab$to)
  ptab$Annualized_km2_per_calendar_year <- ptab$Area_km2_exact / ptab$N_years
  ptab$Share_in_all_turning_pct <- if (total_abrupt > 0) {
    100 * ptab$Area_km2_exact / total_abrupt
  } else NA_real_
  all_period[[as.character(gap)]] <- ptab

  all_summary[[as.character(gap)]] <- data.frame(
    Edge_gap = gap,
    Eligible_first_year = start_year + gap,
    Eligible_last_year = end_year - gap,
    Valid_study_area_km2 = full_area_km2,
    Total_abrupt_area_km2 = total_abrupt,
    Turning_area_1990_1995_km2 = early,
    Share_1990_1995_pct = if (total_abrupt > 0) 100 * early / total_abrupt else NA_real_,
    Area_1992_2013_km2 = common,
    Area_1992_1995_km2 = common_early,
    Share_1992_1995_within_1992_2013_pct = if (common > 0) {
      100 * common_early / common
    } else NA_real_,
    Share_first_six_eligible_years_pct = if (total_abrupt > 0) {
      100 * first_6 / total_abrupt
    } else NA_real_
  )
}
yearly <- do.call(rbind, all_yearly)
period_table <- do.call(rbind, all_period)
summary_table <- do.call(rbind, all_summary)
write_csv(yearly, "S12_A_逐年转折面积_3_5_7年边界.csv")
write_csv(period_table, "S12_B_分时段面积_含年度平均.csv")
write_csv(summary_table, "S12_C_边界敏感性汇总.csv")

# Manuscript's reporting uses integer square kilometres; small rounding
# differences are possible, because the original uses Hamilton allocation.
ref <- summary_table[summary_table$Edge_gap == 5L, , drop = FALSE]
qc <- data.frame(
  Metric = c("Valid area, km2", "Abrupt area, km2",
             "1990-1995 turning area, km2", "1990-1995 share, %"),
  Expected = c(expected_area_km2, expected_abrupt_km2,
               expected_early_km2, expected_early_share_pct),
  Recomputed = c(ref$Valid_study_area_km2,
                 ref$Total_abrupt_area_km2,
                 ref$Turning_area_1990_1995_km2,
                 ref$Share_1990_1995_pct)
)
qc$Difference <- qc$Recomputed - qc$Expected
qc$Relative_error_pct <- 100 * abs(qc$Difference) / qc$Expected
write_csv(qc, "00_原始结果复现核查.csv")
if (any(qc$Relative_error_pct[1:3] > 100 * max_area_relative_error) ||
    abs(qc$Difference[4]) > max_share_absolute_error) {
  stop("原始5年规则未能复现论文的主要面积统计！\n",
       "请先检查年度TIFF、草地矢量、有效区域掩膜、统计精度。",
       "已输出: 00_原始结果复现核查.csv；不应引用其余结果。")
}
log_msg("原始5年规则复现核查通过（允许的小差异来自面积取整）。")

# Comparison against the same pixels under the original 5-year rule.
base <- value_by_gap[["5"]]
base_abrupt <- base[, 1] == 2 & is.finite(base[, 2])
base_turn_area <- sum_area(base_abrupt)
compare <- lapply(boundary_gaps, function(gap) {
  v <- value_by_gap[[as.character(gap)]]
  new_abrupt <- v[, 1] == 2 & is.finite(v[, 2])
  both <- base_abrupt & new_abrupt
  both_area <- sum_area(both)
  data.frame(
    Edge_gap = gap,
    Baseline_abrupt_retained_pct = if (base_turn_area > 0) {
      100 * both_area / base_turn_area
    } else NA_real_,
    Same_turning_year_among_shared_abrupt_pct = if (both_area > 0) {
      100 * sum_area(both & base[, 2] == v[, 2]) / both_area
    } else NA_real_,
    Within_2_years_among_shared_abrupt_pct = if (both_area > 0) {
      100 * sum_area(both & abs(base[, 2] - v[, 2]) <= 2) / both_area
    } else NA_real_
  )
})
write_csv(do.call(rbind, compare), "S12_D_与原始5年规则的像元一致性.csv")

# -------------------- 4. EXPLORATORY SPATIALLY SYNCHRONIZED NULL -
# Complete time series are required for the spatially synchronized draws.
# Within approximate 8 x 8 1-km blocks, choose one area-weighted RI pixel.
# This reduces NDVI oversampling and is a MONTE-CARLO BENCHMARK, not an
# unconditional proof of statistical significance for the whole landscape.
if (run_null_model) {
  log_msg("开始零假设模拟：空间分块样本 + 线性趋势/AR(1)残差")
  set.seed(random_seed)
  full_mask_r <- terra::app(ri, fun = function(z) as.integer(all(is.finite(z))))
  full_cells <- which(as.numeric(terra::values(full_mask_r, mat = FALSE)) == 1 &
                        is.finite(area_values) & area_values > 0)
  if (length(full_cells) < 150L) {
    stop("完整36年记录不足150个像元，当前零假设样本设计不适用。",
         "可先使用边界敏感性结果，并单独讨论缺测处理。")
  }
  nc <- terra::ncol(ri)
  cell_row <- ((full_cells - 1L) %/% nc) + 1L
  cell_col <- ((full_cells - 1L) %% nc) + 1L
  tiles <- paste((cell_row - 1L) %/% null_block_cells,
                 (cell_col - 1L) %/% null_block_cells, sep = "_")
  by_tile <- split(seq_along(full_cells), tiles)
  if (length(by_tile) < 30L) {
    stop("完整观测的8x8像元块少于30个，无法进行有意义的空间抽样。")
  }
  chosen_ids <- sample(seq_along(by_tile),
                       min(as.integer(null_max_tiles), length(by_tile)),
                       replace = FALSE)
  picked_cells <- integer(length(chosen_ids))
  picked_tile_weights <- numeric(length(chosen_ids))
  picked_tile_names <- character(length(chosen_ids))
  for (j in seq_along(chosen_ids)) {
    inds <- by_tile[[chosen_ids[j]]]
    cells <- full_cells[inds]
    ws <- area_values[cells]
    picked_cells[j] <- cells[sample.int(length(cells), size = 1L, prob = ws)]
    picked_tile_weights[j] <- sum(ws)
    picked_tile_names[j] <- names(by_tile)[chosen_ids[j]]
  }
  all_ri_values <- as.matrix(terra::values(ri, mat = TRUE))
  Y <- all_ri_values[picked_cells, , drop = FALSE]
  rm(all_ri_values)
  stopifnot(ncol(Y) == length(need_year), all(is.finite(Y)))
  sample_out <- data.frame(
    Raster_cell = picked_cells, Spatial_tile = picked_tile_names,
    Tile_complete_case_area_km2 = picked_tile_weights
  )
  write_csv(sample_out, "S12_E_零模型空间抽样记录.csv")
  n_s <- nrow(Y)
  nt <- ncol(Y)
  time_X <- cbind(1, need_year - need_year[1])
  trend_pred <- matrix(NA_real_, nrow = n_s, ncol = nt)
  residuals <- matrix(NA_real_, nrow = n_s, ncol = nt)
  phi <- numeric(n_s)
  innovations <- matrix(NA_real_, nrow = n_s, ncol = nt - 1L)
  for (i in seq_len(n_s)) {
    f <- stats::lm.fit(time_X, Y[i, ])
    trend_pred[i, ] <- as.vector(time_X %*% f$coefficients)
    residuals[i, ] <- Y[i, ] - trend_pred[i, ]
    e <- residuals[i, ]
    denom <- sum(e[1:(nt - 1L)]^2)
    rho <- if (denom > 1e-15) sum(e[2:nt] * e[1:(nt - 1L)]) / denom else 0
    phi[i] <- max(-0.95, min(0.95, rho))
    ie <- e[2:nt] - phi[i] * e[1:(nt - 1L)]
    innovations[i, ] <- ie - mean(ie)
  }
  observed_det <- vapply(seq_len(n_s), function(i) {
    detect_one(Y[i, ], need_year, 5L)
  }, numeric(3L))
  obs_turn <- observed_det[2, ]
  ratio_stat <- function(bp_years) {
    has <- is.finite(bp_years)
    denom <- sum(picked_tile_weights[has])
    if (!is.finite(denom) || denom <= 0) return(NA_real_)
    100 * sum(picked_tile_weights[has & bp_years >= 1990 &
                                      bp_years <= 1995]) / denom
  }
  observed_share <- ratio_stat(obs_turn)
  if (!is.finite(observed_share)) {
    stop("空间样本中没有识别到任何转折点；无法做比例式零模型比较。")
  }
  null_df <- data.frame(
    Replicate = seq_len(null_repetitions),
    Early_share_pct = NA_real_,
    Detected_turning_area_weight = NA_real_,
    Detected_turning_tile_count = NA_integer_
  )
  for (b in seq_len(null_repetitions)) {
    # Draw one cross-sectional innovation field per year, shared by all
    # sampled tiles. This partially retains cross-location covariance.
    shock_cols <- sample.int(nt - 1L, nt - 1L, replace = TRUE)
    starting_residual_col <- sample.int(nt, 1L)
    sim_e <- matrix(NA_real_, n_s, nt)
    sim_e[, 1] <- residuals[, starting_residual_col]
    for (k in 2:nt) {
      sim_e[, k] <- phi * sim_e[, k - 1L] + innovations[, shock_cols[k - 1L]]
    }
    sim_y <- trend_pred + sim_e
    detected <- vapply(seq_len(n_s), function(i) {
      detect_one(sim_y[i, ], need_year, 5L)
    }, numeric(3L))
    sim_year <- detected[2, ]
    null_df$Early_share_pct[b] <- ratio_stat(sim_year)
    null_df$Detected_turning_area_weight[b] <- sum(picked_tile_weights[is.finite(sim_year)])
    null_df$Detected_turning_tile_count[b] <- sum(is.finite(sim_year))
    if (b %% 25L == 0L || b == null_repetitions) {
      log_msg("零假设模拟完成: ", b, "/", null_repetitions)
    }
  }
  write_csv(null_df, "S12_F_零假设逐次模拟.csv")
  ok_sim <- is.finite(null_df$Early_share_pct)
  if (sum(ok_sim) < 0.8 * null_repetitions) {
    warning("超过20%的零模型重复无转折点，条件比例的零模型参考分布不稳定，谨慎解读。")
  }
  q <- if (any(ok_sim)) {
    stats::quantile(null_df$Early_share_pct[ok_sim],
                    probs = c(0.025, 0.5, 0.975), names = FALSE)
  } else rep(NA_real_, 3L)
  exceed_p <- if (any(ok_sim)) {
    (1 + sum(null_df$Early_share_pct[ok_sim] >= observed_share)) /
      (1 + sum(ok_sim))
  } else NA_real_
  null_summary <- data.frame(
    Dataset = "Complete-case approximate-8km spatial tile sample (not full area)",
    Sample_tile_count = n_s,
    Candidate_complete_case_tiles = length(by_tile),
    Null_repetitions_requested = null_repetitions,
    Null_repetitions_with_detected_breaks = sum(ok_sim),
    Observed_early_share_pct = observed_share,
    Null_early_share_pct_2p5 = q[1],
    Null_early_share_pct_median = q[2],
    Null_early_share_pct_97p5 = q[3],
    Exploratory_upper_tail_p = exceed_p,
    Random_seed = random_seed
  )
  write_csv(null_summary, "S12_G_零假设检验汇总_样本范围.csv")
  log_msg("样本观测早期比例: ", round(observed_share, 3), "%")
  log_msg("零模型中位数: ", round(q[2], 3), "%")
  log_msg("零模型经验上尾概率(探索性): ", round(exceed_p, 5))
}

# -------------------- 5. RUN RECORD AND FINISH -------------------
run_record <- data.frame(
  Parameter = c("Input_RI", "Original_grass_mask", "Start_year", "End_year",
                "Alpha", "AIC_delta", "Min_valid_years", "Boundary_gaps",
                "Null_enabled", "Null_repetitions", "Null_tiles_max",
                "Null_block_size_cells", "Random_seed", "terra_version",
                "R_version"),
  Value = as.character(c(ri_dir, grass_shp, start_year, end_year, alpha,
                         aic_delta, min_valid_years,
                         paste(boundary_gaps, collapse = ","),
                         run_null_model, null_repetitions, null_max_tiles,
                         null_block_cells, random_seed,
                         as.character(utils::packageVersion("terra")),
                         as.character(getRversion())))
)
write_csv(run_record, "00_运行参数与软件版本.csv")
log_msg("完成: ", as.character(Sys.time()), "；结果目录: ", out_dir)
log_msg("提示：只有原始复现核查通过，且读到的CSV显示真实结果后，才能填写论文和回复信。")
