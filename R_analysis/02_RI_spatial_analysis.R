# ============================================================
# Grassland RI pixel-scale spatial heterogeneity analysis (1985-2020)
# Public/reproducible repository version
# ============================================================
# Main functions:
# 1) Read annual RI raster layers for 1985-2020
# 2) Restrict all analyses to the grassland mask
# 3) Identify no-trend, linear-trend, and abrupt/turning-change pixels
# 4) Classify I-D / D-I turning directions and turning years
# 5) Calculate coefficient of variation (CV)
# 6) Export raster results and statistical CSV tables
#
# Repository path convention:
#   project_root/
#   ├── this_script.R
#   ├── data/
#   │   ├── RI_annual/         # annual RI GeoTIFF files
#   │   └── mask/
#   │       └── grassland.shp  # grassland mask (+ .dbf/.shx/.prj etc.)
#   └── output/                # created automatically
#
# IMPORTANT:
# - No local absolute paths are hard-coded in this public version.
# - Run the script with the repository root as the working directory.
# - The analytical/statistical logic is unchanged from the original script.
# ============================================================

rm(list = ls())
gc()

library(terra)

options(scipen = 999)

# ===================== 1. Project paths =====================

# Repository root.
# Recommended usage: set the working directory to the repository root before running.
project_root <- normalizePath(".", winslash = "/", mustWork = FALSE)

# Input directories/files
data_root <- file.path(project_root, "data")
var_folder <- file.path(data_root, "RI_annual")
grass_shp_path <- file.path(data_root, "mask", "grassland.shp")

# Output directories (created automatically)
output_root <- file.path(project_root, "output")
raster_root <- file.path(output_root, "rasters")
report_root <- file.path(output_root, "reports")

# Basic input checks
if (!dir.exists(var_folder)) {
  stop(
    paste0(
      "Input RI folder not found: ", var_folder, "\n",
      "Please place annual RI GeoTIFF files in: data/RI_annual/"
    )
  )
}

if (!file.exists(grass_shp_path)) {
  stop(
    paste0(
      "Grassland shapefile not found: ", grass_shp_path, "\n",
      "Please place the grassland shapefile and its companion files in: data/mask/"
    )
  )
}

# Run tag protects raster outputs from accidental overwriting
run_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Raster output subdirectories
dir_b_raster       <- file.path(raster_root, "trend_type")
dir_c_raster       <- file.path(raster_root, "turning_direction_ID_DI")
dir_d_raster       <- file.path(raster_root, "turning_year")
dir_linear_raster  <- file.path(raster_root, "linear_direction")
dir_subtype_raster <- file.path(raster_root, "trend_subtype")
dir_cv_raster      <- file.path(raster_root, "CV")

# Report output subdirectories
dir_b_report       <- file.path(report_root, "01_trend_type")
dir_c_report       <- file.path(report_root, "02_ID_DI")
dir_d_report       <- file.path(report_root, "03_turning_year")
dir_cv_report      <- file.path(report_root, "04_CV")
dir_linear_report  <- file.path(report_root, "05_linear_direction")
dir_subtype_report <- file.path(report_root, "06_trend_subtype")
dir_diag_report    <- file.path(report_root, "07_model_diagnostics")
dir_all_report     <- file.path(report_root, "08_pixel_level_results")

dir_list <- c(
  dir_b_raster,
  dir_c_raster,
  dir_d_raster,
  dir_linear_raster,
  dir_subtype_raster,
  dir_cv_raster,
  dir_b_report,
  dir_c_report,
  dir_d_report,
  dir_cv_report,
  dir_linear_report,
  dir_subtype_report,
  dir_diag_report,
  dir_all_report
)

for (d in dir_list) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# Raster outputs: run_tag is appended to avoid overwriting previous results
out_b_trend_type <- file.path(
  dir_b_raster,
  paste0("RI_Trend_Type_B_1985_2020_Grassland_", run_tag, ".tif")
)

out_c_id_di <- file.path(
  dir_c_raster,
  paste0("RI_Turning_Direction_C_ID_DI_1985_2020_Grassland_", run_tag, ".tif")
)

out_d_year <- file.path(
  dir_d_raster,
  paste0("RI_Turning_Year_D_1985_2020_Grassland_", run_tag, ".tif")
)

out_linear_dir <- file.path(
  dir_linear_raster,
  paste0("RI_Linear_Direction_1985_2020_Grassland_", run_tag, ".tif")
)

out_subtype <- file.path(
  dir_subtype_raster,
  paste0("RI_Trend_Subtype_1985_2020_Grassland_", run_tag, ".tif")
)

out_cv_value <- file.path(
  dir_cv_raster,
  paste0("RI_CV_Value_1985_2020_Grassland_", run_tag, ".tif")
)

out_cv_class <- file.path(
  dir_cv_raster,
  paste0("RI_CV_Class_1985_2020_Grassland_", run_tag, ".tif")
)

# ===================== 2. 参数设置 =====================

start_year <- 1985
end_year   <- 2020

alpha <- 0.05
aic_delta <- 2
min_gap <- 5

# TRUE = 输出像元级完整 CSV，文件可能较大
write_pixel_csv <- TRUE

# Windows 下建议先用 1，稳定后可尝试 2 或 4
use_cores <- 1

# ===================== 3. 读取年度 RI 栅格 =====================

files <- sort(list.files(var_folder, pattern = "\\.tif$", full.names = TRUE))

years <- sapply(files, function(x) {
  as.numeric(regmatches(basename(x), regexpr("[12][0-9]{3}", basename(x))))
})

sel <- !is.na(years) & years >= start_year & years <= end_year
files <- files[sel]
years <- years[sel]

ord <- order(years)
files <- files[ord]
years <- years[ord]

if (length(files) == 0) {
  stop("错误:未找到 1985-2020 年 tif 文件，请检查输入路径和文件命名。")
}

cat("识别到年份:", paste(years, collapse = ", "), "\n")
cat("共读取栅格层数:", length(files), "\n")

r_stack <- rast(files)
names(r_stack) <- paste0("RI_", years)

grass_shp <- vect(grass_shp_path)

if (!same.crs(r_stack, grass_shp)) {
  grass_shp <- project(grass_shp, crs(r_stack))
}

# 关键:趋势计算前，直接按草地范围裁剪和掩膜
r_stack <- crop(r_stack, grass_shp)
r_stack <- mask(r_stack, grass_shp)

# 草地范围内有效 RI 像元
r_valid <- app(r_stack, function(x) {
  if (all(is.na(x))) {
    return(NA)
  } else {
    return(1)
  }
})

# ===================== 4. 基础函数 =====================

calc_aic <- function(rss, n, k) {
  if (is.na(rss) || rss <= 0 || n <= k) return(NA_real_)
  n * log(rss / n) + 2 * k
}

fit_lm_fast <- function(X, y) {
  
  n <- length(y)
  k <- ncol(X)
  
  if (n <= k + 1) {
    return(list(
      coef = rep(NA_real_, k),
      p = rep(NA_real_, k),
      rss = NA_real_,
      aic = NA_real_
    ))
  }
  
  fit <- try(lm.fit(x = X, y = y), silent = TRUE)
  
  if (inherits(fit, "try-error")) {
    return(list(
      coef = rep(NA_real_, k),
      p = rep(NA_real_, k),
      rss = NA_real_,
      aic = NA_real_
    ))
  }
  
  coef <- fit$coefficients
  res  <- fit$residuals
  rss  <- sum(res^2, na.rm = TRUE)
  df   <- n - k
  
  if (df <= 0 || is.na(rss) || rss <= 0) {
    return(list(
      coef = coef,
      p = rep(NA_real_, k),
      rss = rss,
      aic = calc_aic(rss, n, k)
    ))
  }
  
  sigma2 <- rss / df
  R <- try(qr.R(fit$qr), silent = TRUE)
  
  if (inherits(R, "try-error")) {
    pvals <- rep(NA_real_, k)
  } else {
    XtX_inv <- try(chol2inv(R), silent = TRUE)
    
    if (inherits(XtX_inv, "try-error")) {
      pvals <- rep(NA_real_, k)
    } else {
      se <- sqrt(diag(XtX_inv) * sigma2)
      tval <- coef / se
      pvals <- 2 * pt(abs(tval), df = df, lower.tail = FALSE)
    }
  }
  
  list(
    coef = coef,
    p = pvals,
    rss = rss,
    aic = calc_aic(rss, n, k)
  )
}

pettitt_test_fast <- function(y) {
  
  n <- length(y)
  
  if (n < 8) {
    return(list(p = NA_real_, k_index = NA_integer_, K = NA_real_))
  }
  
  U <- rep(NA_real_, n - 1)
  
  for (t in 1:(n - 1)) {
    x1 <- y[1:t]
    x2 <- y[(t + 1):n]
    U[t] <- sum(outer(x1, x2, function(a, b) sign(a - b)))
  }
  
  K_abs <- max(abs(U), na.rm = TRUE)
  k_index <- which.max(abs(U))
  
  p <- 2 * exp((-6 * K_abs^2) / (n^3 + n^2))
  p <- min(max(p, 0), 1)
  
  list(p = p, k_index = k_index, K = K_abs)
}

get_turning_direction_stats <- function(y, year_vec, bp_year, alpha = 0.05) {
  
  idx1 <- which(year_vec <= bp_year)
  idx2 <- which(year_vec > bp_year)
  
  if (length(idx1) < 4 || length(idx2) < 4) {
    return(c(NA, NA, NA, NA, NA))
  }
  
  y1 <- y[idx1]
  y2 <- y[idx2]
  
  t1 <- year_vec[idx1] - min(year_vec[idx1])
  t2 <- year_vec[idx2] - min(year_vec[idx2])
  
  m1 <- fit_lm_fast(cbind(1, t1), y1)
  m2 <- fit_lm_fast(cbind(1, t2), y2)
  
  slope1 <- m1$coef[2]
  slope2 <- m2$coef[2]
  p1 <- m1$p[2]
  p2 <- m2$p[2]
  
  dir_code <- NA_real_
  
  if (!is.na(slope1) && !is.na(slope2) && !is.na(p1) && !is.na(p2)) {
    
    if (p1 < alpha && p2 < alpha && slope1 > 0 && slope2 < 0) {
      dir_code <- 1
    }
    
    if (p1 < alpha && p2 < alpha && slope1 < 0 && slope2 > 0) {
      dir_code <- 2
    }
  }
  
  return(c(dir_code, slope1, slope2, p1, p2))
}

# ===================== 5. 单像元趋势识别函数 =====================

trend_pixel_fun <- function(x) {
  
  # 返回 25 个值:
  # 1  trend_type
  # 2  linear_direction
  # 3  turning_direction
  # 4  turning_year
  # 5  final_model
  # 6  trend_subtype
  # 7  linear_slope
  # 8  linear_p
  # 9  aic_linear
  # 10 seg_year
  # 11 seg_p_change
  # 12 aic_segmented
  # 13 seg_direction
  # 14 seg_slope_before
  # 15 seg_slope_after
  # 16 seg_p_before
  # 17 seg_p_after
  # 18 pettitt_year
  # 19 pettitt_p
  # 20 aic_pettitt
  # 21 pettitt_direction
  # 22 pettitt_slope_before
  # 23 pettitt_slope_after
  # 24 pettitt_p_before
  # 25 pettitt_p_after
  
  ok <- !is.na(x)
  y <- x[ok]
  yy <- years[ok]
  
  if (length(y) < 10) {
    return(rep(NA_real_, 25))
  }
  
  if (sd(y, na.rm = TRUE) < 1e-12) {
    return(c(
      0, NA, NA, NA, 0, 0,
      0, 1, NA,
      NA, NA, NA, NA, NA, NA, NA, NA,
      NA, NA, NA, NA, NA, NA, NA, NA
    ))
  }
  
  tt <- yy - min(yy)
  
  # ---------- 1) 一次线性回归 ----------
  fit_linear <- fit_lm_fast(cbind(1, tt), y)
  
  linear_slope <- fit_linear$coef[2]
  linear_p <- fit_linear$p[2]
  aic_linear <- fit_linear$aic
  
  linear_sig <- !is.na(linear_p) && linear_p < alpha
  
  linear_direction <- NA_real_
  if (linear_sig && !is.na(linear_slope)) {
    if (linear_slope > 0) linear_direction <- 1
    if (linear_slope < 0) linear_direction <- 2
  }
  
  # ---------- 2) 分段线性回归 ----------
  candidate_years <- yy[
    yy >= (start_year + min_gap) &
      yy <= (end_year - min_gap)
  ]
  
  candidate_years <- unique(candidate_years)
  
  best_seg_aic <- NA_real_
  best_seg_year <- NA_real_
  best_seg_p_change <- NA_real_
  best_seg_dir <- NA_real_
  best_seg_slope1 <- NA_real_
  best_seg_slope2 <- NA_real_
  best_seg_p1 <- NA_real_
  best_seg_p2 <- NA_real_
  
  if (length(candidate_years) > 0) {
    
    for (bp in candidate_years) {
      
      if (sum(yy <= bp) < 4 || sum(yy > bp) < 4) next
      
      bp_t <- bp - min(yy)
      hinge <- pmax(0, tt - bp_t)
      
      fit_seg <- fit_lm_fast(cbind(1, tt, hinge), y)
      aic_seg <- fit_seg$aic
      p_change <- fit_seg$p[3]
      
      if (is.na(aic_seg)) next
      
      dir_stats <- get_turning_direction_stats(
        y = y,
        year_vec = yy,
        bp_year = bp,
        alpha = alpha
      )
      
      dir_code <- dir_stats[1]
      
      if (is.na(best_seg_aic) || aic_seg < best_seg_aic) {
        best_seg_aic <- aic_seg
        best_seg_year <- bp
        best_seg_p_change <- p_change
        best_seg_dir <- dir_code
        best_seg_slope1 <- dir_stats[2]
        best_seg_slope2 <- dir_stats[3]
        best_seg_p1 <- dir_stats[4]
        best_seg_p2 <- dir_stats[5]
      }
    }
  }
  
  seg_sig_raw <- !is.na(best_seg_p_change) && best_seg_p_change < alpha
  seg_valid_abrupt <- seg_sig_raw && !is.na(best_seg_dir)
  
  # ---------- 3) Pettitt 检验 ----------
  pet <- pettitt_test_fast(y)
  
  pet_year <- NA_real_
  pet_p <- NA_real_
  aic_pet <- NA_real_
  pet_dir <- NA_real_
  pet_slope1 <- NA_real_
  pet_slope2 <- NA_real_
  pet_p1 <- NA_real_
  pet_p2 <- NA_real_
  
  pet_sig_raw <- FALSE
  pet_valid_year <- FALSE
  pet_valid_abrupt <- FALSE
  
  if (!is.na(pet$p) && !is.na(pet$k_index)) {
    
    pet_p <- pet$p
    pet_year <- yy[pet$k_index]
    
    pet_valid_year <- (pet_year >= start_year + min_gap) &&
      (pet_year <= end_year - min_gap)
    
    pet_sig_raw <- pet_p < alpha
    
    if (pet_valid_year) {
      
      group <- ifelse(yy <= pet_year, 0, 1)
      fit_pet <- fit_lm_fast(cbind(1, group), y)
      aic_pet <- fit_pet$aic
      
      pet_dir_stats <- get_turning_direction_stats(
        y = y,
        year_vec = yy,
        bp_year = pet_year,
        alpha = alpha
      )
      
      pet_dir <- pet_dir_stats[1]
      pet_slope1 <- pet_dir_stats[2]
      pet_slope2 <- pet_dir_stats[3]
      pet_p1 <- pet_dir_stats[4]
      pet_p2 <- pet_dir_stats[5]
    }
    
    pet_valid_abrupt <- pet_sig_raw && pet_valid_year && !is.na(pet_dir)
  }
  
  # ---------- 4) 候选模型筛选 ----------
  model_names <- c()
  model_aic <- c()
  model_rank <- c()
  
  if (linear_sig && !is.na(aic_linear)) {
    model_names <- c(model_names, "linear")
    model_aic <- c(model_aic, aic_linear)
    model_rank <- c(model_rank, 1)
  }
  
  if (pet_valid_abrupt && !is.na(aic_pet)) {
    model_names <- c(model_names, "pettitt")
    model_aic <- c(model_aic, aic_pet)
    model_rank <- c(model_rank, 2)
  }
  
  if (seg_valid_abrupt && !is.na(best_seg_aic)) {
    model_names <- c(model_names, "segmented")
    model_aic <- c(model_aic, best_seg_aic)
    model_rank <- c(model_rank, 3)
  }
  
  # ---------- 5) 无显著候选模型 ----------
  if (length(model_names) == 0) {
    
    return(c(
      0, NA, NA, NA, 0, 0,
      linear_slope, linear_p, aic_linear,
      best_seg_year, best_seg_p_change, best_seg_aic,
      best_seg_dir, best_seg_slope1, best_seg_slope2, best_seg_p1, best_seg_p2,
      pet_year, pet_p, aic_pet,
      pet_dir, pet_slope1, pet_slope2, pet_p1, pet_p2
    ))
  }
  
  # ---------- 6) AIC + 简洁性选择 ----------
  min_aic <- min(model_aic, na.rm = TRUE)
  candidate_idx <- which(model_aic <= min_aic + aic_delta)
  best_idx <- candidate_idx[which.min(model_rank[candidate_idx])]
  selected_model <- model_names[best_idx]
  
  # ---------- 7) 输出最终分类 ----------
  trend_type <- NA_real_
  turning_direction <- NA_real_
  turning_year <- NA_real_
  final_model <- NA_real_
  trend_subtype <- NA_real_
  
  if (selected_model == "linear") {
    
    trend_type <- 1
    final_model <- 1
    
    if (linear_direction == 1) {
      trend_subtype <- 11
    } else if (linear_direction == 2) {
      trend_subtype <- 12
    } else {
      trend_subtype <- NA_real_
    }
    
  } else if (selected_model == "pettitt") {
    
    trend_type <- 2
    final_model <- 2
    turning_direction <- pet_dir
    turning_year <- pet_year
    
    if (turning_direction == 1) trend_subtype <- 21
    if (turning_direction == 2) trend_subtype <- 22
    
  } else if (selected_model == "segmented") {
    
    trend_type <- 2
    final_model <- 3
    turning_direction <- best_seg_dir
    turning_year <- best_seg_year
    
    if (turning_direction == 1) trend_subtype <- 21
    if (turning_direction == 2) trend_subtype <- 22
  }
  
  return(c(
    trend_type,
    linear_direction,
    turning_direction,
    turning_year,
    final_model,
    trend_subtype,
    linear_slope,
    linear_p,
    aic_linear,
    best_seg_year,
    best_seg_p_change,
    best_seg_aic,
    best_seg_dir,
    best_seg_slope1,
    best_seg_slope2,
    best_seg_p1,
    best_seg_p2,
    pet_year,
    pet_p,
    aic_pet,
    pet_dir,
    pet_slope1,
    pet_slope2,
    pet_p1,
    pet_p2
  ))
}

# ===================== 6. 像元尺度趋势识别 =====================

cat("\n开始进行草地范围内像元尺度趋势识别，请耐心等待...\n")

r_result <- app(
  r_stack,
  fun = trend_pixel_fun,
  cores = use_cores
)

names(r_result) <- c(
  "trend_type",
  "linear_direction",
  "turning_direction",
  "turning_year",
  "final_model",
  "trend_subtype",
  "linear_slope",
  "linear_p",
  "aic_linear",
  "seg_year",
  "seg_p_change",
  "aic_segmented",
  "seg_direction",
  "seg_slope_before",
  "seg_slope_after",
  "seg_p_before",
  "seg_p_after",
  "pettitt_year",
  "pettitt_p",
  "aic_pettitt",
  "pettitt_direction",
  "pettitt_slope_before",
  "pettitt_slope_after",
  "pettitt_p_before",
  "pettitt_p_after"
)

r_b <- r_result[["trend_type"]]
r_c <- r_result[["turning_direction"]]
r_d <- r_result[["turning_year"]]
r_linear_dir <- r_result[["linear_direction"]]
r_subtype <- r_result[["trend_subtype"]]

# ===================== 7. 计算 CV 变异系数与 CV 分类 =====================

cat("\n正在计算 CV 变异系数及分类栅格...\n")

r_mean <- app(r_stack, mean, na.rm = TRUE)
r_sd   <- app(r_stack, sd, na.rm = TRUE)

r_cv <- (r_sd / r_mean) * 100
r_cv <- mask(r_cv, r_valid)

# CV 分类:
# 1 = <20%
# 2 = 20%-30%
# 3 = >30%
r_cv_class <- classify(
  r_cv,
  rcl = matrix(
    c(
      -Inf, 20, 1,
      20, 30, 2,
      30, Inf, 3
    ),
    ncol = 3,
    byrow = TRUE
  ),
  include.lowest = TRUE,
  right = FALSE
)

# ===================== 8. 写出栅格 =====================

cat("\n正在写出草地范围内栅格。本次运行不会覆盖原有栅格文件...\n")

writeRaster(
  r_b,
  out_b_trend_type,
  overwrite = FALSE,
  datatype = "INT1U",
  NAflag = 255,
  gdal = c("COMPRESS=LZW")
)

writeRaster(
  r_c,
  out_c_id_di,
  overwrite = FALSE,
  datatype = "INT1U",
  NAflag = 255,
  gdal = c("COMPRESS=LZW")
)

writeRaster(
  r_d,
  out_d_year,
  overwrite = FALSE,
  datatype = "INT2S",
  NAflag = -9999,
  gdal = c("COMPRESS=LZW")
)

writeRaster(
  r_linear_dir,
  out_linear_dir,
  overwrite = FALSE,
  datatype = "INT1U",
  NAflag = 255,
  gdal = c("COMPRESS=LZW")
)

writeRaster(
  r_subtype,
  out_subtype,
  overwrite = FALSE,
  datatype = "INT2S",
  NAflag = -9999,
  gdal = c("COMPRESS=LZW")
)

writeRaster(
  r_cv,
  out_cv_value,
  overwrite = FALSE,
  datatype = "FLT4S",
  NAflag = -9999,
  gdal = c("COMPRESS=LZW")
)

writeRaster(
  r_cv_class,
  out_cv_class,
  overwrite = FALSE,
  datatype = "INT1U",
  NAflag = 255,
  gdal = c("COMPRESS=LZW")
)

# ===================== 9. 计算面积 =====================

cat("\n正在计算草地范围内面积...\n")

r_area_km2 <- cellSize(r_valid, unit = "km")

grassland_area_km2 <- global(
  mask(r_area_km2, r_valid),
  fun = "sum",
  na.rm = TRUE
)[1, 1]

cat("草地范围内有效 RI 像元面积 km²:", grassland_area_km2, "\n")

# ===================== 10. 分类面积统计函数 =====================

stat_categorical_area <- function(r_cat, r_area, total_area, label_df, out_csv) {
  
  df <- as.data.frame(c(r_cat, r_area), xy = FALSE, na.rm = FALSE)
  names(df) <- c("value", "area_km2")
  df <- df[!is.na(df$value) & !is.na(df$area_km2), ]
  
  if (nrow(df) == 0) {
    
    out <- data.frame(
      value = numeric(),
      label = character(),
      pixel_count = integer(),
      area_km2 = numeric(),
      proportion_percent = numeric()
    )
    
  } else {
    
    area_out <- aggregate(
      area_km2 ~ value,
      data = df,
      FUN = sum,
      na.rm = TRUE
    )
    
    cnt <- as.data.frame(table(df$value))
    names(cnt) <- c("value", "pixel_count")
    cnt$value <- as.numeric(as.character(cnt$value))
    
    out <- merge(area_out, cnt, by = "value", all.x = TRUE)
    out <- merge(out, label_df, by = "value", all.x = TRUE)
    
    out$proportion_percent <- out$area_km2 / total_area * 100
    
    out <- out[, c("value", "label", "pixel_count", "area_km2", "proportion_percent")]
    out <- out[order(out$value), ]
  }
  
  write.csv(out, out_csv, row.names = FALSE, fileEncoding = "UTF-8")
  return(out)
}

# ===================== 11. 各类面积与比例报表 =====================

label_b <- data.frame(
  value = c(0, 1, 2),
  label = c("No trend", "Linear trend", "Abrupt / turning change")
)

label_c <- data.frame(
  value = c(1, 2),
  label = c("I-D: Increase to Decrease", "D-I: Decrease to Increase")
)

label_linear <- data.frame(
  value = c(1, 2),
  label = c("Linear increase", "Linear decrease")
)

label_subtype <- data.frame(
  value = c(0, 11, 12, 21, 22),
  label = c("No trend", "Linear increase", "Linear decrease", "I-D", "D-I")
)

label_cv <- data.frame(
  value = c(1, 2, 3),
  label = c("<20%", "20%-30%", ">30%")
)

stat_b <- stat_categorical_area(
  r_cat = r_b,
  r_area = r_area_km2,
  total_area = grassland_area_km2,
  label_df = label_b,
  out_csv = file.path(dir_b_report, "趋势类型_面积与比例统计_占草地范围.csv")
)

stat_c <- stat_categorical_area(
  r_cat = r_c,
  r_area = r_area_km2,
  total_area = grassland_area_km2,
  label_df = label_c,
  out_csv = file.path(dir_c_report, "ID_DI_面积与比例统计_占草地范围.csv")
)

stat_linear <- stat_categorical_area(
  r_cat = r_linear_dir,
  r_area = r_area_km2,
  total_area = grassland_area_km2,
  label_df = label_linear,
  out_csv = file.path(dir_linear_report, "线性趋势方向_面积与比例统计_占草地范围.csv")
)

stat_subtype <- stat_categorical_area(
  r_cat = r_subtype,
  r_area = r_area_km2,
  total_area = grassland_area_km2,
  label_df = label_subtype,
  out_csv = file.path(dir_subtype_report, "综合趋势子类型_面积与比例统计_占草地范围.csv")
)

stat_cv <- stat_categorical_area(
  r_cat = r_cv_class,
  r_area = r_area_km2,
  total_area = grassland_area_km2,
  label_df = label_cv,
  out_csv = file.path(dir_cv_report, "CV变异系数分类_面积与比例统计_占草地范围.csv")
)

# I-D / D-I 占突变草地像元比例
abrupt_area_km2 <- stat_b$area_km2[stat_b$value == 2]
if (length(abrupt_area_km2) == 0 || is.na(abrupt_area_km2)) {
  abrupt_area_km2 <- NA_real_
}

stat_c_abrupt <- stat_c
stat_c_abrupt$proportion_in_abrupt_percent <- stat_c_abrupt$area_km2 / abrupt_area_km2 * 100

write.csv(
  stat_c_abrupt,
  file.path(dir_c_report, "ID_DI_面积与比例统计_占突变草地像元.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 12. 转折年份统计 =====================

df_d <- as.data.frame(c(r_d, r_area_km2), xy = FALSE, na.rm = FALSE)
names(df_d) <- c("turning_year", "area_km2")
df_d <- df_d[!is.na(df_d$turning_year) & !is.na(df_d$area_km2), ]

if (nrow(df_d) > 0) {
  
  stat_d_year <- aggregate(
    area_km2 ~ turning_year,
    data = df_d,
    FUN = sum,
    na.rm = TRUE
  )
  
  cnt_d <- as.data.frame(table(df_d$turning_year))
  names(cnt_d) <- c("turning_year", "pixel_count")
  cnt_d$turning_year <- as.numeric(as.character(cnt_d$turning_year))
  
  stat_d_year <- merge(stat_d_year, cnt_d, by = "turning_year", all.x = TRUE)
  stat_d_year$proportion_percent <- stat_d_year$area_km2 / grassland_area_km2 * 100
  stat_d_year <- stat_d_year[order(stat_d_year$turning_year), ]
  
} else {
  
  stat_d_year <- data.frame(
    turning_year = numeric(),
    area_km2 = numeric(),
    pixel_count = integer(),
    proportion_percent = numeric()
  )
}

write.csv(
  stat_d_year,
  file.path(dir_d_report, "转折年份_逐年面积与比例统计_占草地范围.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

df_d$turning_period <- cut(
  df_d$turning_year,
  breaks = c(1989, 1995, 2000, 2005, 2010, 2015),
  labels = c("1990-1995", "1995-2000", "2000-2005", "2005-2010", "2010-2015"),
  right = TRUE,
  include.lowest = TRUE
)

df_d_period <- df_d[!is.na(df_d$turning_period), ]

if (nrow(df_d_period) > 0) {
  
  stat_d_period <- aggregate(
    area_km2 ~ turning_period,
    data = df_d_period,
    FUN = sum,
    na.rm = TRUE
  )
  
  cnt_dp <- as.data.frame(table(df_d_period$turning_period))
  names(cnt_dp) <- c("turning_period", "pixel_count")
  
  stat_d_period <- merge(stat_d_period, cnt_dp, by = "turning_period", all.x = TRUE)
  stat_d_period$proportion_percent <- stat_d_period$area_km2 / grassland_area_km2 * 100
  
} else {
  
  stat_d_period <- data.frame(
    turning_period = character(),
    area_km2 = numeric(),
    pixel_count = integer(),
    proportion_percent = numeric()
  )
}

write.csv(
  stat_d_period,
  file.path(dir_d_report, "转折年份_分段面积与比例统计_占草地范围.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 13. 模型诊断表 =====================

cat("\n正在生成模型诊断报表...\n")

df_diag <- as.data.frame(
  c(r_result, r_area_km2),
  xy = FALSE,
  na.rm = FALSE
)

names(df_diag)[ncol(df_diag)] <- "area_km2"

df_diag <- df_diag[!is.na(df_diag$trend_type) & !is.na(df_diag$area_km2), ]

df_diag$linear_sig <- !is.na(df_diag$linear_p) & df_diag$linear_p < alpha
df_diag$seg_sig_raw <- !is.na(df_diag$seg_p_change) & df_diag$seg_p_change < alpha
df_diag$pettitt_sig_raw <- !is.na(df_diag$pettitt_p) & df_diag$pettitt_p < alpha

df_diag$seg_valid_abrupt <- df_diag$seg_sig_raw & !is.na(df_diag$seg_direction)

df_diag$pettitt_valid_abrupt <- df_diag$pettitt_sig_raw &
  !is.na(df_diag$pettitt_direction) &
  !is.na(df_diag$pettitt_year) &
  df_diag$pettitt_year >= (start_year + min_gap) &
  df_diag$pettitt_year <= (end_year - min_gap)

df_diag$final_model_label <- ifelse(df_diag$final_model == 0, "No trend",
                                    ifelse(df_diag$final_model == 1, "Linear",
                                           ifelse(df_diag$final_model == 2, "Pettitt",
                                                  ifelse(df_diag$final_model == 3, "Segmented", NA))))

df_diag$trend_type_label <- ifelse(df_diag$trend_type == 0, "No trend",
                                   ifelse(df_diag$trend_type == 1, "Linear trend",
                                          ifelse(df_diag$trend_type == 2, "Abrupt / turning change", NA)))

df_diag$trend_subtype_label <- ifelse(df_diag$trend_subtype == 0, "No trend",
                                      ifelse(df_diag$trend_subtype == 11, "Linear increase",
                                             ifelse(df_diag$trend_subtype == 12, "Linear decrease",
                                                    ifelse(df_diag$trend_subtype == 21, "I-D",
                                                           ifelse(df_diag$trend_subtype == 22, "D-I", NA)))))

condition_area <- function(cond) {
  sum(df_diag$area_km2[cond], na.rm = TRUE)
}

diagnosis_table <- data.frame(
  Item = c(
    "Linear regression significant",
    "Segmented regression raw significant",
    "Pettitt test raw significant",
    "Segmented valid abrupt and classified as I-D/D-I",
    "Pettitt valid abrupt and classified as I-D/D-I",
    "Final No trend",
    "Final Linear trend",
    "Final Abrupt / turning change",
    "Final Linear increase",
    "Final Linear decrease",
    "Final I-D",
    "Final D-I"
  ),
  Area_km2 = c(
    condition_area(df_diag$linear_sig),
    condition_area(df_diag$seg_sig_raw),
    condition_area(df_diag$pettitt_sig_raw),
    condition_area(df_diag$seg_valid_abrupt),
    condition_area(df_diag$pettitt_valid_abrupt),
    condition_area(df_diag$trend_type == 0),
    condition_area(df_diag$trend_type == 1),
    condition_area(df_diag$trend_type == 2),
    condition_area(df_diag$trend_subtype == 11),
    condition_area(df_diag$trend_subtype == 12),
    condition_area(df_diag$trend_subtype == 21),
    condition_area(df_diag$trend_subtype == 22)
  )
)

diagnosis_table$Proportion_in_grassland_percent <- diagnosis_table$Area_km2 / grassland_area_km2 * 100

write.csv(
  diagnosis_table,
  file.path(dir_diag_report, "模型诊断_显著性筛选与最终分类面积统计_占草地范围.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# p 值和 AIC 分组统计
summarise_num <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    return(c(
      n = 0,
      mean = NA,
      median = NA,
      min = NA,
      q25 = NA,
      q75 = NA,
      max = NA
    ))
  }
  return(c(
    n = length(x),
    mean = mean(x),
    median = median(x),
    min = min(x),
    q25 = as.numeric(quantile(x, 0.25)),
    q75 = as.numeric(quantile(x, 0.75)),
    max = max(x)
  ))
}

vars_to_summary <- c(
  "linear_p",
  "seg_p_change",
  "pettitt_p",
  "aic_linear",
  "aic_segmented",
  "aic_pettitt",
  "linear_slope",
  "seg_slope_before",
  "seg_slope_after",
  "pettitt_slope_before",
  "pettitt_slope_after"
)

summary_list <- list()
idx <- 1

groups <- unique(df_diag$trend_subtype_label)
groups <- groups[!is.na(groups)]

for (g in groups) {
  sub <- df_diag[df_diag$trend_subtype_label == g, ]
  
  for (v in vars_to_summary) {
    s <- summarise_num(sub[[v]])
    
    summary_list[[idx]] <- data.frame(
      group = g,
      variable = v,
      n = s["n"],
      mean = s["mean"],
      median = s["median"],
      min = s["min"],
      q25 = s["q25"],
      q75 = s["q75"],
      max = s["max"]
    )
    
    idx <- idx + 1
  }
}

if (length(summary_list) > 0) {
  p_aic_summary <- do.call(rbind, summary_list)
} else {
  p_aic_summary <- data.frame()
}

write.csv(
  p_aic_summary,
  file.path(dir_diag_report, "模型诊断_p值_AIC_斜率_分组统计.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# 最终模型来源统计
model_source <- aggregate(
  area_km2 ~ final_model_label,
  data = df_diag,
  FUN = sum,
  na.rm = TRUE
)

model_count <- as.data.frame(table(df_diag$final_model_label))
names(model_count) <- c("final_model_label", "pixel_count")

model_source <- merge(model_source, model_count, by = "final_model_label", all.x = TRUE)
model_source$proportion_in_grassland_percent <- model_source$area_km2 / grassland_area_km2 * 100

write.csv(
  model_source,
  file.path(dir_diag_report, "模型诊断_最终模型来源统计_占草地范围.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 14. 综合表 =====================

get_prop <- function(stat_df, value_id) {
  x <- stat_df$proportion_percent[stat_df$value == value_id]
  if (length(x) == 0 || is.na(x)) return(0)
  round(x, 2)
}

get_area <- function(stat_df, value_id) {
  x <- stat_df$area_km2[stat_df$value == value_id]
  if (length(x) == 0 || is.na(x)) return(0)
  round(x, 2)
}

get_period_prop <- function(stat_df, period_name) {
  x <- stat_df$proportion_percent[as.character(stat_df$turning_period) == period_name]
  if (length(x) == 0 || is.na(x)) return(0)
  round(x, 2)
}

get_period_area <- function(stat_df, period_name) {
  x <- stat_df$area_km2[as.character(stat_df$turning_period) == period_name]
  if (length(x) == 0 || is.na(x)) return(0)
  round(x, 2)
}

summary_table_percent <- data.frame(
  Object = "Grassland range",
  
  Grassland_valid_area_km2 = round(grassland_area_km2, 2),
  
  CV_lt20_percent = get_prop(stat_cv, 1),
  CV_20_30_percent = get_prop(stat_cv, 2),
  CV_gt30_percent = get_prop(stat_cv, 3),
  
  No_trend_percent = get_prop(stat_b, 0),
  Linear_trend_percent = get_prop(stat_b, 1),
  Abrupt_change_percent = get_prop(stat_b, 2),
  
  Linear_increase_percent = get_prop(stat_linear, 1),
  Linear_decrease_percent = get_prop(stat_linear, 2),
  
  I_D_percent = get_prop(stat_c, 1),
  D_I_percent = get_prop(stat_c, 2),
  
  Turning_1990_1995_percent = get_period_prop(stat_d_period, "1990-1995"),
  Turning_1995_2000_percent = get_period_prop(stat_d_period, "1995-2000"),
  Turning_2000_2005_percent = get_period_prop(stat_d_period, "2000-2005"),
  Turning_2005_2010_percent = get_period_prop(stat_d_period, "2005-2010"),
  Turning_2010_2015_percent = get_period_prop(stat_d_period, "2010-2015")
)

summary_table_area <- data.frame(
  Object = "Grassland range",
  
  Grassland_valid_area_km2 = round(grassland_area_km2, 2),
  
  CV_lt20_area_km2 = get_area(stat_cv, 1),
  CV_20_30_area_km2 = get_area(stat_cv, 2),
  CV_gt30_area_km2 = get_area(stat_cv, 3),
  
  No_trend_area_km2 = get_area(stat_b, 0),
  Linear_trend_area_km2 = get_area(stat_b, 1),
  Abrupt_change_area_km2 = get_area(stat_b, 2),
  
  Linear_increase_area_km2 = get_area(stat_linear, 1),
  Linear_decrease_area_km2 = get_area(stat_linear, 2),
  
  I_D_area_km2 = get_area(stat_c, 1),
  D_I_area_km2 = get_area(stat_c, 2),
  
  Turning_1990_1995_area_km2 = get_period_area(stat_d_period, "1990-1995"),
  Turning_1995_2000_area_km2 = get_period_area(stat_d_period, "1995-2000"),
  Turning_2000_2005_area_km2 = get_period_area(stat_d_period, "2000-2005"),
  Turning_2005_2010_area_km2 = get_period_area(stat_d_period, "2005-2010"),
  Turning_2010_2015_area_km2 = get_period_area(stat_d_period, "2010-2015")
)

write.csv(
  summary_table_percent,
  file.path(report_root, "综合表_RI空间异质性统计_比例_占草地范围.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  summary_table_area,
  file.path(report_root, "综合表_RI空间异质性统计_面积_km2.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 15. 像元级综合报表 =====================

if (write_pixel_csv) {
  
  cat("\n正在输出像元级综合报表，像元较多时文件会比较大...\n")
  
  df_all <- as.data.frame(
    c(
      r_cv,
      r_cv_class,
      r_result,
      r_area_km2
    ),
    xy = TRUE,
    na.rm = FALSE
  )
  
  names(df_all)[1:2] <- c("x", "y")
  names(df_all)[3] <- "CV_value"
  names(df_all)[4] <- "CV_class"
  names(df_all)[ncol(df_all)] <- "area_km2"
  
  df_all <- df_all[!is.na(df_all$trend_type) & !is.na(df_all$area_km2), ]
  
  df_all$CV_class_label <- ifelse(df_all$CV_class == 1, "<20%",
                                  ifelse(df_all$CV_class == 2, "20%-30%",
                                         ifelse(df_all$CV_class == 3, ">30%", NA)))
  
  df_all$trend_type_label <- ifelse(df_all$trend_type == 0, "No trend",
                                    ifelse(df_all$trend_type == 1, "Linear trend",
                                           ifelse(df_all$trend_type == 2, "Abrupt / turning change", NA)))
  
  df_all$linear_direction_label <- ifelse(df_all$linear_direction == 1, "Linear increase",
                                          ifelse(df_all$linear_direction == 2, "Linear decrease", NA))
  
  df_all$turning_direction_label <- ifelse(df_all$turning_direction == 1, "I-D",
                                           ifelse(df_all$turning_direction == 2, "D-I", NA))
  
  df_all$final_model_label <- ifelse(df_all$final_model == 0, "No trend",
                                     ifelse(df_all$final_model == 1, "Linear",
                                            ifelse(df_all$final_model == 2, "Pettitt",
                                                   ifelse(df_all$final_model == 3, "Segmented", NA))))
  
  df_all$trend_subtype_label <- ifelse(df_all$trend_subtype == 0, "No trend",
                                       ifelse(df_all$trend_subtype == 11, "Linear increase",
                                              ifelse(df_all$trend_subtype == 12, "Linear decrease",
                                                     ifelse(df_all$trend_subtype == 21, "I-D",
                                                            ifelse(df_all$trend_subtype == 22, "D-I", NA)))))
  
  write.csv(
    df_all,
    file.path(dir_all_report, "像元尺度_RI_CV_趋势类型_IDDI_转折年份_p值_AIC_综合结果.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
}

# ===================== 16. 一致性检查 =====================

abrupt_area <- get_area(stat_b, 2)
id_area <- get_area(stat_c, 1)
di_area <- get_area(stat_c, 2)

check_table <- data.frame(
  Check_item = c(
    "Abrupt area",
    "I-D area + D-I area",
    "Difference"
  ),
  Area_km2 = c(
    abrupt_area,
    id_area + di_area,
    abrupt_area - id_area - di_area
  )
)

write.csv(
  check_table,
  file.path(dir_diag_report, "一致性检查_Abrupt是否等于ID加DI.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 17. 输出提示 =====================

cat("\n全部完成!\n\n")

cat("本次运行批次号:", run_tag, "\n\n")

cat("栅格输出总路径:\n", raster_root, "\n\n")
cat("报表输出路径:\n", report_root, "\n\n")

cat("图 b 趋势类型栅格:\n", out_b_trend_type, "\n")
cat("编码:0 = No trend;1 = Linear trend;2 = Abrupt / turning change\n\n")

cat("图 c I-D / D-I 栅格:\n", out_c_id_di, "\n")
cat("编码:1 = I-D;2 = D-I;NoData = 非突变像元\n\n")

cat("图 d 转折年份栅格:\n", out_d_year, "\n")
cat("编码:像元值即转折年份;NoData = 非突变像元\n\n")

cat("线性方向补充栅格:\n", out_linear_dir, "\n")
cat("编码:1 = Linear increase;2 = Linear decrease\n\n")

cat("综合趋势子类型栅格:\n", out_subtype, "\n")
cat("编码:0 = No trend;11 = Linear increase;12 = Linear decrease;21 = I-D;22 = D-I\n\n")

cat("CV 原始值栅格:\n", out_cv_value, "\n")
cat("CV 分类栅格:\n", out_cv_class, "\n")
cat("编码:1 = <20%;2 = 20%-30%;3 = >30%\n\n")

cat("草地范围内有效 RI 像元面积 km²:", round(grassland_area_km2, 2), "\n\n")

cat("CV 变异系数统计，占草地范围:\n")
print(stat_cv)

cat("\n趋势类型统计，占草地范围:\n")
print(stat_b)

cat("\nI-D / D-I 统计，占草地范围:\n")
print(stat_c)

cat("\n线性方向统计，占草地范围:\n")
print(stat_linear)

cat("\n综合趋势子类型统计，占草地范围:\n")
print(stat_subtype)

cat("\n转折年份分段统计，占草地范围:\n")
print(stat_d_period)

cat("\n模型诊断统计:\n")
print(diagnosis_table)

cat("\n一致性检查:Abrupt 是否等于 I-D + D-I\n")
print(check_table)

cat("\n提醒:\n")
cat("1. 本代码输出的所有趋势类 tif 均已按草地范围掩膜。\n")
cat("2. CSV 比例分母为草地范围内有效 RI 像元面积。\n")
cat("3. 栅格文件名已添加时间戳，且 overwrite = FALSE，不会覆盖原有栅格。\n")