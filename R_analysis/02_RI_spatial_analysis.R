# Script: 02_RI_spatial_analysis.R
# Final publication/GitHub version: 2026-08-30
#
# ============================================================
# 1985—2020 年川西北草地 RI 像元尺度空间异质性分析
# 统一有效分析范围优化版
#
# 核心规则：
# 1. 所有年度 RI 栅格先裁剪并掩膜到草地范围。
# 2. 共同有效分析范围要求：
#    (a) 1985—2020 年有效 RI 观测数不少于 min_valid_years；
#    (b) 能够计算均值和标准差；
#    (c) RI 均值绝对值大于 cv_mean_eps，保证 CV 有定义。
# 3. CV、最终趋势类型、最终线性方向、I-D/D-I、转折年份及
#    综合趋势子类型全部使用同一共同有效分析范围和同一面积分母。
# 4. 最终线性方向只统计“最终被选为线性趋势”的像元，
#    因而线性增加面积 + 线性降低面积 = 最终线性趋势面积。
# 5. 转折时段沿用论文图件中的边界式标签：
#    1990—1995、1995—2000、2000—2005、2005—2010、2010—2015。
#    代码按左开右闭规则分组，边界年份不会重复统计：
#    (1989,1995]、(1995,2000]、(2000,2005]、(2005,2010]、(2010,2015]。
# 6. 面积首先由 terra::cellSize() 计算真实像元面积；论文表格报告时统一取整到 1 km²。
#    对互斥且完备的分类采用“总量约束取整”（largest remainder / Hamilton 法），
#    保证各类别整数面积之和严格等于父级整数总面积，再据此重算比例。
# 7. 栅格每次运行自动建立 Run_时间戳 独立目录，集中保存且不覆盖旧栅格；
#    统计报表固定输出到原报表目录，使用最新结果覆盖同名 CSV。
# ============================================================

rm(list = ls())
gc()

if (!requireNamespace("terra", quietly = TRUE)) {
  stop("未安装 terra 包。请先运行：install.packages('terra')")
}
library(terra)

options(scipen = 999)

# ===================== 1. 路径设置 =====================

# 输入年度 RI 栅格文件夹
var_folder <- "F:/牧区草地研究/川西北mq_ri"

# 草地范围矢量
grass_shp_path <- "F:/牧区草地研究/mask/甘孜阿坝草甸_沼泽类型.shp"

# 本次运行批次号：用于建立独立运行目录，避免覆盖旧结果
run_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")

# ------------------------------------------------------------
# 栅格统一集中输出
# 所有本次生成的栅格均放在：
# F:/牧区草地研究/牧区_像元尺度/RI空间异质性_1985_2020_最终结果/Run_时间戳/
# ------------------------------------------------------------
raster_parent <- "F:/牧区草地研究/牧区_像元尺度"
raster_project_root <- file.path(
  raster_parent,
  "RI空间异质性_1985_2020_最终结果"
)
raster_root <- file.path(
  raster_project_root,
  paste0("Run_", run_tag)
)

# 栅格输出子文件夹：全部集中在同一个本次运行目录中
dir_b_raster       <- file.path(raster_root, "01_趋势类型")
dir_c_raster       <- file.path(raster_root, "02_ID_DI转折类型")
dir_d_raster       <- file.path(raster_root, "03_转折年份")
dir_linear_raster  <- file.path(raster_root, "04_最终线性趋势方向")
dir_subtype_raster <- file.path(raster_root, "05_综合趋势子类型")
dir_cv_raster      <- file.path(raster_root, "06_CV")
dir_qc_raster      <- file.path(raster_root, "07_质控")

# ------------------------------------------------------------
# 报表输出：保持原来的固定位置
# 本次重新运行会在该目录下更新同名 CSV，不再建立 Run_时间戳 子目录。
# ------------------------------------------------------------
report_root <- "C:/Users/yihai/Desktop/川西北/草地_ri年际变化空间异质性_优化版/统计报表最新1111"

# 报表输出子文件夹
dir_b_report       <- file.path(report_root, "01_趋势类型统计")
dir_c_report       <- file.path(report_root, "02_ID_DI转折类型统计")
dir_d_report       <- file.path(report_root, "03_转折年份统计")
dir_cv_report      <- file.path(report_root, "04_CV变异系数统计")
dir_linear_report  <- file.path(report_root, "05_最终线性方向统计")
dir_subtype_report <- file.path(report_root, "06_综合子类型统计")
dir_diag_report    <- file.path(report_root, "07_模型诊断与一致性检查")
dir_all_report     <- file.path(report_root, "08_像元级综合CSV")
dir_qc_report      <- file.path(report_root, "09_有效范围与分母检查")

dir_list <- c(
  dir_b_raster,
  dir_c_raster,
  dir_d_raster,
  dir_linear_raster,
  dir_subtype_raster,
  dir_cv_raster,
  dir_qc_raster,
  dir_b_report,
  dir_c_report,
  dir_d_report,
  dir_cv_report,
  dir_linear_report,
  dir_subtype_report,
  dir_diag_report,
  dir_all_report,
  dir_qc_report
)

for (d in dir_list) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# 栅格输出文件：本次运行目录已含时间戳，文件名保持简洁；overwrite = FALSE
out_b_trend_type <- file.path(
  dir_b_raster,
  "RI_Trend_Type_B_1985_2020_CommonValid.tif"
)

out_c_id_di <- file.path(
  dir_c_raster,
  "RI_Turning_Direction_C_ID_DI_1985_2020_CommonValid.tif"
)

out_d_year <- file.path(
  dir_d_raster,
  "RI_Turning_Year_D_1985_2020_CommonValid.tif"
)

out_linear_dir_final <- file.path(
  dir_linear_raster,
  "RI_Final_Linear_Direction_1985_2020_CommonValid.tif"
)

out_subtype <- file.path(
  dir_subtype_raster,
  "RI_Trend_Subtype_1985_2020_CommonValid.tif"
)

out_cv_value <- file.path(
  dir_cv_raster,
  "RI_CV_Value_1985_2020_CommonValid.tif"
)

out_cv_class <- file.path(
  dir_cv_raster,
  "RI_CV_Class_1985_2020_CommonValid.tif"
)

out_valid_year_count <- file.path(
  dir_qc_raster,
  "RI_Valid_Year_Count_1985_2020_Grassland.tif"
)

out_common_mask <- file.path(
  dir_qc_raster,
  "RI_Common_Analysis_Mask_1985_2020.tif"
)

# ===================== 2. 参数设置 =====================

start_year <- 1985L
end_year   <- 2020L

alpha <- 0.05
aic_delta <- 2
min_gap <- 5L

# 方式 2：共同有效分析范围至少需要 10 个有效年份
min_valid_years <- 10L

# CV 分母接近 0 时视为不可计算
cv_mean_eps <- 1e-12

# TRUE = 输出像元级完整 CSV，文件可能较大
write_pixel_csv <- TRUE

# Windows 下建议先用 1；运行稳定后可尝试 2 或 4
use_cores <- 1L

# 论文表格面积报告精度：1 km²
area_reporting_unit_km2 <- 1

# ===================== 3. 读取年度 RI 栅格 =====================

# 仅识别形如 RI1985_n_VG.tif、RI1986_n_VG.tif 的年度 RI 栅格。
# 这样可自动排除 RI_mean_1985_2020.tif 等汇总栅格，以及 .tfw、.xml、.aux.xml 等辅助文件。
annual_file_pattern <- "^RI([12][0-9]{3})_n_VG\\.(tif|tiff)$"

files <- sort(
  list.files(
    var_folder,
    pattern = annual_file_pattern,
    full.names = TRUE,
    ignore.case = TRUE
  )
)

if (length(files) == 0) {
  stop(
    "错误：未找到命名格式为 RI年份_n_VG.tif 的年度 RI 栅格。",
    "请检查输入路径和文件名，例如 RI1985_n_VG.tif。"
  )
}

# 从严格匹配的年度文件名中提取年份。
years <- as.integer(
  sub(
    annual_file_pattern,
    "\\1",
    basename(files),
    ignore.case = TRUE
  )
)

sel <- !is.na(years) & years >= start_year & years <= end_year
files <- files[sel]
years <- years[sel]

ord <- order(years)
files <- files[ord]
years <- years[ord]

if (length(files) == 0) {
  stop("错误：未找到 1985—2020 年 tif 文件，请检查输入路径和文件命名。")
}

if (anyDuplicated(years)) {
  dup_years <- unique(years[duplicated(years)])
  stop(
    "错误：检测到重复年份栅格：",
    paste(dup_years, collapse = ", "),
    "。请确保每年仅有一个 RI 栅格。"
  )
}

missing_years <- setdiff(start_year:end_year, years)
if (length(missing_years) > 0) {
  warning(
    "以下年份未找到 RI 栅格：",
    paste(missing_years, collapse = ", "),
    "。代码仍会运行，但有效年份数按实际读取层数统计。"
  )
}

if (length(files) < min_valid_years) {
  stop(
    "错误：读取到的年度栅格层数少于 min_valid_years = ",
    min_valid_years,
    "，无法开展趋势分析。"
  )
}

cat("识别到年份：", paste(years, collapse = ", "), "\n")
cat("共读取栅格层数：", length(files), "\n")

r_stack_grass <- rast(files)
names(r_stack_grass) <- paste0("RI_", years)

grass_shp <- vect(grass_shp_path)

if (!same.crs(r_stack_grass, grass_shp)) {
  grass_shp <- project(grass_shp, crs(r_stack_grass))
}

# 先按草地范围裁剪和掩膜
r_stack_grass <- crop(r_stack_grass, grass_shp)
r_stack_grass <- mask(r_stack_grass, grass_shp)

# ===================== 4. 构建统一有效分析范围 =====================

cat("\n正在构建统一有效分析范围...\n")

# 每个像元的有限 RI 观测数
r_n_valid <- app(
  r_stack_grass,
  function(x) sum(is.finite(x))
)
names(r_n_valid) <- "valid_year_count"

# 计算用于 CV 的均值和标准差；仅使用有限值
r_mean_all <- app(
  r_stack_grass,
  function(x) {
    z <- x[is.finite(x)]
    if (length(z) == 0) return(NA_real_)
    mean(z)
  }
)
names(r_mean_all) <- "RI_mean"

r_sd_all <- app(
  r_stack_grass,
  function(x) {
    z <- x[is.finite(x)]
    if (length(z) < 2) return(NA_real_)
    sd(z)
  }
)
names(r_sd_all) <- "RI_sd"

# 草地范围内至少存在 1 个有效 RI 的像元
r_any_ri_mask <- ifel(r_n_valid >= 1, 1, NA)
names(r_any_ri_mask) <- "any_valid_RI_mask"

# 至少满足 min_valid_years 的像元
r_min_year_mask <- ifel(r_n_valid >= min_valid_years, 1, NA)
names(r_min_year_mask) <- "min_valid_year_mask"

# 共同有效分析范围：
# 1) 有效 RI 年份数 >= min_valid_years
# 2) 均值和标准差有效
# 3) 均值不接近 0，CV 有定义
common_condition <- (
  r_n_valid >= min_valid_years &
    !is.na(r_mean_all) &
    !is.na(r_sd_all) &
    abs(r_mean_all) > cv_mean_eps
)

r_analysis_mask <- ifel(common_condition, 1, NA)
names(r_analysis_mask) <- "common_analysis_mask"

# 后续 CV 和趋势分析全部统一到这一范围
r_stack <- mask(r_stack_grass, r_analysis_mask)

# 面积栅格：仍使用 cellSize() 计算每个像元的真实面积（km²）
# 注意：这里不把每个像元“硬设为 1 km²”。
# 老师要求的“统一取整到 1 km²”在报表统计阶段完成，
# 并通过总量约束取整保证各分类面积严格闭合。
r_cell_area_km2 <- cellSize(r_stack_grass[[1]], unit = "km")

mask_area_km2 <- function(mask_r, cell_area_r) {
  x <- global(
    mask(cell_area_r, mask_r),
    fun = "sum",
    na.rm = TRUE
  )[1, 1]
  as.numeric(x)
}

mask_pixel_count <- function(mask_r) {
  x <- global(mask_r, fun = "sum", na.rm = TRUE)[1, 1]
  as.numeric(x)
}

# 真实面积（内部计算与审计使用）
any_ri_area_km2_exact <- mask_area_km2(r_any_ri_mask, r_cell_area_km2)
min_year_area_km2_exact <- mask_area_km2(r_min_year_mask, r_cell_area_km2)
analysis_area_km2_exact <- mask_area_km2(r_analysis_mask, r_cell_area_km2)

# 像元数
any_ri_pixel_count <- mask_pixel_count(r_any_ri_mask)
min_year_pixel_count <- mask_pixel_count(r_min_year_mask)
analysis_pixel_count <- mask_pixel_count(r_analysis_mask)

# 论文表格使用的整数面积（1 km² 精度）
any_ri_area_km2 <- round(any_ri_area_km2_exact / area_reporting_unit_km2) * area_reporting_unit_km2
min_year_area_km2 <- round(min_year_area_km2_exact / area_reporting_unit_km2) * area_reporting_unit_km2
analysis_area_km2 <- round(analysis_area_km2_exact / area_reporting_unit_km2) * area_reporting_unit_km2

# 后续分类聚合仍使用真实像元面积；报表阶段再统一闭合取整
r_area_km2 <- mask(r_cell_area_km2, r_analysis_mask)
names(r_area_km2) <- "area_km2"

cat("至少 1 个有效 RI 的草地面积（真实 km²）：", any_ri_area_km2_exact, "\n")
cat("有效年份数不少于", min_valid_years, "年的面积（真实 km²）：", min_year_area_km2_exact, "\n")
cat("统一有效分析面积（真实 km²）：", analysis_area_km2_exact, "\n")
cat("统一有效分析面积（论文报告，取整到 1 km²）：", analysis_area_km2, "\n")

# 有效范围与分母检查表
# 同时保留真实面积与论文报告整数面积，便于审稿时追溯。
denominator_audit <- data.frame(
  Item = c(
    "Grassland pixels with at least 1 valid RI observation",
    paste0("Pixels with at least ", min_valid_years, " valid RI years"),
    "Common valid analysis range",
    paste0("Excluded because valid RI years < ", min_valid_years),
    "Additionally excluded because CV is undefined"
  ),
  Pixel_count = c(
    any_ri_pixel_count,
    min_year_pixel_count,
    analysis_pixel_count,
    any_ri_pixel_count - min_year_pixel_count,
    min_year_pixel_count - analysis_pixel_count
  ),
  Area_km2_exact = c(
    any_ri_area_km2_exact,
    min_year_area_km2_exact,
    analysis_area_km2_exact,
    any_ri_area_km2_exact - min_year_area_km2_exact,
    min_year_area_km2_exact - analysis_area_km2_exact
  ),
  Area_km2_report_1km2 = c(
    any_ri_area_km2,
    min_year_area_km2,
    analysis_area_km2,
    round((any_ri_area_km2_exact - min_year_area_km2_exact) / area_reporting_unit_km2) * area_reporting_unit_km2,
    round((min_year_area_km2_exact - analysis_area_km2_exact) / area_reporting_unit_km2) * area_reporting_unit_km2
  ),
  Proportion_of_any_RI_area_percent = c(
    100,
    min_year_area_km2 / any_ri_area_km2 * 100,
    analysis_area_km2 / any_ri_area_km2 * 100,
    (any_ri_area_km2 - min_year_area_km2) / any_ri_area_km2 * 100,
    (min_year_area_km2 - analysis_area_km2) / any_ri_area_km2 * 100
  )
)

write.csv(
  denominator_audit,
  file.path(dir_qc_report, "统一有效分析范围_面积分母核查.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 5. 基础函数 =====================

calc_aic <- function(rss, n, k) {
  if (is.na(rss) || !is.finite(rss) || rss <= 0 || n <= k) {
    return(NA_real_)
  }
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

  if (df <= 0 || is.na(rss) || !is.finite(rss) || rss <= 0) {
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

  c(dir_code, slope1, slope2, p1, p2)
}

# ===================== 6. 单像元趋势识别函数 =====================

trend_pixel_fun <- function(x) {

  # 返回 25 个值：
  # 1  trend_type
  # 2  linear_direction_raw
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

  ok <- is.finite(x)
  y <- x[ok]
  yy <- years[ok]

  if (length(y) < min_valid_years) {
    return(rep(NA_real_, 25))
  }

  if (sd(y) < 1e-12) {
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

  linear_direction_raw <- NA_real_
  if (linear_sig && !is.na(linear_slope)) {
    if (linear_slope > 0) linear_direction_raw <- 1
    if (linear_slope < 0) linear_direction_raw <- 2
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

    pet_valid_year <- (
      pet_year >= start_year + min_gap &&
        pet_year <= end_year - min_gap
    )

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

    pet_valid_abrupt <- (
      pet_sig_raw &&
        pet_valid_year &&
        !is.na(pet_dir)
    )
  }

  # ---------- 4) 候选模型筛选 ----------
  model_names <- character(0)
  model_aic <- numeric(0)
  model_rank <- numeric(0)

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
      best_seg_dir, best_seg_slope1, best_seg_slope2,
      best_seg_p1, best_seg_p2,
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

    if (linear_direction_raw == 1) {
      trend_subtype <- 11
    } else if (linear_direction_raw == 2) {
      trend_subtype <- 12
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

  c(
    trend_type,
    linear_direction_raw,
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
  )
}

# ===================== 7. 像元尺度趋势识别 =====================

cat("\n开始进行统一有效分析范围内的像元尺度趋势识别，请耐心等待...\n")

r_result <- app(
  r_stack,
  fun = trend_pixel_fun,
  cores = use_cores
)

names(r_result) <- c(
  "trend_type",
  "linear_direction_raw",
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

r_result <- mask(r_result, r_analysis_mask)

r_b <- r_result[["trend_type"]]
r_c <- r_result[["turning_direction"]]
r_d <- r_result[["turning_year"]]
r_subtype <- r_result[["trend_subtype"]]

# 最终线性方向：只保留最终趋势类型为 Linear trend 的像元
r_linear_dir_final <- ifel(
  r_b == 1,
  r_result[["linear_direction_raw"]],
  NA
)
names(r_linear_dir_final) <- "final_linear_direction"

# ===================== 8. 计算 CV 及分类 =====================

cat("\n正在计算统一有效分析范围内的 CV 及分类...\n")

r_mean <- mask(r_mean_all, r_analysis_mask)
r_sd <- mask(r_sd_all, r_analysis_mask)

r_cv <- (r_sd / r_mean) * 100
r_cv <- mask(r_cv, r_analysis_mask)
names(r_cv) <- "CV_value"

# 分类边界：
# 1 = CV < 20%
# 2 = 20% <= CV <= 30%
# 3 = CV > 30%
r_cv_class <- ifel(
  is.na(r_cv),
  NA,
  ifel(
    r_cv < 20,
    1,
    ifel(r_cv <= 30, 2, 3)
  )
)
names(r_cv_class) <- "CV_class"

# ===================== 9. 写出栅格 =====================

cat("\n正在写出栅格。本次运行不会覆盖原有栅格文件...\n")

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
  r_linear_dir_final,
  out_linear_dir_final,
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

writeRaster(
  mask(r_n_valid, r_any_ri_mask),
  out_valid_year_count,
  overwrite = FALSE,
  datatype = "INT1U",
  NAflag = 255,
  gdal = c("COMPRESS=LZW")
)

writeRaster(
  r_analysis_mask,
  out_common_mask,
  overwrite = FALSE,
  datatype = "INT1U",
  NAflag = 255,
  gdal = c("COMPRESS=LZW")
)

# ===================== 10. 分类面积统计函数 =====================

# -----------------------------------------------------------------
# 10.1 总量约束取整（largest remainder / Hamilton 法）
# -----------------------------------------------------------------
# 目的：
# 1) 先用 cellSize() 得到真实面积；
# 2) 论文表格统一报告到 1 km²；
# 3) 各互斥类别取整后之和严格等于父级整数总面积；
# 4) 避免 142,243.74 vs 142,243.73、27,463.65 vs 27,463.64
#    这类由独立保留小数造成的不闭合。
#
# 例如：趋势类型的 3 类整数面积之和严格 = analysis_area_km2；
#       I-D + D-I 严格 = Abrupt 整数面积；
#       5 个转折时段整数面积之和严格 = Abrupt 整数面积。
# -----------------------------------------------------------------
balanced_round_area <- function(
  exact_areas,
  target_total_km2,
  unit_km2 = area_reporting_unit_km2
) {

  x <- as.numeric(exact_areas)
  x[is.na(x)] <- 0

  if (any(x < 0)) {
    stop("balanced_round_area: 面积不能为负数。")
  }

  if (!is.finite(target_total_km2) || target_total_km2 < 0) {
    stop("balanced_round_area: target_total_km2 非法。")
  }

  scaled <- x / unit_km2
  base_units <- floor(scaled)
  frac <- scaled - base_units

  target_units <- as.integer(round(target_total_km2 / unit_km2))
  need <- target_units - sum(base_units)

  out_units <- base_units

  if (need > 0) {
    # 将剩余单位优先分配给小数部分最大的类别。
    # order(..., seq_along()) 保证完全相同时结果可复现。
    ord <- order(-frac, seq_along(frac))
    if (need > length(ord)) {
      stop("balanced_round_area: 需要补加的单位数异常，请检查输入面积。")
    }
    out_units[ord[seq_len(need)]] <- out_units[ord[seq_len(need)]] + 1
  }

  if (need < 0) {
    # 理论上在正常的四舍五入目标下几乎不会发生；保留稳健兜底。
    can_subtract <- which(out_units > 0)
    ord <- can_subtract[order(frac[can_subtract], can_subtract)]
    n_sub <- abs(need)
    if (n_sub > length(ord)) {
      stop("balanced_round_area: 需要扣减的单位数异常，请检查输入面积。")
    }
    out_units[ord[seq_len(n_sub)]] <- out_units[ord[seq_len(n_sub)]] - 1
  }

  out <- out_units * unit_km2

  # 最终硬检查：整数报告面积必须严格闭合。
  if (abs(sum(out) - target_total_km2) > 1e-9) {
    stop(
      "balanced_round_area: 取整后未闭合。sum = ", sum(out),
      "; target = ", target_total_km2
    )
  }

  as.numeric(out)
}


# -----------------------------------------------------------------
# 10.2 分类面积统计
# -----------------------------------------------------------------
stat_categorical_area <- function(
  r_cat,
  r_area,
  denominator_area_km2,
  closure_total_area_km2,
  label_df,
  out_csv
) {

  df <- as.data.frame(
    c(r_cat, r_area),
    xy = FALSE,
    na.rm = FALSE
  )
  names(df) <- c("value", "area_km2_exact")

  df <- df[
    !is.na(df$value) &
      !is.na(df$area_km2_exact),
  ]

  if (nrow(df) > 0) {

    area_out <- aggregate(
      area_km2_exact ~ value,
      data = df,
      FUN = sum,
      na.rm = TRUE
    )

    cnt <- as.data.frame(table(df$value))
    names(cnt) <- c("value", "pixel_count")
    cnt$value <- as.numeric(as.character(cnt$value))

    out <- merge(label_df, area_out, by = "value", all.x = TRUE)
    out <- merge(out, cnt, by = "value", all.x = TRUE)

  } else {

    out <- label_df
    out$area_km2_exact <- 0
    out$pixel_count <- 0
  }

  out$area_km2_exact[is.na(out$area_km2_exact)] <- 0
  out$pixel_count[is.na(out$pixel_count)] <- 0
  out$pixel_count <- as.integer(out$pixel_count)

  # 先按 value 排序，保证取整分配稳定、可复现。
  out <- out[order(out$value), ]

  # 论文正式报告面积：统一到 1 km²，且严格服从父级总量约束。
  out$area_km2 <- balanced_round_area(
    exact_areas = out$area_km2_exact,
    target_total_km2 = closure_total_area_km2,
    unit_km2 = area_reporting_unit_km2
  )

  # 百分比使用“取整后的正式报告面积”重新计算。
  out$proportion_percent <-
    out$area_km2 / denominator_area_km2 * 100

  out <- out[
    ,
    c(
      "value",
      "label",
      "pixel_count",
      "area_km2_exact",
      "area_km2",
      "proportion_percent"
    )
  ]

  write.csv(
    out,
    out_csv,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  out
}

# ===================== 11. 各类面积与比例报表 =====================

label_b <- data.frame(
  value = c(0, 1, 2),
  label = c(
    "No trend",
    "Linear trend",
    "Abrupt / turning change"
  )
)

label_c <- data.frame(
  value = c(1, 2),
  label = c(
    "I-D: Increase to Decrease",
    "D-I: Decrease to Increase"
  )
)

label_linear_final <- data.frame(
  value = c(1, 2),
  label = c(
    "Final linear increase",
    "Final linear decrease"
  )
)

label_subtype <- data.frame(
  value = c(0, 11, 12, 21, 22),
  label = c(
    "No trend",
    "Final linear increase",
    "Final linear decrease",
    "I-D",
    "D-I"
  )
)

label_cv <- data.frame(
  value = c(1, 2, 3),
  label = c(
    "<20%",
    "20%-30%",
    ">30%"
  )
)

stat_b <- stat_categorical_area(
  r_cat = r_b,
  r_area = r_area_km2,
  denominator_area_km2 = analysis_area_km2,
  closure_total_area_km2 = analysis_area_km2,
  label_df = label_b,
  out_csv = file.path(
    dir_b_report,
    "趋势类型_面积与比例统计_占统一有效分析范围.csv"
  )
)

# 父级整数面积：Abrupt / turning change
abrupt_area_km2 <- stat_b$area_km2[stat_b$value == 2]
if (length(abrupt_area_km2) == 0 || is.na(abrupt_area_km2)) {
  abrupt_area_km2 <- 0
}

# 父级整数面积：Linear trend
linear_area_km2 <- stat_b$area_km2[stat_b$value == 1]
if (length(linear_area_km2) == 0 || is.na(linear_area_km2)) {
  linear_area_km2 <- 0
}

stat_c <- stat_categorical_area(
  r_cat = r_c,
  r_area = r_area_km2,
  denominator_area_km2 = analysis_area_km2,
  closure_total_area_km2 = abrupt_area_km2,
  label_df = label_c,
  out_csv = file.path(
    dir_c_report,
    "ID_DI_面积与比例统计_占统一有效分析范围.csv"
  )
)

stat_linear_final <- stat_categorical_area(
  r_cat = r_linear_dir_final,
  r_area = r_area_km2,
  denominator_area_km2 = analysis_area_km2,
  closure_total_area_km2 = linear_area_km2,
  label_df = label_linear_final,
  out_csv = file.path(
    dir_linear_report,
    "最终线性趋势方向_面积与比例统计_占统一有效分析范围.csv"
  )
)

stat_subtype <- stat_categorical_area(
  r_cat = r_subtype,
  r_area = r_area_km2,
  denominator_area_km2 = analysis_area_km2,
  closure_total_area_km2 = analysis_area_km2,
  label_df = label_subtype,
  out_csv = file.path(
    dir_subtype_report,
    "综合趋势子类型_面积与比例统计_占统一有效分析范围.csv"
  )
)

stat_cv <- stat_categorical_area(
  r_cat = r_cv_class,
  r_area = r_area_km2,
  denominator_area_km2 = analysis_area_km2,
  closure_total_area_km2 = analysis_area_km2,
  label_df = label_cv,
  out_csv = file.path(
    dir_cv_report,
    "CV变异系数分类_面积与比例统计_占统一有效分析范围.csv"
  )
)

# I-D / D-I 占最终突变草地像元比例
stat_c_abrupt <- stat_c

if (abrupt_area_km2 > 0) {
  stat_c_abrupt$proportion_in_abrupt_percent <-
    stat_c_abrupt$area_km2 / abrupt_area_km2 * 100
} else {
  stat_c_abrupt$proportion_in_abrupt_percent <- NA_real_
}

write.csv(
  stat_c_abrupt,
  file.path(
    dir_c_report,
    "ID_DI_面积与比例统计_占最终突变草地像元.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 12. 转折年份统计 =====================

df_d <- as.data.frame(
  c(r_d, r_area_km2),
  xy = FALSE,
  na.rm = FALSE
)

names(df_d) <- c("turning_year", "area_km2_exact")

df_d <- df_d[
  !is.na(df_d$turning_year) &
    !is.na(df_d$area_km2_exact),
]

if (nrow(df_d) > 0) {

  stat_d_year <- aggregate(
    area_km2_exact ~ turning_year,
    data = df_d,
    FUN = sum,
    na.rm = TRUE
  )

  cnt_d <- as.data.frame(table(df_d$turning_year))
  names(cnt_d) <- c("turning_year", "pixel_count")
  cnt_d$turning_year <- as.numeric(
    as.character(cnt_d$turning_year)
  )

  stat_d_year <- merge(
    stat_d_year,
    cnt_d,
    by = "turning_year",
    all.x = TRUE
  )

  # 各转折年份也是 Abrupt 的进一步划分：整数面积严格闭合到 Abrupt 总面积。
  stat_d_year$area_km2 <- balanced_round_area(
    exact_areas = stat_d_year$area_km2_exact,
    target_total_km2 = abrupt_area_km2,
    unit_km2 = area_reporting_unit_km2
  )

  stat_d_year$proportion_percent <-
    stat_d_year$area_km2 / analysis_area_km2 * 100

  if (abrupt_area_km2 > 0) {
    stat_d_year$proportion_in_abrupt_percent <-
      stat_d_year$area_km2 / abrupt_area_km2 * 100
  } else {
    stat_d_year$proportion_in_abrupt_percent <- NA_real_
  }

  stat_d_year <- stat_d_year[
    order(stat_d_year$turning_year),
  ]

} else {

  stat_d_year <- data.frame(
    turning_year = numeric(),
    area_km2_exact = numeric(),
    area_km2 = numeric(),
    pixel_count = integer(),
    proportion_percent = numeric(),
    proportion_in_abrupt_percent = numeric()
  )
}

write.csv(
  stat_d_year,
  file.path(
    dir_d_report,
    "转折年份_逐年面积与比例统计_占统一有效分析范围.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# 自动输出转折面积最大的前 3 个年份，避免人工选错年份
if (nrow(stat_d_year) > 0) {
  stat_d_top3 <- head(
    stat_d_year[
      order(-stat_d_year$area_km2_exact),
    ],
    3
  )
} else {
  stat_d_top3 <- stat_d_year
}

write.csv(
  stat_d_top3,
  file.path(
    dir_d_report,
    "转折年份_面积排序前3位_占统一有效分析范围.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# 转折时段采用论文图件中的“边界式标签”。
# 注意：right = TRUE 表示左开右闭，因此边界年份不会重复统计：
# 1990-1995 对应 (1989, 1995]，实际包括 1990—1995 年；
# 1995-2000 对应 (1995, 2000]，实际包括 1996—2000 年；
# 2000-2005 对应 (2000, 2005]，实际包括 2001—2005 年；
# 2005-2010 对应 (2005, 2010]，实际包括 2006—2010 年；
# 2010-2015 对应 (2010, 2015]，实际包括 2011—2015 年。
period_breaks <- c(1989, 1995, 2000, 2005, 2010, 2015)

period_labels <- c(
  "1990-1995",
  "1995-2000",
  "2000-2005",
  "2005-2010",
  "2010-2015"
)

df_d$turning_period <- cut(
  df_d$turning_year,
  breaks = period_breaks,
  labels = period_labels,
  right = TRUE,
  include.lowest = TRUE
)

df_d_period <- df_d[
  !is.na(df_d$turning_period),
]

if (nrow(df_d_period) > 0) {

  stat_d_period <- aggregate(
    area_km2_exact ~ turning_period,
    data = df_d_period,
    FUN = sum,
    na.rm = TRUE
  )

  cnt_dp <- as.data.frame(
    table(df_d_period$turning_period)
  )
  names(cnt_dp) <- c(
    "turning_period",
    "pixel_count"
  )

  stat_d_period <- merge(
    stat_d_period,
    cnt_dp,
    by = "turning_period",
    all.x = TRUE
  )

  # 五个转折时段是 Abrupt 的进一步划分：整数面积严格闭合到 Abrupt 总面积。
  stat_d_period$area_km2 <- balanced_round_area(
    exact_areas = stat_d_period$area_km2_exact,
    target_total_km2 = abrupt_area_km2,
    unit_km2 = area_reporting_unit_km2
  )

  stat_d_period$proportion_percent <-
    stat_d_period$area_km2 / analysis_area_km2 * 100

  if (abrupt_area_km2 > 0) {
    stat_d_period$proportion_in_abrupt_percent <-
      stat_d_period$area_km2 / abrupt_area_km2 * 100
  } else {
    stat_d_period$proportion_in_abrupt_percent <- NA_real_
  }

  period_order <- period_labels

  stat_d_period$turning_period <- factor(
    stat_d_period$turning_period,
    levels = period_order
  )

  stat_d_period <- stat_d_period[
    order(stat_d_period$turning_period),
  ]

  stat_d_period$turning_period <- as.character(
    stat_d_period$turning_period
  )

} else {

  stat_d_period <- data.frame(
    turning_period = character(),
    area_km2_exact = numeric(),
    area_km2 = numeric(),
    pixel_count = integer(),
    proportion_percent = numeric(),
    proportion_in_abrupt_percent = numeric()
  )
}

write.csv(
  stat_d_period,
  file.path(
    dir_d_report,
    "转折年份_分段面积与比例统计_占统一有效分析范围.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)


# 输出边界式标签与实际纳入年份的对应关系，便于论文方法和图注核查
period_definition <- data.frame(
  display_label = period_labels,
  interval_rule = c(
    "(1989, 1995]",
    "(1995, 2000]",
    "(2000, 2005]",
    "(2005, 2010]",
    "(2010, 2015]"
  ),
  actual_integer_years = c(
    "1990-1995",
    "1996-2000",
    "2001-2005",
    "2006-2010",
    "2011-2015"
  ),
  boundary_year_repeated = rep(FALSE, length(period_labels)),
  stringsAsFactors = FALSE
)

write.csv(
  period_definition,
  file.path(
    dir_d_report,
    "转折时段_边界式标签与实际年份范围说明.csv"
  ),
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

df_diag <- df_diag[
  !is.na(df_diag$trend_type) &
    !is.na(df_diag$area_km2),
]

df_diag$linear_sig <- (
  !is.na(df_diag$linear_p) &
    df_diag$linear_p < alpha
)

df_diag$seg_sig_raw <- (
  !is.na(df_diag$seg_p_change) &
    df_diag$seg_p_change < alpha
)

df_diag$pettitt_sig_raw <- (
  !is.na(df_diag$pettitt_p) &
    df_diag$pettitt_p < alpha
)

df_diag$seg_valid_abrupt <- (
  df_diag$seg_sig_raw &
    !is.na(df_diag$seg_direction)
)

df_diag$pettitt_valid_abrupt <- (
  df_diag$pettitt_sig_raw &
    !is.na(df_diag$pettitt_direction) &
    !is.na(df_diag$pettitt_year) &
    df_diag$pettitt_year >= (start_year + min_gap) &
    df_diag$pettitt_year <= (end_year - min_gap)
)

df_diag$final_model_label <- ifelse(
  df_diag$final_model == 0,
  "No trend",
  ifelse(
    df_diag$final_model == 1,
    "Linear",
    ifelse(
      df_diag$final_model == 2,
      "Pettitt",
      ifelse(
        df_diag$final_model == 3,
        "Segmented",
        NA
      )
    )
  )
)

df_diag$trend_type_label <- ifelse(
  df_diag$trend_type == 0,
  "No trend",
  ifelse(
    df_diag$trend_type == 1,
    "Linear trend",
    ifelse(
      df_diag$trend_type == 2,
      "Abrupt / turning change",
      NA
    )
  )
)

df_diag$trend_subtype_label <- ifelse(
  df_diag$trend_subtype == 0,
  "No trend",
  ifelse(
    df_diag$trend_subtype == 11,
    "Final linear increase",
    ifelse(
      df_diag$trend_subtype == 12,
      "Final linear decrease",
      ifelse(
        df_diag$trend_subtype == 21,
        "I-D",
        ifelse(
          df_diag$trend_subtype == 22,
          "D-I",
          NA
        )
      )
    )
  )
)

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

diagnosis_table$Proportion_in_common_analysis_percent <-
  diagnosis_table$Area_km2 / analysis_area_km2_exact * 100

write.csv(
  diagnosis_table,
  file.path(
    dir_diag_report,
    "模型诊断_显著性筛选与最终分类面积统计_占统一有效分析范围.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# p 值、AIC 和斜率分组统计
summarise_num <- function(x) {

  x <- x[is.finite(x)]

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

  c(
    n = length(x),
    mean = mean(x),
    median = median(x),
    min = min(x),
    q25 = as.numeric(quantile(x, 0.25)),
    q75 = as.numeric(quantile(x, 0.75)),
    max = max(x)
  )
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
idx <- 1L

groups <- unique(df_diag$trend_subtype_label)
groups <- groups[!is.na(groups)]

for (g in groups) {

  sub <- df_diag[
    df_diag$trend_subtype_label == g,
  ]

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

    idx <- idx + 1L
  }
}

if (length(summary_list) > 0) {
  p_aic_summary <- do.call(rbind, summary_list)
} else {
  p_aic_summary <- data.frame()
}

write.csv(
  p_aic_summary,
  file.path(
    dir_diag_report,
    "模型诊断_p值_AIC_斜率_分组统计.csv"
  ),
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

model_count <- as.data.frame(
  table(df_diag$final_model_label)
)
names(model_count) <- c(
  "final_model_label",
  "pixel_count"
)

model_source <- merge(
  model_source,
  model_count,
  by = "final_model_label",
  all.x = TRUE
)

model_source$proportion_in_common_analysis_percent <-
  model_source$area_km2 / analysis_area_km2_exact * 100

write.csv(
  model_source,
  file.path(
    dir_diag_report,
    "模型诊断_最终模型来源统计_占统一有效分析范围.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 14. 综合表 =====================

get_prop <- function(stat_df, value_id) {
  x <- stat_df$proportion_percent[
    stat_df$value == value_id
  ]
  if (length(x) == 0 || is.na(x)) return(0)
  round(x, 2)
}

get_area <- function(stat_df, value_id) {
  x <- stat_df$area_km2[
    stat_df$value == value_id
  ]
  if (length(x) == 0 || is.na(x)) return(0)
  as.numeric(x)
}

get_period_prop <- function(stat_df, period_name) {
  x <- stat_df$proportion_percent[
    stat_df$turning_period == period_name
  ]
  if (length(x) == 0 || is.na(x)) return(0)
  round(x, 2)
}

get_period_area <- function(stat_df, period_name) {
  x <- stat_df$area_km2[
    stat_df$turning_period == period_name
  ]
  if (length(x) == 0 || is.na(x)) return(0)
  as.numeric(x)
}

summary_table_percent <- data.frame(
  Object = "Common valid grassland analysis range",

  Common_analysis_area_km2 = analysis_area_km2,

  Minimum_valid_years = min_valid_years,

  CV_lt20_percent = get_prop(stat_cv, 1),
  CV_20_30_percent = get_prop(stat_cv, 2),
  CV_gt30_percent = get_prop(stat_cv, 3),

  No_trend_percent = get_prop(stat_b, 0),
  Linear_trend_percent = get_prop(stat_b, 1),
  Abrupt_change_percent = get_prop(stat_b, 2),

  Final_linear_increase_percent =
    get_prop(stat_linear_final, 1),
  Final_linear_decrease_percent =
    get_prop(stat_linear_final, 2),

  I_D_percent = get_prop(stat_c, 1),
  D_I_percent = get_prop(stat_c, 2),

  Turning_1990_1995_percent =
    get_period_prop(stat_d_period, "1990-1995"),
  Turning_1995_2000_percent =
    get_period_prop(stat_d_period, "1995-2000"),
  Turning_2000_2005_percent =
    get_period_prop(stat_d_period, "2000-2005"),
  Turning_2005_2010_percent =
    get_period_prop(stat_d_period, "2005-2010"),
  Turning_2010_2015_percent =
    get_period_prop(stat_d_period, "2010-2015")
)

summary_table_area <- data.frame(
  Object = "Common valid grassland analysis range",

  Common_analysis_area_km2 = analysis_area_km2,

  Minimum_valid_years = min_valid_years,

  CV_lt20_area_km2 = get_area(stat_cv, 1),
  CV_20_30_area_km2 = get_area(stat_cv, 2),
  CV_gt30_area_km2 = get_area(stat_cv, 3),

  No_trend_area_km2 = get_area(stat_b, 0),
  Linear_trend_area_km2 = get_area(stat_b, 1),
  Abrupt_change_area_km2 = get_area(stat_b, 2),

  Final_linear_increase_area_km2 =
    get_area(stat_linear_final, 1),
  Final_linear_decrease_area_km2 =
    get_area(stat_linear_final, 2),

  I_D_area_km2 = get_area(stat_c, 1),
  D_I_area_km2 = get_area(stat_c, 2),

  Turning_1990_1995_area_km2 =
    get_period_area(stat_d_period, "1990-1995"),
  Turning_1995_2000_area_km2 =
    get_period_area(stat_d_period, "1995-2000"),
  Turning_2000_2005_area_km2 =
    get_period_area(stat_d_period, "2000-2005"),
  Turning_2005_2010_area_km2 =
    get_period_area(stat_d_period, "2005-2010"),
  Turning_2010_2015_area_km2 =
    get_period_area(stat_d_period, "2010-2015")
)

write.csv(
  summary_table_percent,
  file.path(
    report_root,
    "综合表_RI空间异质性统计_比例_占统一有效分析范围.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  summary_table_area,
  file.path(
    report_root,
    "综合表_RI空间异质性统计_面积_km2_统一有效分析范围.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# ===================== 15. 一致性检查 =====================

sum_area_by_value <- function(stat_df) {
  sum(stat_df$area_km2, na.rm = TRUE)
}

trend_area_sum <- sum_area_by_value(stat_b)
cv_area_sum <- sum_area_by_value(stat_cv)
subtype_area_sum <- sum_area_by_value(stat_subtype)

abrupt_area <- get_area(stat_b, 2)
id_area <- get_area(stat_c, 1)
di_area <- get_area(stat_c, 2)

linear_area <- get_area(stat_b, 1)
linear_inc_area <- get_area(stat_linear_final, 1)
linear_dec_area <- get_area(stat_linear_final, 2)

turning_year_area_sum <- sum(
  stat_d_year$area_km2,
  na.rm = TRUE
)

turning_period_area_sum <- sum(
  stat_d_period$area_km2,
  na.rm = TRUE
)

check_table <- data.frame(
  Check_item = c(
    "Trend classes sum versus common analysis area",
    "CV classes sum versus common analysis area",
    "Trend subtypes sum versus common analysis area",
    "Abrupt area versus I-D plus D-I",
    "Linear trend area versus final increase plus final decrease",
    "Abrupt area versus all turning-year areas",
    "Abrupt area versus all turning-period areas"
  ),
  Left_area_km2 = c(
    trend_area_sum,
    cv_area_sum,
    subtype_area_sum,
    abrupt_area,
    linear_area,
    abrupt_area,
    abrupt_area
  ),
  Right_area_km2 = c(
    analysis_area_km2,
    analysis_area_km2,
    analysis_area_km2,
    id_area + di_area,
    linear_inc_area + linear_dec_area,
    turning_year_area_sum,
    turning_period_area_sum
  )
)

check_table$Difference_km2 <-
  check_table$Left_area_km2 -
  check_table$Right_area_km2

check_table$Absolute_difference_km2 <-
  abs(check_table$Difference_km2)

check_table$Pass_exact_closure <-
  check_table$Absolute_difference_km2 < 1e-9

write.csv(
  check_table,
  file.path(
    dir_diag_report,
    "一致性检查_统一有效分析范围.csv"
  ),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# 若关键分类面积不一致，则给出警告
if (any(!check_table$Pass_exact_closure)) {
  warning(
    "部分一致性检查未通过，请查看：",
    file.path(
      dir_diag_report,
      "一致性检查_统一有效分析范围.csv"
    )
  )
}

# ===================== 16. 像元级综合报表 =====================

if (write_pixel_csv) {

  cat("\n正在输出像元级综合报表，像元较多时文件会比较大...\n")

  r_n_valid_common <- mask(
    r_n_valid,
    r_analysis_mask
  )

  df_all <- as.data.frame(
    c(
      r_n_valid_common,
      r_cv,
      r_cv_class,
      r_result,
      r_area_km2
    ),
    xy = TRUE,
    na.rm = FALSE
  )

  names(df_all) <- c(
    "x",
    "y",
    "valid_year_count",
    "CV_value",
    "CV_class",
    names(r_result),
    "area_km2"
  )

  df_all <- df_all[
    !is.na(df_all$trend_type) &
      !is.na(df_all$area_km2),
  ]

  df_all$CV_class_label <- ifelse(
    df_all$CV_class == 1,
    "<20%",
    ifelse(
      df_all$CV_class == 2,
      "20%-30%",
      ifelse(
        df_all$CV_class == 3,
        ">30%",
        NA
      )
    )
  )

  df_all$trend_type_label <- ifelse(
    df_all$trend_type == 0,
    "No trend",
    ifelse(
      df_all$trend_type == 1,
      "Linear trend",
      ifelse(
        df_all$trend_type == 2,
        "Abrupt / turning change",
        NA
      )
    )
  )

  # 原始一次线性回归方向，仅用于诊断
  df_all$linear_direction_raw_label <- ifelse(
    df_all$linear_direction_raw == 1,
    "Raw linear increase",
    ifelse(
      df_all$linear_direction_raw == 2,
      "Raw linear decrease",
      NA
    )
  )

  # 最终线性方向，仅适用于最终趋势类型为 Linear trend 的像元
  df_all$final_linear_direction_label <- ifelse(
    df_all$trend_subtype == 11,
    "Final linear increase",
    ifelse(
      df_all$trend_subtype == 12,
      "Final linear decrease",
      NA
    )
  )

  df_all$turning_direction_label <- ifelse(
    df_all$turning_direction == 1,
    "I-D",
    ifelse(
      df_all$turning_direction == 2,
      "D-I",
      NA
    )
  )

  df_all$final_model_label <- ifelse(
    df_all$final_model == 0,
    "No trend",
    ifelse(
      df_all$final_model == 1,
      "Linear",
      ifelse(
        df_all$final_model == 2,
        "Pettitt",
        ifelse(
          df_all$final_model == 3,
          "Segmented",
          NA
        )
      )
    )
  )

  df_all$trend_subtype_label <- ifelse(
    df_all$trend_subtype == 0,
    "No trend",
    ifelse(
      df_all$trend_subtype == 11,
      "Final linear increase",
      ifelse(
        df_all$trend_subtype == 12,
        "Final linear decrease",
        ifelse(
          df_all$trend_subtype == 21,
          "I-D",
          ifelse(
            df_all$trend_subtype == 22,
            "D-I",
            NA
          )
        )
      )
    )
  )

  write.csv(
    df_all,
    file.path(
      dir_all_report,
      "像元尺度_RI_CV_趋势类型_IDDI_转折年份_p值_AIC_统一有效范围.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
}

# ===================== 17. 输出提示 =====================

cat("\n全部完成！\n\n")
cat("本次运行批次号：", run_tag, "\n\n")

cat("本次栅格集中输出目录：\n", raster_root, "\n\n")
cat("统计报表固定输出路径：\n", report_root, "\n\n")
cat("本次栅格目录结构：\n")
cat("  01_趋势类型\n")
cat("  02_ID_DI转折类型\n")
cat("  03_转折年份\n")
cat("  04_最终线性趋势方向\n")
cat("  05_综合趋势子类型\n")
cat("  06_CV\n")
cat("  07_质控\n\n")

cat("统一有效分析条件：\n")
cat("1. 有效 RI 年份数不少于：", min_valid_years, "\n")
cat("2. RI 均值和标准差可计算；\n")
cat("3. RI 均值绝对值大于：", cv_mean_eps, "\n\n")

cat("统一有效分析面积（真实 km²）：", round(analysis_area_km2_exact, 4), "\n")
cat("统一有效分析面积（论文报告，1 km² 精度）：", analysis_area_km2, "\n")
cat("统一有效分析像元数：", analysis_pixel_count, "\n\n")

cat("图 b 趋势类型栅格：\n", out_b_trend_type, "\n")
cat("编码：0 = No trend；1 = Linear trend；2 = Abrupt / turning change\n\n")

cat("图 c I-D / D-I 栅格：\n", out_c_id_di, "\n")
cat("编码：1 = I-D；2 = D-I；NoData = 非最终突变像元\n\n")

cat("图 d 转折年份栅格：\n", out_d_year, "\n")
cat("编码：像元值即转折年份；NoData = 非最终突变像元\n\n")

cat("最终线性方向栅格：\n", out_linear_dir_final, "\n")
cat("编码：1 = Final linear increase；2 = Final linear decrease\n\n")

cat("综合趋势子类型栅格：\n", out_subtype, "\n")
cat("编码：0 = No trend；11 = Final linear increase；")
cat("12 = Final linear decrease；21 = I-D；22 = D-I\n\n")

cat("CV 原始值栅格：\n", out_cv_value, "\n")
cat("CV 分类栅格：\n", out_cv_class, "\n")
cat("编码：1 = CV<20%；2 = 20%≤CV≤30%；3 = CV>30%\n\n")

cat("有效年份数栅格：\n", out_valid_year_count, "\n")
cat("统一有效分析范围掩膜：\n", out_common_mask, "\n\n")

cat("CV 分类统计：\n")
print(stat_cv)

cat("\n最终趋势类型统计：\n")
print(stat_b)

cat("\nI-D / D-I 统计：\n")
print(stat_c)

cat("\n最终线性方向统计：\n")
print(stat_linear_final)

cat("\n综合趋势子类型统计：\n")
print(stat_subtype)

cat("\n转折年份分段统计：\n")
print(stat_d_period)

cat("\n转折时段边界式标签与实际年份范围：\n")
print(period_definition)

cat("\n转折面积最大的前 3 个年份：\n")
print(stat_d_top3)

cat("\n统一有效范围与分母检查：\n")
print(denominator_audit)

cat("\n一致性检查：\n")
print(check_table)

cat("\n说明：\n")
cat("1. CV 与全部趋势统计采用同一统一有效分析范围和同一面积分母。\n")
cat("2. 趋势类型、CV 分类及综合子类型的面积占比应分别合计为约 100%。\n")
cat("3. 最终线性增加面积与最终线性降低面积之和应等于最终线性趋势面积。\n")
cat("4. 转折时段沿用边界式标签，并按左开右闭规则划分，边界年份不重复统计。\n")
cat("   转折年份/时段报表同时输出占统一有效范围比例和占全部转折区域比例。\n")
cat("5. 面积内部由 cellSize() 按真实像元面积计算；论文表格统一报告到 1 km²。\n")
cat("6. 互斥分类采用总量约束取整，整数面积严格闭合，并据此重算比例。\n")
cat("7. 栅格每次运行自动建立 Run_时间戳 独立目录并集中保存，overwrite = FALSE；统计报表固定输出到原目录并更新同名 CSV。\n")
