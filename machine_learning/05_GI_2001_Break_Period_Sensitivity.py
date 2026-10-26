# -*- coding: utf-8 -*-
"""
06_GI_2001_Break_Period_Sensitivity.py
===============================================================
川西北草地韧性 RI：审稿人2 Comment 15 / 3.11
GI 原始空间分辨率 2001 年变化的敏感性检验（独立补充分析）

正式分析内容
------------
01. 逐年 GI 描述、2001 年相邻年份跳变、所有年度跃变的参考排名；
    基于 Block100 的空间块 bootstrap 95% CI（仅描述观测变化，不能认定分辨率造成变化）。
02. 对 1985–2000 和 2001–2020 两段分别拟合主分析实际胜出模型（通常 RF）；
    7 个生态因子 + Year；完全重复原有外层5折×5次、内层5折、patch_id1 分组；
    每个外层任务重新 Optuna-TPE 调参，沿用原 ml_common 里的正式搜索空间。
03. 只计算 GI 的 25 套外层测试折 ALE，不在训练折计算解释曲线；
    在两段 GI 的共同支持范围内对比，使用相同参考 GI 值重新定零点。
04. 保存分期 ALE 的均值、2.5–97.5百分位经验区间、模型间变异、
    高 GI 相对响应方向及可用时的全时期主分析 GI-ALE 作为参考。
05. 输出审稿人使用的关键结果表、高清 PNG/TIF 600 dpi、PDF/SVG、日志和断点。

重要科研说明
------------
* 代码只能检验已聚合20 km CSV 中2001年前后的不连续性和模型响应稳健性。
  单凭 CSV 无法将后期0.0025°栅格重新聚合为早期0.1°原始格网，也无法证明
  任何跃变一定由空间分辨率变化引起。实现该检验必须另提供逐年 GI 原始栅格。
* 两段模型各自重新调参；不同年份区间的 R² 因目标方差不同，不应简单视为因果差异。
* 两组 ALE 在相同 GI 参考点定零，只能比较共同数据支持范围内的响应形态；
  本脚本所有经验区间均不是独立的95%置信区间或生态阈值置信区间。
* 不覆盖 01–05 的任何文件，也不修改原输入 CSV。

安装依赖：与之前01–05相同的Python环境（含 numpy、pandas、scikit-learn、
scipy、optuna、matplotlib、openpyxl、shap、xgboost、lightgbm、joblib）；
使用同文件夹的 ml_common.py 或 ml_common(4).py。

使用：直接在 PyCharm 运行这个文件；不需要修改 Windows 输入/输出路径。
      也可执行 python 06_GI_2001_Break_Period_Sensitivity.py --self-test
      只做纯合成数据的本地函数测试（不读取真实数据）。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# -----------------------------------------------------------------------------
# 1. 本机路径。以下已按用户要求固定，无需调整。
# -----------------------------------------------------------------------------
CSV_FILE_PATH = Path(
    r"C:\Users\yihai\Desktop\川西北\机器学习建模数据\ABCD_加入DEM坡度均值_log处理\建模数据格网20_669完整格网_含Block100.csv"
)
OUTPUT_ROOT_FOLDER = Path(
    r"C:\Users\yihai\Desktop\川西北\机器学习建模数据\XGBOOST+SHAP最新20km"
)

# 与 03、04 的正式主分析路径一致。若该目录不存在，回退到 latest_main_run.txt。
EXPLICIT_MAIN_RUN_DIR = OUTPUT_ROOT_FOLDER / "Final_ML_Main_20260812_230042"
PREFER_EXPLICIT_MAIN_RUN = True
LATEST_MAIN_POINTER = OUTPUT_ROOT_FOLDER / "latest_main_run.txt"

RUN_PREFIX = "06_GI_2001_Break_Sensitivity"
ACTIVE_POINTER = "active_06_gi_break_run.txt"
LATEST_POINTER = "latest_06_gi_break_run.txt"

PERIODS = {"Early_1985_2000": (1985, 2000), "Late_2001_2020": (2001, 2020)}
CHANGE_YEAR = 2001
MODEL_INTERNAL_JOBS_FOR_ALE = 4
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 20261025
ALE_N_BINS = 20                 # 与原 01 研究一致
ALE_Q_RANGE = (0.02, 0.98)     # 与原 01 研究一致
COMMON_SUPPORT_QUANTILES = (0.05, 0.95)
COMMON_GRID_POINTS = 81
MIN_VALID_CURVES_FOR_SUMMARY = 10
FIG_DPI = 600

# 不默认增加第四种敏感性：02 已完成完整时期 No-Year 检验。
# 此项只考察论文正式八变量模型在2001年前后的 GI-ALE 一致性。


# -----------------------------------------------------------------------------
# 2. 导入用户上传版本的公共模块。不要求把 ml_common(4).py 手动改名。
# -----------------------------------------------------------------------------
def import_existing_common():
    # 优先使用本次用户上传的最新版(4)，避免同目录还留有旧 ml_common.py。
    base = Path(__file__).resolve().parent
    special = base / "ml_common(4).py"
    if special.exists():
        spec = importlib.util.spec_from_file_location("ml_common", special)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载用户公共模块：{special}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ml_common"] = module
        spec.loader.exec_module(module)
        return module
    try:
        import ml_common
        return ml_common
    except ModuleNotFoundError as exc:
        if exc.name != "ml_common":
            raise
    # 文件名含其他序号时，也可识别，例如 ml_common(2).py。
    for file in sorted(base.glob("ml_common(*).py"), reverse=True):
        spec = importlib.util.spec_from_file_location("ml_common", file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["ml_common"] = module
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError(
        "06 未找到 ml_common.py 或 ml_common(4).py。请把06与之前01所用的公共模块放在同一个目录。"
    )

mc = import_existing_common()
FEATURES = list(mc.MAIN_MODEL_FEATURES)
GROUP = mc.GROUP_COLUMN
YEAR = mc.YEAR_COLUMN
TARGET = mc.TARGET
GI = "GI"
BLOCK = mc.BLOCK100_COLUMN
SAMPLE = mc.SAMPLE_ID_COLUMN

plt.rcParams.update({
    "font.family": "Times New Roman", "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False, "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.labelsize": 11, "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none"
})
COLORS = {"Early_1985_2000": "#577EA3", "Late_2001_2020": "#BF7459"}


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def sha256(path: Path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", width=0.8, length=3.5)


def save_figure(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.10, "facecolor": "white"}
    fig.savefig(str(stem) + ".png", dpi=FIG_DPI, **kwargs)
    try:
        fig.savefig(str(stem) + ".tif", dpi=FIG_DPI, format="tiff",
                    pil_kwargs={"compression": "tiff_lzw"}, **kwargs)
    except Exception:
        fig.savefig(str(stem) + ".tif", dpi=FIG_DPI, format="tiff", **kwargs)
    fig.savefig(str(stem) + ".pdf", **kwargs)
    fig.savefig(str(stem) + ".svg", **kwargs)
    plt.close(fig)


def resolve_main_run() -> Path:
    if PREFER_EXPLICIT_MAIN_RUN and EXPLICIT_MAIN_RUN_DIR.exists():
        result = EXPLICIT_MAIN_RUN_DIR
    elif LATEST_MAIN_POINTER.exists():
        result = Path(LATEST_MAIN_POINTER.read_text(encoding="utf-8-sig").strip())
    else:
        raise FileNotFoundError(
            f"找不到原主分析 {EXPLICIT_MAIN_RUN_DIR}，也找不到 {LATEST_MAIN_POINTER}。"
        )
    if not result.is_dir():
        raise FileNotFoundError(f"主分析目录不存在：{result}")
    required = ["selected_model_for_sensitivity.json", "01_split_manifest/outer_split_assignment_by_observation.csv",
                "02_main_nested_cv/all_outer_scores.csv"]
    missing = [p for p in required if not (result / p).exists()]
    if missing:
        raise FileNotFoundError(f"主分析缺少必要文件：{missing}；目录：{result}")
    return result


def check_main_input_provenance(df: pd.DataFrame, main: Path, input_hash: str):
    """避免样本顺序变化后继续重用旧外层划分。"""
    manifest = pd.read_csv(main / "01_split_manifest" / "outer_split_assignment_by_observation.csv")
    for c in (SAMPLE, "repeat", "test_fold", "group_id"):
        if c not in manifest.columns:
            raise ValueError(f"原主分析 split manifest 缺少字段：{c}")
    subset = manifest[manifest["repeat"] == manifest["repeat"].min()].copy()
    group_map = df.set_index(SAMPLE)[GROUP].astype(str)
    if len(subset) != len(df) or subset[SAMPLE].nunique() != len(df):
        raise ValueError("主分析 split manifest 的样本数不匹配当前 CSV，不能重用划分。")
    actual = subset[SAMPLE].map(group_map)
    if actual.isna().any() or not (actual.to_numpy(str) == subset["group_id"].astype(str).to_numpy()).all():
        raise ValueError("主分析样本ID与当前格网ID不一致：数据可能已更换或排序改变，不能重用旧划分。")

    hash_file = main / "00_config_environment" / "file_hashes.csv"
    if hash_file.exists():
        hashes = pd.read_csv(hash_file)
        if {"label", "sha256"}.issubset(hashes.columns):
            old = hashes.loc[hashes["label"] == "input_csv", "sha256"]
            if len(old) and old.iloc[0] not in ("NOT_FOUND", "nan", ""):
                if old.iloc[0] != input_hash:
                    raise ValueError(
                        "当前 CSV SHA256 与原01主分析文件不同；为防止错配，已停止。"
                        "如果你修改过CSV，请使用对应新主分析和新外层划分。"
                    )
    return manifest


def make_run_directory(input_hash: str, main: Path, model_name: str) -> tuple[Path, bool]:
    OUTPUT_ROOT_FOLDER.mkdir(parents=True, exist_ok=True)
    active = OUTPUT_ROOT_FOLDER / ACTIVE_POINTER
    signature = {"input_sha256": input_hash, "main_run": str(main),
                 "model": model_name, "periods": PERIODS,
                 "ale_bins": ALE_N_BINS, "ale_quantiles": ALE_Q_RANGE,
                 "common_support_quantiles": COMMON_SUPPORT_QUANTILES,
                 "optuna_trials": int(mc.OPTUNA_TRIALS.get(model_name, 0))}
    if active.exists():
        old = Path(active.read_text(encoding="utf-8-sig").strip())
        conf = old / "00_config" / "run_signature.json"
        if old.exists() and not (old / "RUN_COMPLETE.flag").exists() and conf.exists():
            with open(conf, "r", encoding="utf-8") as f:
                if json.load(f) == json.loads(json.dumps(signature)):
                    return old, True
    while True:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = OUTPUT_ROOT_FOLDER / f"{RUN_PREFIX}_{stamp}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    save_json(signature, path / "00_config" / "run_signature.json")
    active.write_text(str(path), encoding="utf-8")
    return path, False


# -----------------------------------------------------------------------------
# 3. 2001 年 GI 断点及趋势诊断
# -----------------------------------------------------------------------------
def cluster_bootstrap_means(rows: pd.DataFrame, value_columns: list[str],
                            n_boot=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """重抽 100 km 空间块；每次纳入抽中块的全部格网，避免像元伪重复。"""
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby(BLOCK)[value_columns].agg(["sum", "count"])
    blocks = grouped.index.to_numpy()
    n_groups = len(blocks)
    if n_groups < 5:
        return pd.DataFrame([{"metric": c, "estimate": float(rows[c].mean()),
                              "CI_lower_2.5": np.nan, "CI_upper_97.5": np.nan,
                              "n_blocks": n_groups, "note": "<5 blocks, CI not estimated"}
                             for c in value_columns])
    out = []
    for col in value_columns:
        v_s = grouped[(col, "sum")].to_numpy(float)
        v_n = grouped[(col, "count")].to_numpy(float)
        # 所有变量用相同的簇重复重采样序列，便于复现。
        rg = np.random.default_rng(seed)
        idx = rg.integers(0, n_groups, size=(n_boot, n_groups))
        numerator = v_s[idx].sum(axis=1)
        denominator = v_n[idx].sum(axis=1)
        values = numerator / np.maximum(denominator, 1)
        out.append({"metric": col, "estimate": float(rows[col].mean()),
                    "CI_lower_2.5": float(np.percentile(values, 2.5)),
                    "CI_upper_97.5": float(np.percentile(values, 97.5)),
                    "n_blocks": n_groups, "bootstrap_B": n_boot,
                    "method": "100-km spatial-block cluster bootstrap"})
    return pd.DataFrame(out)


def gi_break_diagnostics(df: pd.DataFrame, out_dir: Path, fig_dir: Path, logger):
    early = df[(df[YEAR] >= 1985) & (df[YEAR] <= 2000)][GI].dropna().to_numpy(float)
    late = df[(df[YEAR] >= 2001) & (df[YEAR] <= 2020)][GI].dropna().to_numpy(float)
    rows = []
    for key, arr in [("Early_1985_2000", early), ("Late_2001_2020", late)]:
        if len(arr) < 2:
            raise ValueError(f"{key} GI有效观测数少于2。")
        rows.append({"period": key, "n": len(arr), "mean": np.mean(arr), "median": np.median(arr),
                     "SD": np.std(arr, ddof=1), "q02": np.quantile(arr, .02),
                     "q05": np.quantile(arr, .05), "q25": np.quantile(arr, .25),
                     "q75": np.quantile(arr, .75), "q95": np.quantile(arr, .95),
                     "q98": np.quantile(arr, .98), "minimum": np.min(arr),
                     "maximum": np.max(arr), "negative_fraction": float(np.mean(arr < 0))})
    write_csv(pd.DataFrame(rows), out_dir / "01_GI_distribution_by_period.csv")

    annual = df.groupby(YEAR, sort=True).agg(
        n_records=(GI, "size"), n_valid=(GI, "count"),
        mean_GI=(GI, "mean"), median_GI=(GI, "median"),
        sd_GI=(GI, "std"), q25_GI=(GI, lambda a: a.quantile(.25)),
        q75_GI=(GI, lambda a: a.quantile(.75))
    ).reset_index().sort_values(YEAR)
    annual["delta_annual_mean"] = annual["mean_GI"].diff()
    annual["abs_delta_annual_mean"] = annual["delta_annual_mean"].abs()
    annual["missing_fraction"] = 1 - annual["n_valid"] / annual["n_records"]
    write_csv(annual, out_dir / "02_GI_annual_mean_median.csv")

    piv = df.pivot(index=GROUP, columns=YEAR, values=GI)
    ranked = []
    years = sorted(int(y) for y in piv.columns if pd.notna(y))
    for y in years[1:]:
        if y - 1 not in piv.columns:
            continue
        two = piv[[y - 1, y]].dropna()
        if len(two) < 3:
            continue
        delta = two[y].to_numpy(float) - two[y-1].to_numpy(float)
        ranked.append({"transition": f"{y-1}-{y}", "end_year": y,
                       "n_grid_pairs": len(delta), "mean_signed_delta": float(np.mean(delta)),
                       "mean_abs_delta": float(np.mean(np.abs(delta))),
                       "median_abs_delta": float(np.median(np.abs(delta))),
                       "positive_fraction": float(np.mean(delta > 0))})
    all_changes = pd.DataFrame(ranked)
    if all_changes.empty or CHANGE_YEAR not in set(all_changes["end_year"]):
        raise ValueError("找不到 2000—2001 年的有效GI配对，不能做断点诊断。")
    all_changes["rank_mean_abs_delta_desc"] = all_changes["mean_abs_delta"].rank(
        method="min", ascending=False).astype(int)
    all_changes["rank_abs_mean_signed_delta_desc"] = all_changes["mean_signed_delta"].abs().rank(
        method="min", ascending=False).astype(int)
    write_csv(all_changes, out_dir / "03_GI_all_annual_jump_rankings.csv")

    needed = [1999, 2000, 2001, 2002]
    if not set(needed).issubset(set(piv.columns)):
        raise ValueError("断点配对分析需要1999、2000、2001、2002四个年度的GI值。")
    pair = piv[needed].dropna().copy()
    pair["pre_1999_2000"] = pair[2000] - pair[1999]
    pair["break_2000_2001"] = pair[2001] - pair[2000]
    pair["post_2001_2002"] = pair[2002] - pair[2001]
    pair["abs_break"] = pair["break_2000_2001"].abs()
    pair["neighbor_abs_mean"] = 0.5 * (pair["pre_1999_2000"].abs() + pair["post_2001_2002"].abs())
    pair["excess_abs_break"] = pair["abs_break"] - pair["neighbor_abs_mean"]
    blocks = df[[GROUP, BLOCK]].drop_duplicates(subset=GROUP).set_index(GROUP)[BLOCK]
    pair[BLOCK] = pair.index.map(blocks)
    pair.index.name = GROUP
    write_csv(pair.reset_index(), out_dir / "04_GI_patch_paired_1999_2002.csv")

    boot = cluster_bootstrap_means(pair.reset_index(),
                                   ["break_2000_2001", "abs_break", "neighbor_abs_mean", "excess_abs_break"])
    write_csv(boot, out_dir / "05_GI_100km_block_bootstrap_CI.csv")
    br = all_changes[all_changes["end_year"] == CHANGE_YEAR].iloc[0]
    summary = pd.DataFrame([{
        "transition": "2000-2001", "n_paired_grids_4years": len(pair),
        "mean_signed_delta_GI": float(pair["break_2000_2001"].mean()),
        "median_signed_delta_GI": float(pair["break_2000_2001"].median()),
        "mean_abs_break": float(pair["abs_break"].mean()),
        "mean_neighbor_abs_change": float(pair["neighbor_abs_mean"].mean()),
        "mean_excess_abs_break": float(pair["excess_abs_break"].mean()),
        "fraction_excess_abs_break_gt0": float((pair["excess_abs_break"] > 0).mean()),
        "2001_rank_mean_abs_change": int(br["rank_mean_abs_delta_desc"]),
        "2001_rank_abs_mean_signed_change": int(br["rank_abs_mean_signed_delta_desc"]),
        "n_annual_transitions": len(all_changes),
        "interpretation": "Descriptive discontinuity diagnostic, NOT attribution to resolution change"
    }])
    write_csv(summary, out_dir / "06_GI_break_key_summary.csv")

    # 作为附加诊断的分段时间趋势（HAC标准误；不将统计显著等同于数据伪影）。
    try:
        import statsmodels.api as sm
        reg = annual.dropna(subset=["mean_GI"]).copy()
        time = reg[YEAR].to_numpy(float)
        post = (time >= CHANGE_YEAR).astype(float)
        X = pd.DataFrame({"const": np.ones(len(time)), "Year_minus_2000": time-2000,
                          "Post2001": post, "Year_after_2001": post * (time-2001)})
        result = sm.OLS(reg["mean_GI"].to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        ci = result.conf_int()
        reg_table = pd.DataFrame({"term": X.columns, "coefficient": result.params.values,
                                  "HAC_SE": result.bse.values, "z": result.tvalues.values,
                                  "HAC_p": result.pvalues.values,
                                  "CI_lower": ci.iloc[:, 0].values, "CI_upper": ci.iloc[:, 1].values,
                                  "maxlags": 3})
        write_csv(reg_table, out_dir / "07_GI_annual_segmented_regression_HAC_exploratory.csv")
    except ImportError:
        logger.warning("statsmodels不可用，跳过HAC分段回归；其他分析不受影响。")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(annual[YEAR], annual["mean_GI"], color="#416B87", marker="o", ms=3,
            lw=1.7, label="Annual mean GI")
    ax.plot(annual[YEAR], annual["median_GI"], color="#B37B60", lw=1.4,
            ls="--", label="Annual median GI")
    ax.axvline(CHANGE_YEAR, color="#B44F47", lw=1.2, ls="--", label="2001 boundary")
    ax.set(xlabel="Year", ylabel="GI (SU/ha)")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax); fig.tight_layout()
    save_figure(fig, fig_dir / "Fig_06A_GI_annual_mean_median")

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(all_changes["end_year"], all_changes["mean_abs_delta"],
            lw=1.5, marker="o", ms=3, color="#547C97")
    ax.scatter([CHANGE_YEAR], [float(br["mean_abs_delta"])], color="#C0504D",
               zorder=5, s=38, label="2000–2001")
    ax.axvline(CHANGE_YEAR, color="#C0504D", ls="--", lw=1)
    ax.set(xlabel="End year of adjacent-year pair", ylabel="Mean absolute grid-level GI change (SU/ha)")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax); fig.tight_layout()
    save_figure(fig, fig_dir / "Fig_06B_GI_annual_jump_ranking")

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    qlow, qhigh = np.quantile(np.r_[early, late], [.01, .99])
    for period, arr in [("Early_1985_2000", early), ("Late_2001_2020", late)]:
        a = arr[(arr >= qlow) & (arr <= qhigh)]
        ax.hist(a, bins=60, density=True, alpha=.33, histtype="stepfilled",
                color=COLORS[period], label=period.replace("_", " "))
        ax.hist(a, bins=60, density=True, histtype="step", lw=1.5,
                color=COLORS[period])
    ax.set(xlabel="GI (SU/ha)", ylabel="Density")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax); fig.tight_layout()
    save_figure(fig, fig_dir / "Fig_06C_GI_period_distributions_1to99_percentiles")

    logger.info("2000–2001 GI网格平均绝对变化排名：%s/%s",
                int(br["rank_mean_abs_delta_desc"]), len(all_changes))
    return annual, all_changes, summary, boot


# -----------------------------------------------------------------------------
# 4. 时段分层的嵌套分组CV；原主分析25套patch_id1分组不变
# -----------------------------------------------------------------------------
def get_period_data_and_tasks(df: pd.DataFrame, assignment: pd.DataFrame,
                              period: str, limits: tuple[int, int]):
    lo, hi = limits
    part = df[(df[YEAR] >= lo) & (df[YEAR] <= hi)].copy().reset_index(drop=True)
    if len(part) == 0 or part[GROUP].nunique() < 10:
        raise ValueError(f"{period} 样本或格网数不足：{len(part)}行，{part[GROUP].nunique()}组")
    if part[GI].count() < 60:
        raise ValueError(f"{period} GI数据不足，不能可靠计算ALE。")
    sub_manifest = assignment[assignment[SAMPLE].isin(part[SAMPLE])].copy()
    expected = len(part) * int(mc.N_REPEATS)
    if len(sub_manifest) != expected:
        raise ValueError(f"{period} 外层manifest不完整：{len(sub_manifest)} vs {expected}")
    tasks = mc.load_outer_tasks_from_assignment(part, sub_manifest, GROUP)
    if len(tasks) != int(mc.N_REPEATS * mc.OUTER_N_SPLITS):
        raise RuntimeError(f"{period} 嵌套CV外层任务数量异常：{len(tasks)}")
    checks = mc.validate_outer_tasks(tasks, part, GROUP)
    if checks["overlap_groups"].sum() != 0:
        raise ValueError(f"{period} 外层出现格网泄漏")
    for repeat in sorted(sub_manifest["repeat"].unique()):
        actual = sub_manifest[sub_manifest["repeat"] == repeat]
        if len(actual) != len(part) or actual[SAMPLE].nunique() != len(part):
            raise RuntimeError(f"{period} repeat={repeat} 未覆盖每条观测一次")
    return part, tasks, sub_manifest, checks


def run_period_nested_cv(part, tasks, period, model, run_dir, logger):
    folder = run_dir / "02_period_nested_cv" / period
    results = mc.run_repeated_nested_cv(
        df=part, features=FEATURES, group_column=GROUP,
        outer_tasks=tasks, model_names=[model],
        output_folder=folder / "nested_cv",
        checkpoint_folder=run_dir / "06_logs_checkpoints" / period,
        sqlite_db=run_dir / "06_logs_checkpoints" / f"optuna_{period}.sqlite3",
        study_prefix=f"06_GIBREAK_{period}_{model}",
        logger=logger,
        progress_label=f"06 {period} | NestedCV"
    )
    expected = len(tasks)
    if len(results) != expected or results["Model"].ne(model).any():
        raise RuntimeError(f"{period} 外层模型结果数量/名称异常")
    summary = mc.summarize_model_performance(results)
    write_csv(summary, folder / "model_performance_mean_SD.csv")
    mc.export_best_params(results, folder, f"{period}_Best_Hyperparameters_25_outer_tasks")
    return results


# -----------------------------------------------------------------------------
# 5. 可比 ALE 支持范围与折内检验；不在没有数据的范围外插
# -----------------------------------------------------------------------------
def derive_common_gi_domain(df: pd.DataFrame) -> dict[str, float | bool]:
    arrays = {}
    for name, (a, b) in PERIODS.items():
        arrays[name] = df.loc[df[YEAR].between(a, b), GI].dropna().to_numpy(float)
    qlo, qhi = COMMON_SUPPORT_QUANTILES
    lo = max(np.quantile(x, qlo) for x in arrays.values())
    hi = min(np.quantile(x, qhi) for x in arrays.values())
    if hi <= lo or not np.isfinite([lo, hi]).all():
        return {"valid": False, "lower": float(lo), "upper": float(hi),
                "reference": np.nan, "reason": "No overlapping 5th–95th percentile GI support"}
    pooled = np.concatenate([x[(x >= lo) & (x <= hi)] for x in arrays.values()])
    ref = float(np.median(pooled))
    if not (lo < ref < hi):
        ref = 0.5 * (float(lo) + float(hi))
    return {"valid": True, "lower": float(lo), "upper": float(hi),
            "reference": ref, "contains_published_peak_0p39": bool(lo <= 0.39 <= hi),
            "reason": "Intersection of period-specific 5th–95th percentiles"}


def run_period_ale(part, tasks, scores, period, model_name, run_dir,
                   domain: dict, logger):
    base = run_dir / "03_GI_ALE" / period
    fold_dir = base / "fold_curves"
    fold_dir.mkdir(parents=True, exist_ok=True)
    results_lookup = scores.set_index(["Repeat", "Outer_Fold"])
    common_grid = (np.linspace(domain["lower"], domain["upper"], COMMON_GRID_POINTS)
                   if domain["valid"] else np.array([], dtype=float))
    finished = 0
    for task in tasks:
        r, f, seed = int(task["repeat"]), int(task["fold"]), int(task["seed"])
        raw_out = fold_dir / f"R{r:02d}_F{f:02d}_raw.csv"
        common_out = fold_dir / f"R{r:02d}_F{f:02d}_common_reference.csv"
        flag = fold_dir / f"R{r:02d}_F{f:02d}.done"
        if flag.exists() and raw_out.exists() and common_out.exists():
            finished += 1
            continue
        row = results_lookup.loc[(r, f)]
        params = json.loads(row["Best_Params_JSON"])
        n_est = row.get("XGB_Final_n_estimators", np.nan)
        n_est = None if pd.isna(n_est) else int(n_est)
        Xtr = part[FEATURES].iloc[task["train_idx"]].copy()
        Xte = part[FEATURES].iloc[task["test_idx"]].copy()
        ytr = part[TARGET].iloc[task["train_idx"]]
        mdl = mc.fit_outer_model(model_name, params, n_est, Xtr, ytr,
                                 seed + r * 100 + f, FEATURES,
                                 model_n_jobs=MODEL_INTERNAL_JOBS_FOR_ALE)
        raw = mc.calculate_ale_1d(mdl, Xte, GI, n_bins=ALE_N_BINS, q_range=ALE_Q_RANGE)
        if raw.empty:
            raise RuntimeError(f"{period} R{r}F{f} GI-ALE 为空，检查样本取值与缺失情况")
        raw.insert(0, "period", period)
        raw.insert(1, "repeat", r)
        raw.insert(2, "fold", f)
        write_csv(raw, raw_out)
        common = pd.DataFrame(columns=["period", "repeat", "fold", "x", "ALE_refcentered"])
        if domain["valid"]:
            xx = raw["x"].to_numpy(float)
            yy = raw["ALE"].to_numpy(float)
            ref = domain["reference"]
            if xx.min() <= ref <= xx.max():
                baseline = float(np.interp(ref, xx, yy))
                xx2 = common_grid[(common_grid >= xx.min()) & (common_grid <= xx.max())]
                y2 = np.interp(xx2, xx, yy) - baseline
                common = pd.DataFrame({"period": period, "repeat": r, "fold": f,
                                       "x": xx2, "ALE_refcentered": y2})
            else:
                logger.warning("%s R%dF%d 无法在共同 GI 参考点插值，保留原始曲线并跳过跨期对照", period, r, f)
        write_csv(common, common_out)
        flag.write_text(datetime.now().isoformat(), encoding="utf-8")
        del mdl
        gc.collect()
        finished += 1
        logger.info("GI ALE: %s %d/%d 折完成", period, finished, len(tasks))

    raw = pd.concat([pd.read_csv(f) for f in sorted(fold_dir.glob("*_raw.csv"))], ignore_index=True)
    cmn_files = sorted(fold_dir.glob("*_common_reference.csv"))
    common = pd.concat([pd.read_csv(f) for f in cmn_files], ignore_index=True)
    write_csv(raw, base / "GI_ALE_25_outer_curves_raw.csv")
    write_csv(common, base / "GI_ALE_25_outer_curves_common_reference.csv")
    summary_rows = []
    if not common.empty:
        for xval, g in common.groupby("x", sort=True):
            yy = g["ALE_refcentered"].dropna().to_numpy(float)
            if not len(yy):
                continue
            summary_rows.append({"period": period, "x": float(xval),
                                 "mean_ALE": float(np.mean(yy)),
                                 "median_ALE": float(np.median(yy)),
                                 "q2.5": float(np.quantile(yy, .025)),
                                 "q97.5": float(np.quantile(yy, .975)),
                                 "SD_across_outer_models": float(np.std(yy, ddof=1)) if len(yy)>1 else np.nan,
                                 "n_curves": len(yy)})
    summ = pd.DataFrame(summary_rows, columns=["period", "x", "mean_ALE", "median_ALE", "q2.5",
                                                "q97.5", "SD_across_outer_models", "n_curves"])
    write_csv(summ, base / "GI_ALE_mean_2.5_97.5_empirical_intervals.csv")
    return common, summ


def find_descriptive_local_peak(x: np.ndarray, y: np.ndarray):
    """仅描述可视 ALE 局部峰值；不能将其解释为放牧生态最适阈值。"""
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]; y = np.asarray(y)[mask]
    if len(x) < 9:
        return np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]
    window = min(11, len(y) if len(y) % 2 else len(y)-1)
    smooth = savgol_filter(y, window, 2) if window >= 5 else y
    amplitude = float(np.ptp(smooth))
    if amplitude <= 1e-12:
        return np.nan
    peaks, props = find_peaks(smooth, prominence=max(1e-9, 0.08 * amplitude))
    peaks = [p for p in peaks if 0.10 < p/(len(x)-1) < 0.90]
    if not peaks:
        return np.nan
    j = max(peaks, key=lambda k: smooth[k])
    return float(x[j])


def summarize_raw_period_peaks(raw_curves: pd.DataFrame, output_dir: Path):
    """各时期本身的 GI 支持范围内的峰值，和共同支持范围的峰值分别报告。"""
    out=[]
    for (period,r,f), part in raw_curves.groupby(["period","repeat","fold"]):
        part=part.sort_values("x")
        xx=part["x"].to_numpy(float); yy=part["ALE"].to_numpy(float)
        out.append({"period":period,"repeat":int(r),"fold":int(f),
                    "period_fold_GI_min":float(np.min(xx)),
                    "period_fold_GI_max":float(np.max(xx)),
                    "published_0p39_inside_own_support":bool(np.min(xx)<=.39<=np.max(xx)),
                    "local_peak_period_own_support":find_descriptive_local_peak(xx,yy)})
    result=pd.DataFrame(out)
    write_csv(result,output_dir/"07_GI_ALE_peaks_in_each_period_own_support.csv")
    return result


def summarize_ale_period_features(common: pd.DataFrame, domain: dict, output_dir: Path):
    rows=[]
    if domain["valid"] and not common.empty:
        lo, hi = domain["lower"], domain["upper"]
        upper_start = lo + 0.75*(hi-lo)
        for (period,r,f), g in common.groupby(["period", "repeat", "fold"]):
            g = g.sort_values("x")
            x, y = g["x"].to_numpy(float), g["ALE_refcentered"].to_numpy(float)
            peak = find_descriptive_local_peak(x,y)
            high_mask = (x >= upper_start) & np.isfinite(y)
            high_change = (float(y[high_mask][-1] - y[high_mask][0])
                           if high_mask.sum()>=3 else np.nan)
            rows.append({"period":period,"repeat":r,"fold":f,
                         "local_peak_GI_SU_per_ha":peak,
                         "upper25pct_GI_ALE_change":high_change,
                         "high_GI_direction": "increasing" if high_change>1e-10 else (
                             "decreasing" if high_change < -1e-10 else "flat_or_missing")
                         if np.isfinite(high_change) else "unsupported",
                         "n_grid_points":len(x),"common_domain_lower":lo,"common_domain_upper":hi})
    detail = pd.DataFrame(rows, columns=["period","repeat","fold","local_peak_GI_SU_per_ha",
                                         "upper25pct_GI_ALE_change","high_GI_direction","n_grid_points",
                                         "common_domain_lower","common_domain_upper"])
    write_csv(detail, output_dir / "01_GI_ALE_peak_and_highGI_each_outer_model.csv")
    summary = []
    for period, g in detail.groupby("period"):
        peaks = g["local_peak_GI_SU_per_ha"].dropna()
        hchange = g["upper25pct_GI_ALE_change"].dropna()
        summary.append({"period":period,"n_valid_reference_curves":len(g),
                        "n_local_peak_detected":len(peaks),
                        "fraction_local_peak_detected":len(peaks)/len(g) if len(g) else np.nan,
                        "local_peak_median":float(peaks.median()) if len(peaks) else np.nan,
                        "local_peak_Q25":float(peaks.quantile(.25)) if len(peaks) else np.nan,
                        "local_peak_Q75":float(peaks.quantile(.75)) if len(peaks) else np.nan,
                        "n_highGI_slopes_supported":len(hchange),
                        "fraction_highGI_decreasing":float((hchange<0).mean()) if len(hchange) else np.nan,
                        "median_highGI_ALE_change":float(hchange.median()) if len(hchange) else np.nan})
    summed = pd.DataFrame(summary)
    write_csv(summed, output_dir / "02_GI_ALE_period_feature_summary.csv")
    return detail, summed


def compare_ale_periods(early: pd.DataFrame, late: pd.DataFrame,
                        d_early: pd.DataFrame, d_late: pd.DataFrame,
                        out_dir: Path):
    cols=["x", "mean_ALE", "n_curves"]
    if early.empty or late.empty:
        write_csv(pd.DataFrame(columns=["x","early_mean","late_mean","late_minus_early"]),
                  out_dir / "03_GI_ALE_period_curve_difference.csv")
        return pd.DataFrame(),pd.DataFrame()
    merged=early[cols].merge(late[cols],on="x",how="inner",suffixes=("_early","_late"))
    merged["late_minus_early"] = merged["mean_ALE_late"]-merged["mean_ALE_early"]
    write_csv(merged, out_dir / "03_GI_ALE_period_curve_difference.csv")
    d1 = d_early.rename(columns={"ALE_refcentered":"early_ALE"})
    d2 = d_late.rename(columns={"ALE_refcentered":"late_ALE"})
    paired = d1.merge(d2, on=["repeat","fold","x"],how="inner")
    if not paired.empty:
        paired["late_minus_early"] = paired["late_ALE"]-paired["early_ALE"]
    write_csv(paired, out_dir / "04_GI_ALE_foldwise_paired_differences.csv")
    qualified = merged[(merged["n_curves_early"] >= MIN_VALID_CURVES_FOR_SUMMARY) &
                       (merged["n_curves_late"] >= MIN_VALID_CURVES_FOR_SUMMARY)]
    shape_rho = np.nan
    if len(qualified)>=7:
        shape_rho = float(stats.spearmanr(qualified["mean_ALE_early"], qualified["mean_ALE_late"]).statistic)
    result = pd.DataFrame([{
        "n_common_grid_points":len(merged), "n_adequately_supported_points":len(qualified),
        "n_curves_required_each_period":MIN_VALID_CURVES_FOR_SUMMARY,
        "descriptive_Spearman_shape_rho":shape_rho,
        "mean_abs_difference_over_qualified_points":float(qualified["late_minus_early"].abs().mean())
                                                if len(qualified) else np.nan,
        "comparison_note":"Shared GI reference and overlapping data support; rho is descriptive, not a statistical p-value",
    }])
    write_csv(result, out_dir / "05_GI_ALE_shape_similarity_summary.csv")
    return merged, paired


# -----------------------------------------------------------------------------
# 6. 图件。每条曲线所示区间是25套外层模型经验分散度，并非95%置信区间。
# -----------------------------------------------------------------------------
def plot_ale_compare(early: pd.DataFrame, late: pd.DataFrame, domain: dict,
                     fig_dir: Path, main_run: Path, logger):
    if not domain["valid"] or early.empty or late.empty:
        logger.warning("没有共同 GI 支持范围，无法绘制分期 GI-ALE 同轴对照图。")
        return
    fig,ax=plt.subplots(figsize=(7.4,4.7))
    for name, dat in [("Early_1985_2000", early),("Late_2001_2020", late)]:
        dat=dat[dat["n_curves"]>=MIN_VALID_CURVES_FOR_SUMMARY].sort_values("x")
        if dat.empty:
            logger.warning("%s 没有达到最少%d条折内ALE支持的点，曲线不绘制。",
                           name,MIN_VALID_CURVES_FOR_SUMMARY)
            continue
        x=dat["x"].to_numpy(float)
        y=dat["mean_ALE"].to_numpy(float)
        ax.plot(x,y,lw=2, color=COLORS[name], label=name.replace("_", " "))
        ax.fill_between(x,dat["q2.5"].to_numpy(float),dat["q97.5"].to_numpy(float),
                        color=COLORS[name],alpha=.17,lw=0)
    ax.axhline(0,color="0.5",ls="--",lw=.9)
    ax.axvline(domain["reference"],color="0.5",ls=":",lw=.9,label="Shared ALE reference")
    ax.set(xlabel="GI (SU/ha)",ylabel="GI ALE effect on predicted RI")
    ax.legend(frameon=False,fontsize=8)
    style_axis(ax); fig.tight_layout()
    save_figure(fig,fig_dir / "Fig_06D_GI_ALE_1985_2000_vs_2001_2020_common_support")

    # 可选叠加原1985–2020主分析均值曲线，仅供参考：主分析分箱采用自身分位数。
    orig=None
    for p in (main_run/"05_ALE_Top6"/"ALE_summary_95_empirical_interval.csv",
              main_run/"05_ALE"/"ALE_summary_95_empirical_interval.csv"):
        if p.exists():
            cand=pd.read_csv(p)
            if "feature" in cand.columns and "GI" in set(cand["feature"].astype(str)):
                orig=cand[cand["feature"].astype(str)=="GI"].dropna(subset=["x","mean_ALE"]).sort_values("x")
                break
    if orig is None or len(orig)<3:
        logger.info("主分析 GI-ALE 表未找到，跳过全时期参考曲线叠加。")
        return
    xo=orig["x"].to_numpy(float)
    yo=orig["mean_ALE"].to_numpy(float)
    ref=domain["reference"]
    if ref < xo.min() or ref > xo.max():
        logger.warning("原主分析 ALE 不覆盖共同参考点，跳过主分析叠加。")
        return
    xdom=np.linspace(domain["lower"],domain["upper"],COMMON_GRID_POINTS)
    xdom=xdom[(xdom>=xo.min())&(xdom<=xo.max())]
    ydom=np.interp(xdom,xo,yo)-np.interp(ref,xo,yo)
    fig,ax=plt.subplots(figsize=(7.4,4.7))
    ax.plot(xdom,ydom,color="0.25",lw=1.8,ls=":",label="Main model 1985–2020 (reference)")
    for name,dat in [("Early_1985_2000",early),("Late_2001_2020",late)]:
        dat=dat[dat["n_curves"]>=MIN_VALID_CURVES_FOR_SUMMARY].sort_values("x")
        if not dat.empty:
            ax.plot(dat["x"],dat["mean_ALE"],color=COLORS[name],lw=2,label=name.replace("_"," "))
    ax.axhline(0,color="0.55",ls="--",lw=.85)
    ax.axvline(ref,color="0.55",ls=":",lw=.85)
    ax.set(xlabel="GI (SU/ha)",ylabel="GI ALE effect on predicted RI")
    ax.legend(frameon=False,fontsize=8)
    style_axis(ax);fig.tight_layout()
    save_figure(fig,fig_dir / "Fig_06E_GI_ALE_periods_plus_main_reference")


def plot_ale_differences(paired: pd.DataFrame, fig_dir: Path):
    if paired.empty: return
    dd=paired.groupby("x")["late_minus_early"].agg(
        mean="mean",p025=lambda v: v.quantile(.025),p975=lambda v:v.quantile(.975), n="count"
    ).reset_index()
    write_csv(dd,fig_dir.parent / "05_reviewer_summary" / "06_GI_ALE_paired_fold_difference_empirical_band.csv")
    dd=dd[dd["n"]>=MIN_VALID_CURVES_FOR_SUMMARY].copy()
    if dd.empty:
        return
    fig,ax=plt.subplots(figsize=(7.0,4.3))
    ax.fill_between(dd["x"].to_numpy(float),dd["p025"].to_numpy(float),
                    dd["p975"].to_numpy(float),color="#CAB1A5",alpha=.3,lw=0)
    ax.plot(dd["x"].to_numpy(float),dd["mean"].to_numpy(float),color="#A45A43",lw=1.9)
    ax.axhline(0,color="0.4",ls="--",lw=.9)
    ax.set(xlabel="GI (SU/ha)",ylabel="ALE difference: 2001–2020 minus 1985–2000")
    style_axis(ax);fig.tight_layout()
    save_figure(fig,fig_dir / "Fig_06F_GI_ALE_late_minus_early_fold_empirical_band")


def write_readme(run_dir: Path, main_run: Path, domain: dict, model: str):
    text = f"""审稿人2第15条（原3.11）GI 2001年分辨率变化：输出说明
==================================================
原数据：{CSV_FILE_PATH}
关联01主分析：{main_run}
实际建模方法：{model}（与主分析最优模型一致）
输入变量：{', '.join(FEATURES)}
分组：{GROUP}；外层5折重复5次，内层5折；种子42–46；原01外层划分原样沿用。
两时段：1985–2000（原GI分辨率0.1°）、2001–2020（原GI分辨率0.0025°）。
每期25套外层测试任务单独训练、折内中位数填补、重新调参。
GI ALE设置：分位数范围2–98%，20个分箱，先按各折测试集GI计算。
统一比较范围：{domain}
把每条曲线通过相同GI参考点进行垂直平移，再比较共同支持范围的形状。
分期ALE带=25个外层模型2.5–97.5经验分散区间，不是严格统计95%置信区间。
两期模型测试集R²不应简单视为同一分布下的严格性能优劣。

01_GI_discontinuity：
  GI逐年均值/中位数、2000–2001跃变与所有年度跃变排名、
  1999–2002格网成对变化、100km空间块bootstrap、辅助HAC分段回归。
02_period_nested_cv：
  每期25个嵌套CV结果及独立Optuna最优超参数；可以断点续跑。
03_GI_ALE：
  每折原始GI ALE、统一参考点ALE、两时期均值和经验区间。
04_figures：
  高分辨率PNG/TIFF和矢量PDF/SVG；适合用作补充材料。
05_reviewer_summary：
  两期性能表、ALE形态比较、每个外层折GI局部峰值与高值段趋势。
06_logs_checkpoints：
  Optuna SQLite、checkpoint、运行日志。

严禁将观察到的GI跃变直接解释成分辨率变化引起的误差。
严禁将ALE局部峰值当成因果阈值、最适载畜量。
如果需要直接检验0.1°尺度统一重采样，需要2001年后每年的原始0.0025°GI栅格。
本06未执行这种原始栅格的分辨率统一，不能在论文中声称已经执行。
"""
    (run_dir / "README_06_GI_2001_审稿分析说明.txt").write_text(text, encoding="utf-8")


def main():
    start=datetime.now()
    if not CSV_FILE_PATH.is_file():
        raise FileNotFoundError(f"Windows原始CSV不存在：{CSV_FILE_PATH}")
    main_run=resolve_main_run()
    df=mc.load_model_data(CSV_FILE_PATH, FEATURES, require_block100=True)
    year_values=set(df[YEAR].astype(int))
    expected=set(range(1985,2021))
    if not expected.issubset(year_values):
        raise ValueError(f"1985—2020年份不完整，缺少{sorted(expected-year_values)}")
    if df[GROUP].nunique()!=669 or len(df)!=24084:
        print(f"注意：实际数据为{len(df)}行/{df[GROUP].nunique()}个格网，非预期24084/669，请检查。")
    neg=int((df[GI]<0).sum())
    if neg:
        print(f"警告：GI发现{neg}个负值。若GI的单位为SU/ha，请核查这些值及是否曾做对数变换。")

    inp_hash=sha256(CSV_FILE_PATH)
    manifest=check_main_input_provenance(df,main_run,inp_hash)
    selection=json.loads((main_run/"selected_model_for_sensitivity.json").read_text(encoding="utf-8-sig"))
    selected=str(selection["selected_model"])
    scores_main=pd.read_csv(main_run/"02_main_nested_cv"/"all_outer_scores.csv")
    if len(scores_main[scores_main["Model"]==selected])!=int(mc.N_REPEATS*mc.OUTER_N_SPLITS):
        raise ValueError("原主分析最终模型25个外层测试结果不完整。")

    run_dir,resumed=make_run_directory(inp_hash,main_run,selected)
    logdir=run_dir/"06_logs_checkpoints";logdir.mkdir(parents=True,exist_ok=True)
    logger=mc.setup_logger(logdir/"06_GI_2001_sensitivity.log",name="NW_Sichuan_06_GI")
    logger.info("06 GO | 原主分析:%s | 最优模型:%s | 断点续跑:%s",main_run,selected,resumed)
    logger.info("变量:%s | 分组:%s | 内层5折/外层5折重复5次",FEATURES,GROUP)
    if selected!="RF":
        logger.warning("正式主分析胜出模型是 %s 而非 RF；06 自动采用该真实最优模型，避免偷偷更换模型。",selected)
    mc.export_software_versions(run_dir/"00_config")
    mc.export_file_hashes({"input_csv":CSV_FILE_PATH,"main_selection":main_run/"selected_model_for_sensitivity.json",
                           "06_script":Path(__file__), "common_module":Path(mc.__file__)},run_dir/"00_config")
    save_json({"run_start":start.isoformat(),"main_run":str(main_run),"model":selected,
               "n_grid":df[GROUP].nunique(),"n_rows":len(df),"features":FEATURES,
               "split_design":"same 25 outer splits of original 01; inner 5-fold by patch_id1",
               "original_module":str(mc.__file__),
               "spatial_resolution_note":"0.1 degrees through 2000, 0.0025 degrees from 2001"},
              run_dir/"00_config"/"run_description.json")

    figdir=run_dir/"04_figures"
    diagdir=run_dir/"01_GI_discontinuity"
    summdir=run_dir/"05_reviewer_summary"
    summdir.mkdir(parents=True,exist_ok=True)
    logger.info("开始GI逐年断点、空间块bootstrap诊断。")
    annual,jumps,break_result,boot=gi_break_diagnostics(df,diagdir,figdir,logger)
    domain=derive_common_gi_domain(df)
    save_json(domain,summdir/"00_common_GI_support_and_reference.json")
    logger.info("共同GI范围:%s",domain)
    if not domain["valid"]:
        logger.warning("无充分的共同GI分布支持，仍将运行分期模型，但禁止虚假跨期ALE外推。")

    period_scores={};period_ales={};parts={}
    for period,limits in PERIODS.items():
        part,tasks,assn,checks=get_period_data_and_tasks(df,manifest,period,limits)
        parts[period]=part
        path=run_dir/"02_period_nested_cv"/period
        write_csv(assn,path/"reused_outer_split_assignment.csv")
        write_csv(checks,path/"outer_group_leakage_check.csv")
        logger.info("[%s] N=%s，格网=%s，GI有效=%s，外层任务=%s。",
                    period,len(part),part[GROUP].nunique(),part[GI].count(),len(tasks))
        scores=run_period_nested_cv(part,tasks,period,selected,run_dir,logger)
        period_scores[period]=scores
        logger.info("[%s] 独立训练后，开始25个测试折的GI-ALE。",period)
        period_ales[period]=run_period_ale(part,tasks,scores,period,selected,run_dir,domain,logger)

    performance=[]
    for name,sc in period_scores.items():
        for metric in ["Test_R2","Test_RMSE","Test_MAE","Test_MSE"]:
            performance.append({"period":name,"model":selected,"metric":metric,
                                "mean":float(sc[metric].mean()),
                                "SD":float(sc[metric].std(ddof=1)),
                                "n_outer_tasks":len(sc),"n_observations":len(parts[name])})
    perf=pd.DataFrame(performance)
    write_csv(perf,summdir/"01_Period_nestedCV_performance_mean_SD.csv")
    write_csv(pd.concat([period_scores[k].assign(period=k) for k in PERIODS],ignore_index=True),
              summdir/"02_Period_all_50_outer_task_scores.csv")

    combined=pd.concat([period_ales[k][0] for k in PERIODS],ignore_index=True)
    all_raw=pd.concat([pd.read_csv(run_dir/"03_GI_ALE"/k/"GI_ALE_25_outer_curves_raw.csv")
                       for k in PERIODS],ignore_index=True)
    raw_peak=summarize_raw_period_peaks(all_raw,summdir)
    peak_detail,peak_sum=summarize_ale_period_features(combined,domain,summdir)
    key=list(PERIODS)
    diff,paired=compare_ale_periods(period_ales[key[0]][1],period_ales[key[1]][1],
                                     period_ales[key[0]][0],period_ales[key[1]][0],summdir)
    plot_ale_compare(period_ales[key[0]][1],period_ales[key[1]][1],domain,
                     figdir,main_run,logger)
    plot_ale_differences(paired,figdir)

    with pd.ExcelWriter(summdir/"06_GI_2001_Reviewer_Key_Tables.xlsx",engine="openpyxl") as writer:
        break_result.to_excel(writer,sheet_name="GI_jump_2001",index=False)
        jumps.to_excel(writer,sheet_name="Annual_jump_ranking",index=False)
        boot.to_excel(writer,sheet_name="Block100_bootstrap",index=False)
        perf.to_excel(writer,sheet_name="Period_model_performance",index=False)
        peak_sum.to_excel(writer,sheet_name="ALE_feature_summary",index=False)
        peak_detail.to_excel(writer,sheet_name="ALE_each_model",index=False)
        raw_peak.to_excel(writer,sheet_name="ALE_raw_support_peaks",index=False)
        diff.to_excel(writer,sheet_name="ALE_curve_difference",index=False)

    write_readme(run_dir,main_run,domain,selected)
    save_json({"run_start":start.isoformat(),"run_end":datetime.now().isoformat(),
               "selected_model":selected,"completed":True,
               "run_dir":str(run_dir),"n_periods":len(PERIODS),
               "n_outer_tasks_per_period":int(mc.N_REPEATS*mc.OUTER_N_SPLITS),
               "common_GI_domain":domain,
               "note":"Do not infer resolution-caused artifacts or causal grazing thresholds."},
              run_dir/"00_config"/"run_complete_summary.json")
    mc.save_output_manifest(run_dir)
    (run_dir/"RUN_COMPLETE.flag").write_text(datetime.now().isoformat(),encoding="utf-8")
    (OUTPUT_ROOT_FOLDER/LATEST_POINTER).write_text(str(run_dir),encoding="utf-8")
    active=OUTPUT_ROOT_FOLDER/ACTIVE_POINTER
    if active.exists() and active.read_text(encoding="utf-8").strip()==str(run_dir):
        active.unlink()
    logger.info("审稿人2第15条 GI 2001分析全部完成。保存到 %s",run_dir)
    logger.info("关键表：%s",summdir/"06_GI_2001_Reviewer_Key_Tables.xlsx")


def self_test():
    """快速函数测试：不访问用户本地 Windows 路径，也不生成正式实验结果。"""
    import tempfile
    from sklearn.ensemble import RandomForestRegressor
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder)
        rng=np.random.default_rng(2026)
        p=[]
        for gid in range(20):
            for year in range(1985,2021):
                gi=0.13 + .002*(year-1985)+.02*np.sin(gid)+rng.normal(0,.02)
                p.append({GROUP:f"p{gid:02d}",BLOCK:f"block{gid//4}",YEAR:year,GI:gi,
                          TARGET:.1+.4*gi-0.45*(gi-.19)**2+rng.normal(0,.01)})
        df=pd.DataFrame(p)
        logger=logging.getLogger("test06")
        domain=derive_common_gi_domain(df)
        assert domain["valid"] and domain["lower"]<domain["reference"]<domain["upper"]
        annual,rank,summ,boot=gi_break_diagnostics(df,out/"data",out/"fig",logger)
        assert len(rank)==35 and len(summ)==1 and len(boot)==4
        Xi=df[[GI,YEAR]].copy()
        reg=RandomForestRegressor(n_estimators=30,min_samples_leaf=3,random_state=42,n_jobs=1)
        reg.fit(Xi,df[TARGET])
        cv=mc.calculate_ale_1d(reg,Xi,GI,n_bins=ALE_N_BINS,q_range=ALE_Q_RANGE)
        assert len(cv)>5 and cv["ALE"].notna().all()
        if not len(list((out/"fig").glob("Fig_06A*.png"))):
            raise RuntimeError("绘图测试未生成PNG")
    print("SELF TEST PASSED: GI breakpoint, Block100 bootstrap, GI overlap, ALE and figures.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="06 GI 2001 breakpoint and period nested-CV sensitivity")
    parser.add_argument("--self-test",action="store_true",help="用合成数据自检，无需真实CSV")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        try:
            main()
        except Exception as exc:
            print("\n[06运行失败]",repr(exc))
            traceback.print_exc()
            print("若需要反馈故障，请保留对应06输出文件夹内06_logs_checkpoints日志和报错信息。")
            raise
