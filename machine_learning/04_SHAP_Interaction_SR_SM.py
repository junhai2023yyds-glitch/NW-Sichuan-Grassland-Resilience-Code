# -*- coding: utf-8 -*-
"""
04_SHAP_Interaction_SR_SM.py

川西北草地生态系统韧性驱动因素 —— SR × SM TreeSHAP interaction 补充分析

用途
----
针对导师提出的“SR × SM 的 SHAP interaction 建议补做”进行正式补充分析。
本脚本不重新运行五模型 Nested CV，不重新运行 Optuna，也不改变 01/02/03 的任何结果。
它读取已经完成的主分析目录，严格复用：
1) 主分析真实最优模型；
2) 5 次重复 × 5 折（25 套）patch_id1 外层划分；
3) 每个外层任务已经确定的最优超参数；
4) 主分析相同的 7 个生态驱动变量 + Year 时间协助变量；
5) 每个外层测试折仅用其对应外层训练集重建模型，再计算测试折 TreeSHAP interaction。

正式输出
--------
A. SR × SM 的 repeated OOF TreeSHAP interaction 样本结果；
B. 每个观测在 5 次重复中的 interaction 平均值（用于正式绘图，避免一个观测重复画 5 次）；
C. 7 个生态驱动变量 21 对两两 interaction 的 25 折强度排名（诊断/防止只挑 SR×SM）；
D. SR × SM mean(|interaction|) 的 patch_id1 cluster-bootstrap 95% CI；
E. 600 dpi SR × SM interaction 正式补充图。

重要说明
--------
- 当前正式最优模型为 RF 时，本脚本直接使用 RandomForestRegressor 的 TreeSHAP interaction。
- 同时兼容 LightGBM / XGBoost 成为主分析最优模型的情况。
- 若主分析最优模型为 OLS 或 SVR，TreeSHAP interaction 不适用，脚本会明确停止。
- Year 仍进入模型，但“生态驱动变量两两 interaction 排名”只统计 7 个正式生态驱动变量，排除 Year。
- 图中横轴为 SR，纵轴为真正的 SR×SM SHAP interaction value，颜色表示 SM。
- 图件排版与最新 03 最终版同步：横轴刻度值第一行；SR 单位位于最右侧并与刻度值平行；SR 名称在下一行中央；interaction 纵轴固定 3 位小数。

运行环境
--------
请将本脚本与最新版 ml_common.py 放在同一 machine_learning 文件夹中运行。
默认读取 <repo>/data/modeling_data.csv，并从 <repo>/output/machine_learning/latest_main_run.txt 自动定位主分析。
推荐运行顺序：01 -> 02 -> 03 -> 04（03 与 04 实际互不依赖，先后均可）。
"""

from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

import shap

from ml_common import (
    TARGET,
    GROUP_COLUMN,
    YEAR_COLUMN,
    SAMPLE_ID_COLUMN,
    DRIVER_FEATURES,
    MAIN_MODEL_FEATURES,
    FEATURE_UNITS,
    DATA_FILE_PATH,
    OUTPUT_ROOT_PATH,
    ensure_dir,
    load_model_data,
    load_outer_tasks_from_assignment,
    fit_outer_model,
)


# =============================================================================
# 1. GitHub 公共复现路径与固定配置
# =============================================================================
CSV_FILE_PATH = DATA_FILE_PATH
OUTPUT_ROOT_FOLDER = OUTPUT_ROOT_PATH

# 04 始终读取 01 完成后写出的 latest_main_run.txt，
# 不再绑定任何本机绝对路径或固定时间戳目录。
LATEST_MAIN_POINTER = OUTPUT_ROOT_FOLDER / "latest_main_run.txt"

INTERACTION_RUN_PREFIX = "Final_ML_Interaction_SR_SM"
ACTIVE_POINTER = OUTPUT_ROOT_FOLDER / "active_interaction_sr_sm_run.txt"
LATEST_POINTER = OUTPUT_ROOT_FOLDER / "latest_interaction_sr_sm_run.txt"

# 与当前 SHAP 外层并行设置保持一致：3 路并行。
INTERACTION_PARALLEL_JOBS = 3

# 只用于 SR×SM mean(|interaction|) 的格网级 bootstrap。
CLUSTER_BOOTSTRAP_B = 1000
CLUSTER_BOOTSTRAP_SEED = 20260815

# 正式图件：与 03_Redraw_Final_Figures_COMPLETE_FINAL.py 的排版规范同步
FIG_DPI = 600
SCATTER_ALPHA = 0.42
SCATTER_SIZE = 10
SMOOTH_BINS = 35

FONT_FAMILY = "Times New Roman"
FONT_SIZE = 10.5
AXIS_LABEL_SIZE = 11.0
TICK_SIZE = 9.5
LEGEND_SIZE = 7.5
AXIS_LINE_WIDTH = 0.8

# 与 03 最终版完全一致：变量名居中；单位位于最右侧 x 刻度值右边，
# 并与横轴刻度值处于同一视觉水平位置。
FEATURE_LABEL_Y = -0.145
UNIT_TICK_Y = -0.055
UNIT_TICK_X = 1.015

# interaction effect 与 ALE effect 同属模型输出效应量，通常处于 10^-3~10^-2 量级。
# 为避免两位小数造成 0.00 / -0.00 重复，纵轴固定显示 3 位小数。
INTERACTION_Y_DECIMALS = 3
INTERACTION_Y_NBINS = 5

# 只允许 TreeSHAP interaction 的树模型。
TREE_MODELS = {"RF", "LightGBM", "XGBoost"}


# =============================================================================
# 2. 绘图与日志工具
# =============================================================================
def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.serif"] = [FONT_FAMILY]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_SIZE
    plt.rcParams["axes.linewidth"] = AXIS_LINE_WIDTH
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


configure_matplotlib()


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("NW_Sichuan_SR_SM_Interaction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def safe_json_dump(obj: Any, file_path: Path) -> None:
    def default(o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def read_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figure_all_formats(fig: plt.Figure, output_base: Path) -> None:
    """与 03 最终版一致：PNG/TIF 600 dpi，同时输出 PDF/SVG 矢量格式。"""
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    try:
        fig.savefig(
            output_base.with_suffix(".tif"),
            dpi=FIG_DPI,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
    except Exception:
        fig.savefig(
            output_base.with_suffix(".tif"),
            dpi=FIG_DPI,
            format="tiff",
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    fig.savefig(
        output_base.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(fig)


def set_sr_xaxis_label(ax: plt.Axes) -> None:
    """
    SCI 最终格式，与最新 03 代码完全同步：
    1. 刻度值位于第一行；
    2. 单位位于横轴最右侧，与刻度值平行；
    3. 变量名称单独位于下一行中央。

    示例：
    1200    1400    1600    1800   (kWh/m²)
                      SR
    """

    # 不使用 matplotlib 默认 xlabel，避免名称与单位互相影响。
    ax.set_xlabel("")

    # 变量名称：横轴下方中央，单独一行。
    ax.text(
        0.5,
        FEATURE_LABEL_Y,
        "SR",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=AXIS_LABEL_SIZE,
        fontfamily=FONT_FAMILY,
        clip_on=False,
    )

    # 单位：位于最右端，并与横轴刻度值基本平行。
    unit = FEATURE_UNITS.get("SR", "kWh/m²")
    if unit:
        ax.text(
            UNIT_TICK_X,
            UNIT_TICK_Y,
            f"({unit})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TICK_SIZE,
            fontfamily=FONT_FAMILY,
            clip_on=False,
        )


def format_interaction_yaxis(ax: plt.Axes) -> None:
    """
    interaction effect 纵轴固定 3 位小数，与 03 中 ALE effect 的显示逻辑一致。
    仅统一显示精度，不强制统一数值范围；同时消除 -0.000。
    """
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=INTERACTION_Y_NBINS, steps=[1, 2, 5, 10])
    )

    def _formatter(value: float, pos: int) -> str:
        threshold = 0.5 * 10 ** (-INTERACTION_Y_DECIMALS)
        if abs(value) < threshold:
            value = 0.0
        return f"{value:.{INTERACTION_Y_DECIMALS}f}"

    ax.yaxis.set_major_formatter(FuncFormatter(_formatter))


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(axis="both", direction="out", length=3.5, width=AXIS_LINE_WIDTH)
    # 横轴与 03 一样不强制补固定小数位，仅让 Matplotlib 选择清晰的实际量纲刻度。
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)


# =============================================================================
# 3. 主分析目录与 04 输出目录
# =============================================================================
def resolve_main_run() -> Path:
    if not LATEST_MAIN_POINTER.exists():
        raise FileNotFoundError(
            f"未找到 {LATEST_MAIN_POINTER}。\n"
            "请先完整运行 01_Main_ML_NestedCV_OOFSHAP_ALE.py。"
        )

    main_run = Path(LATEST_MAIN_POINTER.read_text(encoding="utf-8").strip())

    if not main_run.exists():
        raise FileNotFoundError(f"latest_main_run 指向的主分析目录不存在：{main_run}")
    if not (main_run / "RUN_COMPLETE.flag").exists():
        raise RuntimeError(f"主分析目录尚未完成（缺少 RUN_COMPLETE.flag）：{main_run}")
    return main_run


def create_or_resume_interaction_run(main_run: Path) -> Tuple[Path, bool]:
    """支持断点续跑；完成后再新建下一次时间戳目录。"""
    if ACTIVE_POINTER.exists():
        candidate = Path(ACTIVE_POINTER.read_text(encoding="utf-8").strip())
        linked = candidate / "linked_main_run.txt"
        if (
            candidate.exists()
            and not (candidate / "RUN_COMPLETE.flag").exists()
            and linked.exists()
            and linked.read_text(encoding="utf-8").strip() == str(main_run)
        ):
            return candidate, True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT_FOLDER / f"{INTERACTION_RUN_PREFIX}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "linked_main_run.txt").write_text(str(main_run), encoding="utf-8")
    ACTIVE_POINTER.write_text(str(run_dir), encoding="utf-8")
    return run_dir, False


def mark_interaction_complete(run_dir: Path) -> None:
    (run_dir / "RUN_COMPLETE.flag").write_text(
        datetime.now().isoformat(), encoding="utf-8"
    )
    LATEST_POINTER.write_text(str(run_dir), encoding="utf-8")
    if ACTIVE_POINTER.exists():
        try:
            if ACTIVE_POINTER.read_text(encoding="utf-8").strip() == str(run_dir):
                ACTIVE_POINTER.unlink()
        except Exception:
            pass


# =============================================================================
# 4. 模型与 TreeSHAP interaction 工具
# =============================================================================
def unwrap_tree_model_and_transform(
    fitted_model: Any,
    X: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[Any, pd.DataFrame]:
    """
    将 ml_common.fit_outer_model 返回的模型转换成 TreeExplainer 可直接使用的树模型，
    并用对应折训练好的 imputer 对测试数据做完全相同的折内变换。
    """
    # RF / LightGBM：sklearn Pipeline(imputer -> model)
    if hasattr(fitted_model, "named_steps"):
        steps = fitted_model.named_steps
        if "imputer" not in steps or "model" not in steps:
            raise RuntimeError("Pipeline 中没有找到 imputer/model。")
        arr = steps["imputer"].transform(X[list(feature_names)])
        X_trans = pd.DataFrame(
            arr,
            columns=list(feature_names),
            index=X.index,
        )
        return steps["model"], X_trans

    # XGBoost：ml_common.XGBModelBundle
    if hasattr(fitted_model, "model") and hasattr(fitted_model, "transform"):
        X_trans = fitted_model.transform(X[list(feature_names)])
        X_trans = pd.DataFrame(
            X_trans,
            columns=list(feature_names),
            index=X.index,
        )
        return fitted_model.model, X_trans

    raise TypeError(
        "无法识别 fit_outer_model 返回的模型结构，不能提取 TreeSHAP interaction。"
    )


def normalize_interaction_array(raw: Any) -> np.ndarray:
    """兼容不同 SHAP 版本的回归输出形态，最终统一为 [n, p, p]。"""
    if isinstance(raw, list):
        if len(raw) != 1:
            raise RuntimeError(
                f"SHAP interaction 返回 {len(raw)} 个输出；本研究应为单输出回归。"
            )
        raw = raw[0]

    arr = np.asarray(raw)

    # 某些版本可能返回 [n, p, p, 1]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise RuntimeError(f"无法识别 SHAP interaction 数组形状：{arr.shape}")
    return arr.astype(float, copy=False)


def fold_paths(fold_dir: Path, repeat_id: int, fold_id: int) -> Tuple[Path, Path, Path]:
    stem = f"repeat_{repeat_id:02d}_fold_{fold_id:02d}"
    return (
        fold_dir / f"{stem}_SR_SM_interaction.csv",
        fold_dir / f"{stem}_pairwise_strength.csv",
        fold_dir / f"{stem}.done",
    )


def run_one_interaction_fold(
    df: pd.DataFrame,
    task: Dict[str, Any],
    selected_model: str,
    result_row: Dict[str, Any],
    fold_dir_str: str,
) -> Dict[str, Any]:
    """单个外层折：重建模型 -> 测试折 TreeSHAP interaction -> 落盘。"""
    fold_dir = Path(fold_dir_str)
    r = int(task["repeat"])
    f = int(task["fold"])
    seed = int(task["seed"])
    sample_file, pair_file, done_file = fold_paths(fold_dir, r, f)

    if done_file.exists() and sample_file.exists() and pair_file.exists():
        return {
            "repeat": r,
            "fold": f,
            "status": "skipped",
            "sample_file": str(sample_file),
            "pair_file": str(pair_file),
        }

    params_raw = result_row.get("Best_Params_JSON", "{}")
    params = json.loads(params_raw) if isinstance(params_raw, str) else dict(params_raw or {})

    n_est = result_row.get("XGB_Final_n_estimators", np.nan)
    final_n_estimators = None if pd.isna(n_est) else int(n_est)

    tr = np.asarray(task["train_idx"], dtype=int)
    te = np.asarray(task["test_idx"], dtype=int)

    X_all = df[list(MAIN_MODEL_FEATURES)].copy()
    y_all = df[TARGET].copy()
    X_train = X_all.iloc[tr].copy()
    y_train = y_all.iloc[tr].copy()
    X_test = X_all.iloc[te].copy()

    # 与 01 的 OOF-SHAP 重建逻辑保持一致；外层并行时每个模型内部仅用 1 核。
    random_state = seed + r * 100 + f
    fitted_model = fit_outer_model(
        selected_model,
        params,
        final_n_estimators,
        X_train,
        y_train,
        random_state,
        MAIN_MODEL_FEATURES,
        model_n_jobs=1,
    )

    tree_model, X_test_trans = unwrap_tree_model_and_transform(
        fitted_model,
        X_test,
        MAIN_MODEL_FEATURES,
    )

    # interaction 需要 tree_path_dependent；不提供 background data。
    explainer = shap.TreeExplainer(
        tree_model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    raw_inter = explainer.shap_interaction_values(X_test_trans)
    inter = normalize_interaction_array(raw_inter)

    if inter.shape[0] != len(X_test_trans) or inter.shape[1] != len(MAIN_MODEL_FEATURES):
        raise RuntimeError(
            f"R{r}F{f} interaction 维度与测试数据不一致："
            f"interaction={inter.shape}, X={X_test_trans.shape}"
        )

    sr_idx = list(MAIN_MODEL_FEATURES).index("SR")
    sm_idx = list(MAIN_MODEL_FEATURES).index("SM")
    sr_sm = inter[:, sr_idx, sm_idx]

    # -------------------------------------------------------------------------
    # 1) 每个测试样本的 SR×SM interaction
    # -------------------------------------------------------------------------
    ids = X_test_trans.index.to_numpy(dtype=int)
    sample_out = pd.DataFrame({
        SAMPLE_ID_COLUMN: df.loc[ids, SAMPLE_ID_COLUMN].astype(int).to_numpy(),
        GROUP_COLUMN: df.loc[ids, GROUP_COLUMN].astype(str).to_numpy(),
        YEAR_COLUMN: df.loc[ids, YEAR_COLUMN].to_numpy(),
        "repeat": r,
        "fold": f,
        # 模型真正看到的折内填补后数值
        "SR": X_test_trans["SR"].to_numpy(dtype=float),
        "SM": X_test_trans["SM"].to_numpy(dtype=float),
        "SHAP_interaction_SR_SM": sr_sm.astype(float),
        "abs_SHAP_interaction_SR_SM": np.abs(sr_sm).astype(float),
    })
    sample_out.to_csv(sample_file, index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # 2) 7 个生态驱动变量全部 21 对 pairwise interaction 强度
    #    Year 不进入这一排名。
    # -------------------------------------------------------------------------
    pair_rows: List[Dict[str, Any]] = []
    for a, b in combinations(DRIVER_FEATURES, 2):
        i = list(MAIN_MODEL_FEATURES).index(a)
        j = list(MAIN_MODEL_FEATURES).index(b)
        vals = inter[:, i, j]
        pair_rows.append({
            "repeat": r,
            "fold": f,
            "feature_1": a,
            "feature_2": b,
            "pair": f"{a} × {b}",
            "mean_abs_interaction": float(np.mean(np.abs(vals))),
            "mean_signed_interaction": float(np.mean(vals)),
            "median_abs_interaction": float(np.median(np.abs(vals))),
            "positive_fraction": float(np.mean(vals > 0)),
            "n_test_samples": int(len(vals)),
        })
    pd.DataFrame(pair_rows).to_csv(pair_file, index=False, encoding="utf-8-sig")

    done_file.write_text("done", encoding="utf-8")

    del raw_inter, inter, explainer, tree_model, fitted_model
    gc.collect()

    return {
        "repeat": r,
        "fold": f,
        "status": "completed",
        "sample_file": str(sample_file),
        "pair_file": str(pair_file),
    }


# =============================================================================
# 5. 汇总统计
# =============================================================================
def combine_fold_outputs(fold_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_files = sorted(fold_dir.glob("repeat_*_fold_*_SR_SM_interaction.csv"))
    pair_files = sorted(fold_dir.glob("repeat_*_fold_*_pairwise_strength.csv"))

    if len(sample_files) != 25 or len(pair_files) != 25:
        raise RuntimeError(
            f"04 应有 25 个外层折输出；当前 sample={len(sample_files)}, pair={len(pair_files)}。"
        )

    repeated = pd.concat([pd.read_csv(p) for p in sample_files], ignore_index=True)
    pairwise = pd.concat([pd.read_csv(p) for p in pair_files], ignore_index=True)
    return repeated, pairwise


def aggregate_repeated_oof_by_sample(repeated: pd.DataFrame) -> pd.DataFrame:
    """
    每个观测在 5 次重复中都恰好作为一次外层测试样本出现。
    正式图只画每个 sample_id 的 5 次 interaction 平均值，避免重复绘制同一观测。
    """
    counts = repeated.groupby(SAMPLE_ID_COLUMN)["repeat"].nunique()
    if not counts.eq(5).all():
        bad = counts[counts != 5]
        raise RuntimeError(
            f"部分 sample_id 没有完整 5 次重复 OOF interaction；异常数量={len(bad)}。"
        )

    out = (
        repeated.groupby(SAMPLE_ID_COLUMN, as_index=False)
        .agg({
            GROUP_COLUMN: "first",
            YEAR_COLUMN: "first",
            "SR": "mean",
            "SM": "mean",
            "SHAP_interaction_SR_SM": "mean",
            "abs_SHAP_interaction_SR_SM": "mean",
            "repeat": "nunique",
        })
        .rename(columns={"repeat": "n_repeats_averaged"})
    )
    return out


def summarize_pairwise_25fold(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, g in pairwise.groupby("pair", sort=False):
        rows.append({
            "feature_1": g["feature_1"].iloc[0],
            "feature_2": g["feature_2"].iloc[0],
            "pair": pair,
            "n_outer_models": int(len(g)),
            "mean_abs_interaction_25models_mean": float(g["mean_abs_interaction"].mean()),
            "mean_abs_interaction_25models_SD": float(g["mean_abs_interaction"].std(ddof=1)),
            "mean_abs_interaction_25models_q2.5": float(g["mean_abs_interaction"].quantile(0.025)),
            "mean_abs_interaction_25models_q97.5": float(g["mean_abs_interaction"].quantile(0.975)),
            "mean_signed_interaction_25models_mean": float(g["mean_signed_interaction"].mean()),
            "positive_fraction_25models_mean": float(g["positive_fraction"].mean()),
        })

    summary = pd.DataFrame(rows).sort_values(
        "mean_abs_interaction_25models_mean", ascending=False
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def cluster_bootstrap_sr_sm(
    sample_mean: pd.DataFrame,
    B: int = CLUSTER_BOOTSTRAP_B,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """以 patch_id1 为重采样单位，估计 SR×SM mean(|interaction|) 和 signed mean 的 CI。"""
    d = sample_mean[[GROUP_COLUMN, "SHAP_interaction_SR_SM"]].copy()
    d["abs_interaction"] = np.abs(d["SHAP_interaction_SR_SM"].to_numpy(dtype=float))

    grouped = d.groupby(GROUP_COLUMN).agg(
        abs_sum=("abs_interaction", "sum"),
        signed_sum=("SHAP_interaction_SR_SM", "sum"),
        n=("SHAP_interaction_SR_SM", "size"),
    )

    group_ids = grouped.index.to_numpy()
    abs_sum = grouped["abs_sum"]
    signed_sum = grouped["signed_sum"]
    n = grouped["n"]

    rng = np.random.default_rng(seed)
    boot_abs = np.empty(B, dtype=float)
    boot_signed = np.empty(B, dtype=float)

    for b in range(B):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        total_n = float(n.loc[sampled].sum())
        boot_abs[b] = float(abs_sum.loc[sampled].sum()) / total_n
        boot_signed[b] = float(signed_sum.loc[sampled].sum()) / total_n

    return pd.DataFrame([
        {
            "metric": "mean_abs_SHAP_interaction_SR_SM",
            "estimate": float(d["abs_interaction"].mean()),
            "bootstrap_B": B,
            "cluster": GROUP_COLUMN,
            "CI_lower_2.5": float(np.quantile(boot_abs, 0.025)),
            "CI_upper_97.5": float(np.quantile(boot_abs, 0.975)),
            "bootstrap_mean": float(np.mean(boot_abs)),
            "bootstrap_SD": float(np.std(boot_abs, ddof=1)),
        },
        {
            "metric": "mean_signed_SHAP_interaction_SR_SM",
            "estimate": float(d["SHAP_interaction_SR_SM"].mean()),
            "bootstrap_B": B,
            "cluster": GROUP_COLUMN,
            "CI_lower_2.5": float(np.quantile(boot_signed, 0.025)),
            "CI_upper_97.5": float(np.quantile(boot_signed, 0.975)),
            "bootstrap_mean": float(np.mean(boot_signed)),
            "bootstrap_SD": float(np.std(boot_signed, ddof=1)),
        },
    ])


def sm_tertile_summary(sample_mean: pd.DataFrame) -> pd.DataFrame:
    """仅作为讨论辅助诊断，不要求一定进入论文。"""
    d = sample_mean.copy()
    try:
        d["SM_group"] = pd.qcut(
            d["SM"],
            q=[0, 1/3, 2/3, 1],
            labels=["Low SM", "Middle SM", "High SM"],
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()

    rows = []
    for name, g in d.groupby("SM_group", observed=True):
        rows.append({
            "SM_group": str(name),
            "n": int(len(g)),
            "SM_mean": float(g["SM"].mean()),
            "SR_mean": float(g["SR"].mean()),
            "interaction_mean": float(g["SHAP_interaction_SR_SM"].mean()),
            "interaction_mean_abs": float(np.abs(g["SHAP_interaction_SR_SM"]).mean()),
            "interaction_positive_fraction": float((g["SHAP_interaction_SR_SM"] > 0).mean()),
        })
    return pd.DataFrame(rows)


def build_binned_curve(sample_mean: pd.DataFrame, bins: int = SMOOTH_BINS) -> pd.DataFrame:
    """SR 分位数分箱后计算平均 SR×SM interaction，仅用于给散点图叠加趋势线。"""
    d = sample_mean[["SR", "SHAP_interaction_SR_SM"]].dropna().copy()
    if len(d) < bins * 5:
        return pd.DataFrame()

    try:
        d["bin"] = pd.qcut(d["SR"], q=bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    curve = (
        d.groupby("bin", observed=True)
        .agg(
            SR=("SR", "mean"),
            interaction=("SHAP_interaction_SR_SM", "mean"),
            interaction_median=("SHAP_interaction_SR_SM", "median"),
            n=("SHAP_interaction_SR_SM", "size"),
        )
        .reset_index(drop=True)
        .sort_values("SR")
    )
    return curve


# =============================================================================
# 6. 正式 SR × SM interaction 图
# =============================================================================
def plot_sr_sm_interaction(
    sample_mean: pd.DataFrame,
    curve: pd.DataFrame,
    output_dir: Path,
) -> None:
    d = sample_mean.dropna(subset=["SR", "SM", "SHAP_interaction_SR_SM"]).copy()

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    sc = ax.scatter(
        d["SR"].to_numpy(),
        d["SHAP_interaction_SR_SM"].to_numpy(),
        c=d["SM"].to_numpy(),
        cmap="viridis",
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        rasterized=True,
    )

    ax.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0, zorder=1)

    if curve is not None and not curve.empty:
        ax.plot(
            curve["SR"].to_numpy(),
            curve["interaction"].to_numpy(),
            color="black",
            linewidth=2.0,
            label="Binned mean",
            zorder=5,
        )
        ax.legend(frameon=False, loc="best", fontsize=10)

    set_sr_xaxis_label(ax)
    ax.set_ylabel(
        "SHAP interaction value (SR × SM)",
        fontsize=AXIS_LABEL_SIZE,
        fontfamily=FONT_FAMILY,
    )
    style_axis(ax)
    format_interaction_yaxis(ax)

    cbar = fig.colorbar(sc, ax=ax, pad=0.025, fraction=0.050)
    cbar.set_label(
        f"SM ({FEATURE_UNITS.get('SM', 'mm')})",
        fontsize=AXIS_LABEL_SIZE,
        fontfamily=FONT_FAMILY,
    )
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)

    fig.tight_layout()
    save_figure_all_formats(
        fig,
        output_dir / "Fig_S_SR_SM_TreeSHAP_interaction",
    )


# =============================================================================
# 7. 主程序
# =============================================================================
def main() -> None:
    start_time = datetime.now()
    main_run = resolve_main_run()
    run_dir, resumed = create_or_resume_interaction_run(main_run)

    config_dir = ensure_dir(run_dir / "00_config")
    fold_dir = ensure_dir(run_dir / "01_fold_interactions")
    ranking_dir = ensure_dir(run_dir / "02_pairwise_ranking")
    summary_dir = ensure_dir(run_dir / "03_SR_SM_summary")
    figure_dir = ensure_dir(run_dir / "04_figures")
    log_dir = ensure_dir(run_dir / "05_logs")

    logger = setup_logger(log_dir / "interaction_analysis.log")
    logger.info("=" * 90)
    logger.info("SR × SM TreeSHAP interaction 补充分析开始")
    logger.info(f"主分析目录：{main_run}")
    logger.info(f"04 输出目录：{run_dir}")
    logger.info(f"本次是否断点续跑：{resumed}")
    logger.info(f"外层 interaction 并行：{INTERACTION_PARALLEL_JOBS}")
    logger.info("=" * 90)

    # -------------------------------------------------------------------------
    # A. 读取主分析最优模型与主结果
    # -------------------------------------------------------------------------
    selection_file = main_run / "selected_model_for_sensitivity.json"
    if not selection_file.exists():
        raise FileNotFoundError(f"缺少主分析模型选择文件：{selection_file}")

    selection = read_json(selection_file)
    selected_model = str(selection["selected_model"])
    logger.info(f"主分析真实最优模型：{selected_model}")

    if selected_model not in TREE_MODELS:
        raise RuntimeError(
            f"当前主分析最优模型为 {selected_model}，不是 TreeSHAP interaction 支持的树模型。"
            "本 04 不能用普通 SHAP dependence 冒充真正 interaction。"
        )

    all_scores_file = main_run / "02_main_nested_cv" / "all_outer_scores.csv"
    split_file = main_run / "01_split_manifest" / "outer_split_assignment_by_observation.csv"
    if not all_scores_file.exists():
        raise FileNotFoundError(all_scores_file)
    if not split_file.exists():
        raise FileNotFoundError(split_file)

    model_results = pd.read_csv(all_scores_file)
    selected_results = model_results[model_results["Model"] == selected_model].copy()
    if len(selected_results) != 25:
        raise RuntimeError(
            f"主分析最优模型 {selected_model} 应有 25 个外层结果，实际为 {len(selected_results)}。"
        )

    result_lookup = selected_results.set_index(["Repeat", "Outer_Fold"])

    # -------------------------------------------------------------------------
    # B. 读取建模数据与完全相同的 25 套外层划分
    # -------------------------------------------------------------------------
    df = load_model_data(
        CSV_FILE_PATH,
        MAIN_MODEL_FEATURES,
        require_block100=True,
    )
    assignment = pd.read_csv(split_file)
    outer_tasks = load_outer_tasks_from_assignment(df, assignment, GROUP_COLUMN)

    if len(outer_tasks) != 25:
        raise RuntimeError(f"载入主分析外层任务后不是 25 套，而是 {len(outer_tasks)} 套。")

    # 配置记录
    config = {
        "analysis": "SR × SM TreeSHAP interaction",
        "linked_main_run": str(main_run),
        "input_csv": str(CSV_FILE_PATH),
        "selected_model": selected_model,
        "model_features": MAIN_MODEL_FEATURES,
        "ecological_driver_features": DRIVER_FEATURES,
        "temporal_control": YEAR_COLUMN,
        "outer_design": "same repeated 5 × 5 patch_id1 outer splits as main analysis",
        "n_outer_models": len(outer_tasks),
        "interaction_parallel_jobs": INTERACTION_PARALLEL_JOBS,
        "model_internal_jobs_when_rebuilding": 1,
        "formal_interaction": "TreeSHAP shap_interaction_values; SR × SM",
        "pairwise_ranking_scope": "all 21 pairs among 7 ecological drivers; Year excluded",
        "cluster_bootstrap_B": CLUSTER_BOOTSTRAP_B,
        "cluster_bootstrap_unit": GROUP_COLUMN,
        "figure_dpi": FIG_DPI,
        "figure_formats": ["PNG", "TIF", "PDF", "SVG"],
        "xaxis_unit_layout": "same as 03 final: unit right of rightmost x tick and aligned with tick-label height",
        "interaction_y_decimals": INTERACTION_Y_DECIMALS,
        "xaxis_tick_decimals": "automatic, same rule as 03 final (no forced trailing zeros)",
        "run_start": start_time.isoformat(),
    }
    safe_json_dump(config, config_dir / "interaction_config.json")

    # -------------------------------------------------------------------------
    # C. 25 个外层折计算 interaction，3 路并行 + 每折 checkpoint
    # -------------------------------------------------------------------------
    pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    completed_at_start = 0

    for task in outer_tasks:
        r = int(task["repeat"])
        f = int(task["fold"])
        sample_file, pair_file, done_file = fold_paths(fold_dir, r, f)
        if done_file.exists() and sample_file.exists() and pair_file.exists():
            completed_at_start += 1
        else:
            row = result_lookup.loc[(r, f)].to_dict()
            pending.append((task, row))

    logger.info(
        f"checkpoint 扫描：已完成 {completed_at_start}/25；待计算 {len(pending)}/25。"
    )

    completed = completed_at_start
    if pending:
        workers = min(max(1, INTERACTION_PARALLEL_JOBS), len(pending))
        logger.info(f"开始 {workers} 路并行计算 TreeSHAP interaction。")

        if workers == 1:
            for task, row in pending:
                result = run_one_interaction_fold(
                    df, task, selected_model, row, str(fold_dir)
                )
                completed += 1
                logger.info(
                    f"[{completed}/25] R{result['repeat']}F{result['fold']} {result['status']}"
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        run_one_interaction_fold,
                        df,
                        task,
                        selected_model,
                        row,
                        str(fold_dir),
                    )
                    for task, row in pending
                ]
                for future in as_completed(futures):
                    result = future.result()
                    completed += 1
                    logger.info(
                        f"[{completed}/25] R{result['repeat']}F{result['fold']} {result['status']}"
                    )

    # -------------------------------------------------------------------------
    # D. 汇总 repeated OOF interaction 与 21 对 interaction 排名
    # -------------------------------------------------------------------------
    repeated, pairwise = combine_fold_outputs(fold_dir)
    repeated.to_csv(
        summary_dir / "SR_SM_OOF_interaction_all_5repeats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample_mean = aggregate_repeated_oof_by_sample(repeated)
    sample_mean.to_csv(
        summary_dir / "SR_SM_OOF_interaction_sample_mean_across_5repeats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pairwise.to_csv(
        ranking_dir / "SHAP_interaction_pairwise_21pairs_25models_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pair_rank = summarize_pairwise_25fold(pairwise)
    pair_rank.to_csv(
        ranking_dir / "SHAP_interaction_pairwise_21pairs_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # E. SR×SM 统计与 bootstrap
    # -------------------------------------------------------------------------
    boot = cluster_bootstrap_sr_sm(sample_mean)
    boot.to_csv(
        summary_dir / "SR_SM_interaction_cluster_bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sr_sm_rank_row = pair_rank[
        ((pair_rank["feature_1"] == "SR") & (pair_rank["feature_2"] == "SM"))
        | ((pair_rank["feature_1"] == "SM") & (pair_rank["feature_2"] == "SR"))
    ]
    if sr_sm_rank_row.empty:
        raise RuntimeError("21 对 interaction 排名中没有找到 SR × SM。")
    sr_sm_rank_row = sr_sm_rank_row.iloc[0]

    summary = pd.DataFrame([{
        "selected_model": selected_model,
        "n_outer_models": 25,
        "n_unique_samples": int(sample_mean[SAMPLE_ID_COLUMN].nunique()),
        "n_patch_id1": int(sample_mean[GROUP_COLUMN].nunique()),
        "SR_SM_pair_rank_among_21_driver_pairs": int(sr_sm_rank_row["rank"]),
        "SR_SM_mean_abs_interaction_25models_mean": float(
            sr_sm_rank_row["mean_abs_interaction_25models_mean"]
        ),
        "SR_SM_mean_abs_interaction_25models_SD": float(
            sr_sm_rank_row["mean_abs_interaction_25models_SD"]
        ),
        "SR_SM_sample_averaged_mean_abs_interaction": float(
            np.abs(sample_mean["SHAP_interaction_SR_SM"]).mean()
        ),
        "SR_SM_sample_averaged_mean_signed_interaction": float(
            sample_mean["SHAP_interaction_SR_SM"].mean()
        ),
        "SR_SM_sample_averaged_positive_fraction": float(
            (sample_mean["SHAP_interaction_SR_SM"] > 0).mean()
        ),
        "SR_SM_cluster_bootstrap_mean_abs_CI_lower_2.5": float(
            boot.loc[
                boot["metric"] == "mean_abs_SHAP_interaction_SR_SM",
                "CI_lower_2.5",
            ].iloc[0]
        ),
        "SR_SM_cluster_bootstrap_mean_abs_CI_upper_97.5": float(
            boot.loc[
                boot["metric"] == "mean_abs_SHAP_interaction_SR_SM",
                "CI_upper_97.5",
            ].iloc[0]
        ),
    }])
    summary.to_csv(
        summary_dir / "SR_SM_interaction_key_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tertile = sm_tertile_summary(sample_mean)
    if not tertile.empty:
        tertile.to_csv(
            summary_dir / "SR_SM_interaction_by_SM_tertile_diagnostic.csv",
            index=False,
            encoding="utf-8-sig",
        )

    curve = build_binned_curve(sample_mean, SMOOTH_BINS)
    curve.to_csv(
        summary_dir / "SR_SM_interaction_binned_curve_for_plot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # F. 正式补充图
    # -------------------------------------------------------------------------
    plot_sr_sm_interaction(sample_mean, curve, figure_dir)

    # Excel 汇总，便于整理补充材料
    xlsx_file = run_dir / "SR_SM_SHAP_interaction_summary.xlsx"
    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Key_summary", index=False)
        boot.to_excel(writer, sheet_name="Cluster_bootstrap", index=False)
        pair_rank.to_excel(writer, sheet_name="21pair_ranking", index=False)
        pairwise.to_excel(writer, sheet_name="25model_pair_long", index=False)
        if not tertile.empty:
            tertile.to_excel(writer, sheet_name="SM_tertile_diag", index=False)
        curve.to_excel(writer, sheet_name="Plot_binned_curve", index=False)

    # -------------------------------------------------------------------------
    # G. 完成标记
    # -------------------------------------------------------------------------
    elapsed_min = (datetime.now() - start_time).total_seconds() / 60.0
    logger.info("=" * 90)
    logger.info("SR × SM TreeSHAP interaction 补充分析完成")
    logger.info(f"最优模型：{selected_model}")
    logger.info(
        f"SR×SM 在 7 个生态驱动的 21 对 interaction 中排名："
        f"{int(sr_sm_rank_row['rank'])}/21"
    )
    logger.info(
        "SR×SM mean|interaction|（25模型 mean±SD）："
        f"{sr_sm_rank_row['mean_abs_interaction_25models_mean']:.6g} ± "
        f"{sr_sm_rank_row['mean_abs_interaction_25models_SD']:.6g}"
    )
    logger.info(f"总耗时：{elapsed_min:.1f} min")
    logger.info(f"输出目录：{run_dir}")
    logger.info("=" * 90)

    mark_interaction_complete(run_dir)


if __name__ == "__main__":
    mp.freeze_support()
    main()
