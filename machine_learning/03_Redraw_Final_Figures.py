# -*- coding: utf-8 -*-
r"""
03_Redraw_Final_Figures.py

川西北草地生态系统韧性 —— 01 + 02 全图件独立重绘版

原则
----
1. 只读取已经完成的 01 主分析与 02 敏感性分析结果；
2. 不重新训练模型；
3. 不运行 Optuna；
4. 不重新计算 OOF-SHAP；
5. 不重新计算 ALE；
6. 重新绘制 01 和 02 中涉及的论文图件；
7. 输出文件夹严格按 01 / 02 分开；
8. 不覆盖 01 / 02 原始结果。
9. 横轴单位统一置于最右端并与刻度值平行；变量名单独位于下一行中央。

当前结果目录
------------
01 主分析：
C:\Users\yihai\Desktop\川西北\机器学习建模数据\XGBOOST+SHAP最新20km\Final_ML_Main_20260812_230042

02 敏感性：
C:\Users\yihai\Desktop\川西北\机器学习建模数据\XGBOOST+SHAP最新20km\Final_ML_Sensitivity_20260814_140146
"""

from __future__ import annotations

import gc
import json
import math
import string
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.textpath import TextPath
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import FuncFormatter
from scipy import stats

try:
    import shap
except Exception:
    shap = None


# =============================================================================
# 1. 固定路径
# =============================================================================
MAIN_RUN = Path(
    r"C:\Users\yihai\Desktop\川西北\机器学习建模数据\XGBOOST+SHAP最新20km\Final_ML_Main_20260812_230042"
)

SENS_RUN = Path(
    r"C:\Users\yihai\Desktop\川西北\机器学习建模数据\XGBOOST+SHAP最新20km\Final_ML_Sensitivity_20260814_140146"
)

# 所有重绘结果输出到这里；内部再严格分 01 / 02。
REDRAW_ROOT = MAIN_RUN.parent / "Final_Publication_Figures"

MAIN_OUT = REDRAW_ROOT / "01_Main_Analysis"
SENS_OUT = REDRAW_ROOT / "02_Sensitivity_Analyses"

# 01 主分析下继续分
MAIN_MODEL_OUT = MAIN_OUT / "01_Model_Performance"
MAIN_OOF_OUT = MAIN_OUT / "02_OOF_Prediction"
MAIN_SHAP_OUT = MAIN_OUT / "03_SHAP"
MAIN_ALE_OUT = MAIN_OUT / "04_ALE"

# 02 敏感性下继续按原 02 逻辑分
NOYEAR_OUT = SENS_OUT / "01_NoYear"
BLOCK_OUT = SENS_OUT / "02_100km_BlockCV"
TEMP_OUT = SENS_OUT / "03_Temperature_Tmean_DTR"

NOYEAR_PERF_OUT = NOYEAR_OUT / "01_Performance"
NOYEAR_SHAP_OUT = NOYEAR_OUT / "02_SHAP"
NOYEAR_ALE_OUT = NOYEAR_OUT / "03_ALE"

BLOCK_PERF_OUT = BLOCK_OUT / "01_Performance"

TEMP_PERF_OUT = TEMP_OUT / "01_Performance"
TEMP_SHAP_OUT = TEMP_OUT / "02_SHAP"
TEMP_ALE_OUT = TEMP_OUT / "03_ALE"

PLOT_DATA_OUT = REDRAW_ROOT / "00_Plot_Data_Snapshot"


# =============================================================================
# 2. 输出与字体
# =============================================================================
FIG_DPI = 600
# ALE 组合图会被直接插入 Word 并在论文中放大；单独提供更高分辨率的
# 位图输出。SVG 仍同步保存，可在支持 SVG 的 Word 版本中作为首选矢量图。
WORD_ALE_DPI = 720

SAVE_PNG = True
SAVE_TIF = True
SAVE_PDF = True
SAVE_SVG = True

FONT_FAMILY = "Times New Roman"

FONT_SIZE = 10.5
AXIS_LABEL_SIZE = 11.0
TICK_SIZE = 9.5
PANEL_LABEL_SIZE = 12.5
# ALE Word 组合图画布比 SHAP 组合图窄；用 10.5 pt 可使二者在
# Word 中等宽插入时呈现相同的最终编号视觉大小。
ALE_WORD_PANEL_LABEL_SIZE = 10.5
LEGEND_SIZE = 7.5
TITLE_SIZE = 11.0

# SHAP 条形图百分比字号
BAR_VALUE_SIZE = 11.5

AXIS_LINE_WIDTH = 0.8
ERRORBAR_LINE_WIDTH = 0.9

ALE_LINE_WIDTH = 1.9
COMPARE_LINE_WIDTH = 1.65
ZERO_LINE_WIDTH = 0.9
CI_ALPHA = 0.26

# 正文 ALE 使用 Top-6
ALE_TOP_N = 6

# 图尺寸
MODEL_SINGLE_SIZE = (7.2, 4.8)
MODEL_COMBO_SIZE = (8.8, 5.0)
MODEL_3PANEL_SIZE = (11.5, 3.9)
OOF_SIZE = (6.2, 5.6)

SHAP_BAR_SIZE = (7.0, 5.0)
SHAP_BEE_SIZE = (7.4, 5.5)
SHAP_COMBO_SIZE = (11.8, 5.3)
SHAP_CI_SIZE = (7.2, 5.0)
SHAP_ROSE_SIZE = (8.0, 8.0)
SHAP_ROSE_BEE_SIZE = (13.2, 6.6)
SHAP_DEP_COMBO_SIZE = (13.8, 7.8)
SHAP_DEP_SINGLE_SIZE = (6.3, 4.8)

ALE_COMBO_SIZE = (13.8, 7.8)
ALE_SINGLE_SIZE = (6.3, 4.8)
ALE_COMPARE_SIZE = (13.8, 7.8)

# Word 正文通栏版：10 in 源画布插入 A4 正文后会自动缩至约 6.7 in，
# 配合下方的字号放大，最终文字保持在约 9–10 pt，而不会因 13.8 in
# 的超宽画布被强制缩小而难以辨认。
# Word 正文通栏版保持紧凑横向比例，仅略增加高度，给纵轴顶部留出图例空间。
ALE_WORD_COMBO_SIZE = (10.0, 6.4)
ALE_WORD_TEXT_SCALE = 1.35
ALE_INTERVAL_LABEL = "2.5th–97.5th percentile intervals"

# Word 版画布较窄时，Matplotlib 会自动改变刻度间隔。这里固定为原 13.8 in
# 组合图的刻度，确保仅缩放视觉尺寸，绝不改变读图数值。
ALE_WORD_REFERENCE_TICKS = {
    "TMX": {
        "x": [12, 14, 16, 18, 20, 22, 24],
        "y": [-0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02],
    },
    "SR": {
        "x": [3000, 3200, 3400, 3600, 3800, 4000],
        "y": [-0.01, -0.0075, -0.005, -0.0025, 0.00, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015],
    },
    "TMN": {
        "x": [-30, -25, -20, -15, -10, -5],
        "y": [-0.025, -0.02, -0.015, -0.01, -0.005, 0.00, 0.005, 0.01],
    },
    "SM": {
        "x": [0, 20, 40, 60, 80, 100, 120, 140],
        "y": [-0.015, -0.01, -0.005, 0.00, 0.005, 0.01, 0.015],
    },
    "PRE": {
        "x": [500, 600, 700, 800, 900],
        "y": [-0.015, -0.01, -0.005, 0.00, 0.005, 0.01, 0.015],
    },
    "GI": {
        "x": [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "y": [-0.02, -0.015, -0.01, -0.005, 0.00, 0.005, 0.01],
    },
}

SENS_PERF_SIZE = (8.2, 4.6)
SENS_ALL_PERF_SIZE = (11.5, 3.9)


# =============================================================================
# 3. 单位、显示名
# =============================================================================
FEATURE_UNITS = {
    "Year": "",
    "SM": "mm",
    "PRE": "mm",
    "TMN": "°C",
    "TMX": "°C",
    "SR": "kWh/m²",
    "VPD": "kPa",
    "GI": "SU/ha",
    "Tmean": "°C",
    "DTR": "°C",
}

DISPLAY_NAME = {
    "Year": "Year",
    "SM": "SM",
    "PRE": "PRE",
    "TMN": "TMN",
    "TMX": "TMX",
    "SR": "SR",
    "VPD": "VPD",
    "GI": "GI",
    "Tmean": "Tmean",
    "DTR": "DTR",
}


# =============================================================================
# 4. ALE 配色
#
# 采用“生态含义 + SCI 低饱和”方案：
# TMX 高温 → 暖红
# SR 辐射 → 金黄
# TMN 低温 → 冷蓝
# SM 土壤水分 → 青绿
# PRE 降水 → 绿色
# GI 放牧 → 紫色
# VPD 大气干旱 → 棕色
# =============================================================================
FEATURE_STYLE = {
    "TMX": {"line": "#C65D4B", "fill": "#E8B9AE", "compare": "#8D3C31"},
    "SR":  {"line": "#B78928", "fill": "#E5D29B", "compare": "#80611C"},
    "TMN": {"line": "#4D78A8", "fill": "#BFD3E7", "compare": "#35577E"},
    "SM":  {"line": "#278878", "fill": "#B7DDD5", "compare": "#1D6257"},
    "PRE": {"line": "#559762", "fill": "#C8E1CD", "compare": "#3C6B46"},
    "GI":  {"line": "#8563A8", "fill": "#D8C9E7", "compare": "#60457E"},
    "VPD": {"line": "#987552", "fill": "#DFD0C1", "compare": "#6B513A"},
    "Tmean": {"line": "#C57647", "fill": "#EBC7B1", "compare": "#8A5330"},
    "DTR":   {"line": "#6785A5", "fill": "#CBD8E5", "compare": "#48617A"},
}

COLOR_ZERO = "#404040"
COLOR_GRID = "#B7B7B7"

COLOR_MAIN = "#547B97"
COLOR_SENS = "#B87963"

SHAP_BAR_CMAP = "viridis"
SHAP_BEE_CMAP = "Spectral_r"


# =============================================================================
# 5. Matplotlib
# =============================================================================
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.serif": [FONT_FAMILY],
    "font.size": FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.linewidth": AXIS_LINE_WIDTH,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =============================================================================
# 6. 通用函数
# =============================================================================
def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_file(path: Path | str, label: str = "") -> Path:
    p = Path(path)
    if not p.exists():
        msg = f"未找到文件：{p}"
        if label:
            msg += f"\n用途：{label}"
        raise FileNotFoundError(msg)
    return p


def optional_csv(path: Path | str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        print(f"[跳过] 未找到：{p}")
        return None
    return pd.read_csv(p)


def feature_name(feature: str) -> str:
    return DISPLAY_NAME.get(feature, str(feature))


def feature_unit(feature: str) -> str:
    return FEATURE_UNITS.get(feature, "")


def get_feature_style(feature: str) -> Dict[str, str]:
    return FEATURE_STYLE.get(
        feature,
        {"line": "#477DA8", "fill": "#C6D8E6", "compare": "#555555"},
    )


def panel_label(i: int) -> str:
    return f"({string.ascii_lowercase[i]})"


def format_three_decimal(value: float, _position: int) -> str:
    """标准四舍五入到三位小数，避免 0.0075 被显示为 0.007。"""
    rounded = Decimal(str(float(value))).quantize(
        Decimal("0.001"),
        rounding=ROUND_HALF_UP,
    )
    return f"{rounded:.3f}"


def add_panel_label(
    ax,
    i: int,
    x: float = 0.015,
    y: float = 0.985,
    fontsize: float = PANEL_LABEL_SIZE,
) -> None:
    ax.text(
        x,
        y,
        panel_label(i),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
        zorder=200,
        clip_on=False,
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)

    ax.tick_params(
        axis="both",
        width=AXIS_LINE_WIDTH,
        length=3.5,
    )

    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily(FONT_FAMILY)


def style_beeswarm_axis(ax) -> None:
    """Apply the common axis style and retain the beeswarm left spine."""
    style_axis(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(AXIS_LINE_WIDTH)


def set_feature_xaxis(
    ax,
    feature: str,
    y: float = -0.115,
    font_size: Optional[float] = None,
) -> None:
    """
    SCI 最终横轴格式：

    1. 横轴刻度值正常显示；
    2. 单位与刻度值处于同一水平带；
    3. 单位紧跟横轴最右侧刻度值；
    4. 变量名单独位于下一行中央。

    说明
    ----
    TMN 和 PRE 在当前自动刻度范围下，最右侧刻度可能略超出 xlim，
    之前用“xlim 内最后刻度”定位会误把单位放到前一个刻度之后，
    从而出现：
        (°C) -7.5
        (mm) 850

    因此本版对 TMN / PRE 采用最稳定的方法：
    直接把单位写进“距离右侧 xlim 最近的那个刻度标签”本身，
    形成一个完整 Text 对象：
        -7.5 (°C)
        850 (mm)

    这样数字和单位绝不可能互相重叠或错位。

    其他已经正常的 TMX / SR / SM / GI 等保持上一版方式不变。
    """

    ax.set_xlabel("")

    # ------------------------------------------------------------------
    # A. 变量名：下一行中央
    # ------------------------------------------------------------------
    ax.text(
        0.5,
        -0.145,
        feature_name(feature),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=AXIS_LABEL_SIZE if font_size is None else font_size,
        fontfamily=FONT_FAMILY,
        clip_on=False,
        zorder=300,
    )

    unit = feature_unit(feature)
    if not unit:
        return

    # ==================================================================
    # B. TMN / PRE：
    #    单位直接并入最右侧视觉刻度标签，彻底消除重叠。
    # ==================================================================
    if feature in {"TMN", "PRE"}:

        # Agg 后端下先绘制一次，使 ScalarFormatter 的刻度文字最终确定。
        # 这里只在图构建阶段调用，不绑定 draw_event，
        # 因此不会造成 PDF/SVG 的 FigureCanvasPdf 报错。
        ax.figure.canvas.draw()

        ticks = np.asarray(
            ax.get_xticks(),
            dtype=float,
        )

        ticklabels = ax.get_xticklabels()

        label_texts = [
            lab.get_text()
            for lab in ticklabels
        ]

        if len(ticks) == 0 or len(label_texts) == 0:
            return

        x0, x1 = ax.get_xlim()

        # 正常横轴：
        # 选择“距离右侧 xlim 最近”的主刻度。
        #
        # 这样即使 850 或 -7.5 略微位于自动 xlim 之外，
        # 仍然会被正确识别为视觉上的最右侧刻度，
        # 而不会误选 800 或 -10.0。
        if x1 >= x0:
            idx = int(
                np.nanargmin(
                    np.abs(ticks - x1)
                )
            )
        else:
            idx = int(
                np.nanargmin(
                    np.abs(ticks - x0)
                )
            )

        if idx >= len(label_texts):
            return

        base_label = str(
            label_texts[idx]
        ).strip()

        if not base_label:
            # 极少数情况下 formatter 尚未产生字符串，
            # 用通用格式兜底。
            value = ticks[idx]

            if abs(value - round(value)) < 1e-10:
                base_label = f"{value:.0f}"
            else:
                base_label = f"{value:g}"

        # 数字 + 空格 + 单位，成为同一个 Text 对象。
        # 因此不再存在“单位压住数字”或“单位跑到数字左边”的可能。
        label_texts[idx] = (
            f"{base_label} ({unit})"
        )

        # 固定当前主刻度及其文字。
        # 当前函数调用时数据、xlim 已经确定，
        # 后续 style_axis / legend / subplots_adjust 不会改变数据刻度。
        ax.set_xticks(
            ticks,
            labels=label_texts,
        )

        return

    # ==================================================================
    # C. 其他变量：
    #    保持上一版已经正常的定位方式，不改。
    # ==================================================================
    ticks = np.asarray(
        ax.get_xticks(),
        dtype=float,
    )

    labels = ax.get_xticklabels()

    x0, x1 = ax.get_xlim()
    xmin = min(x0, x1)
    xmax = max(x0, x1)
    tol = max(
        abs(xmax - xmin),
        1.0,
    ) * 1e-10

    pairs = []

    for tick, label in zip(
        ticks,
        labels,
    ):

        if not label.get_visible():
            continue

        if not (
            xmin - tol
            <= tick
            <= xmax + tol
        ):
            continue

        label_text = (
            label.get_text().strip()
        )

        if not label_text:
            continue

        pairs.append(
            (float(tick), label)
        )

    if not pairs:
        return

    if x1 >= x0:
        right_tick, right_label = max(
            pairs,
            key=lambda item: item[0],
        )
    else:
        right_tick, right_label = min(
            pairs,
            key=lambda item: item[0],
        )

    label_text = (
        right_label
        .get_text()
        .strip()
    )

    font_prop = (
        right_label
        .get_fontproperties()
    )

    try:
        text_path = TextPath(
            (0, 0),
            label_text,
            prop=font_prop,
            usetex=False,
        )

        tick_text_width_pt = float(
            text_path
            .get_extents()
            .width
        )

        if (
            not np.isfinite(
                tick_text_width_pt
            )
            or tick_text_width_pt <= 0
        ):
            raise ValueError(
                "invalid tick text width"
            )

    except Exception:

        font_size_pt = float(
            font_prop
            .get_size_in_points()
        )

        tick_text_width_pt = (
            0.56
            * font_size_pt
            * len(label_text)
        )

    UNIT_GAP_PT = 2.2

    horizontal_offset_pt = (
        tick_text_width_pt / 2.0
        + UNIT_GAP_PT
    )

    unit_transform = (
        right_label.get_transform()
        + ScaledTranslation(
            horizontal_offset_pt / 72.0,
            0.0,
            ax.figure.dpi_scale_trans,
        )
    )

    ax.text(
        right_tick,
        0.0,
        f"({unit})",
        transform=unit_transform,
        ha="left",
        va=right_label.get_va(),
        fontsize=TICK_SIZE,
        fontfamily=FONT_FAMILY,
        clip_on=False,
        zorder=400,
    )


def save_figure(
    fig,
    base_file: Path | str,
    tight: bool = True,
    pad_inches: float = 0.08,
    dpi: Optional[int] = None,
) -> None:
    base = Path(base_file).with_suffix("")
    ensure_dir(base.parent)

    kwargs = {"facecolor": "white"}
    png_dpi = FIG_DPI if dpi is None else int(dpi)

    if tight:
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"] = pad_inches

    if SAVE_PNG:
        fig.savefig(str(base) + ".png", dpi=png_dpi, **kwargs)

    if SAVE_TIF:
        try:
            fig.savefig(
                str(base) + ".tif",
                dpi=FIG_DPI,
                format="tiff",
                pil_kwargs={"compression": "tiff_lzw"},
                **kwargs,
            )
        except Exception:
            fig.savefig(
                str(base) + ".tif",
                dpi=FIG_DPI,
                format="tiff",
                **kwargs,
            )

    if SAVE_PDF:
        fig.savefig(str(base) + ".pdf", **kwargs)

    if SAVE_SVG:
        fig.savefig(str(base) + ".svg", **kwargs)

    plt.close(fig)
    gc.collect()


# =============================================================================
# 7. 数据读取
# =============================================================================
def load_selected_model() -> str:
    p = MAIN_RUN / "selected_model_for_sensitivity.json"
    with open(require_file(p, "最终模型"), "r", encoding="utf-8") as f:
        return str(json.load(f)["selected_model"])


def load_model_summary() -> pd.DataFrame:
    p = MAIN_RUN / "03_model_comparison" / "model_summary_mean_SD.csv"
    return pd.read_csv(require_file(p, "模型性能汇总"))


def load_main_contribution() -> pd.DataFrame:
    p = MAIN_RUN / "04_oof_shap" / "SHAP_contribution_bootstrap_CI_and_stability.csv"
    return pd.read_csv(require_file(p, "主分析 SHAP 贡献率"))


def load_main_obs_shap() -> pd.DataFrame:
    p = MAIN_RUN / "04_oof_shap" / "OOF_SHAP_mean_across_5_repeats_per_observation.csv"
    return pd.read_csv(require_file(p, "主分析 OOF-SHAP 观测级汇总"))


def load_main_top6() -> List[str]:
    p = MAIN_RUN / "04_oof_shap" / "Top6_features_for_ALE.csv"

    if p.exists():
        d = pd.read_csv(p).sort_values("rank")
        return d["feature"].astype(str).tolist()[:ALE_TOP_N]

    d = load_main_contribution().sort_values(
        "contribution_percent",
        ascending=False,
    )
    return d["feature"].astype(str).tolist()[:ALE_TOP_N]


def load_main_ale() -> pd.DataFrame:
    p = MAIN_RUN / "05_ALE_Top6" / "ALE_summary_95_empirical_interval.csv"

    if p.exists():
        return pd.read_csv(p)

    p2 = MAIN_RUN / "05_ALE" / "ALE_summary_95_empirical_interval.csv"
    return pd.read_csv(require_file(p2, "主分析 ALE"))


def load_shap_arrays(
    contribution: pd.DataFrame,
    observation_summary: pd.DataFrame,
) -> Tuple[List[str], pd.DataFrame, np.ndarray]:

    ordered = (
        contribution
        .sort_values("contribution_percent", ascending=False)["feature"]
        .astype(str)
        .tolist()
    )

    X = observation_summary[ordered].copy()

    shap_values = np.column_stack([
        observation_summary[f"mean_signed_SHAP_{f}"].to_numpy(float)
        for f in ordered
    ])

    return ordered, X, shap_values


# =============================================================================
# 8. 01 —— 模型性能单图
# =============================================================================
def redraw_main_model_performance_single() -> None:
    d = load_model_summary()

    specs = [
        ("Test_R2_mean", "Test_R2_SD", "Nested CV Test R²", True),
        ("Test_MSE_x1e3_mean", "Test_MSE_x1e3_SD", "Nested CV Test MSE (×10⁻³)", False),
        ("Test_RMSE_mean", "Test_RMSE_SD", "Nested CV Test RMSE", False),
        ("Test_MAE_mean", "Test_MAE_SD", "Nested CV Test MAE", False),
        ("Train_Test_R2_Gap_mean", "Train_Test_R2_Gap_SD", "Mean Train–Test R² gap", False),
    ]

    cmap = plt.get_cmap("viridis")

    for i, (metric, sd_col, ylabel, higher_is_better) in enumerate(specs):

        dd = d.sort_values(
            metric,
            ascending=not higher_is_better,
        ).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=MODEL_SINGLE_SIZE)

        x = np.arange(len(dd))

        colors = cmap(
            np.linspace(0.22, 0.86, len(dd))
        )

        bars = ax.bar(
            x,
            dd[metric],
            yerr=dd[sd_col],
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.55,
            width=0.68,
            error_kw={"elinewidth": ERRORBAR_LINE_WIDTH},
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            dd["Model"],
            rotation=18,
            ha="right",
        )

        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)

        for bar, val in zip(bars, dd[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9.5,
            )

        style_axis(ax)
        fig.tight_layout()

        save_figure(
            fig,
            MAIN_MODEL_OUT / f"Model_Comparison_{metric}",
        )


# =============================================================================
# 9. 01 —— 模型 R² + RMSE 组合
# =============================================================================
def redraw_main_model_combo() -> None:
    d = load_model_summary().sort_values(
        "Test_R2_mean",
        ascending=False,
    ).reset_index(drop=True)

    x = np.arange(len(d))
    width = 0.34

    fig, ax1 = plt.subplots(figsize=MODEL_COMBO_SIZE)
    ax2 = ax1.twinx()

    b1 = ax1.bar(
        x - width / 2,
        d["Test_R2_mean"],
        width,
        yerr=d["Test_R2_SD"],
        capsize=3,
        color="#4F7EA2",
        edgecolor="black",
        linewidth=0.5,
        label="Test R²",
    )

    b2 = ax2.bar(
        x + width / 2,
        d["Test_RMSE_mean"],
        width,
        yerr=d["Test_RMSE_SD"],
        capsize=3,
        color="#B47A64",
        edgecolor="black",
        linewidth=0.5,
        label="Test RMSE",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        d["Model"],
        rotation=18,
        ha="right",
    )

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Test R²")
    ax2.set_ylabel("Test RMSE")

    ax1.legend(
        [b1, b2],
        ["Test R²", "Test RMSE"],
        frameon=False,
        loc="best",
    )

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        MAIN_MODEL_OUT / "Model_Comparison_TestR2_RMSE_Combo",
    )


# =============================================================================
# 10. 01 —— OOF observed vs predicted
# =============================================================================
def redraw_main_oof_observed_predicted() -> None:
    model = load_selected_model()

    p = (
        MAIN_RUN
        / "02_main_nested_cv"
        / f"{model}_OOF_predictions_mean_across_5_repeats.csv"
    )

    d = pd.read_csv(require_file(p, "OOF prediction"))

    fig, ax = plt.subplots(figsize=OOF_SIZE)

    ax.scatter(
        d["observed_RI"],
        d["predicted_RI"],
        s=13,
        alpha=0.40,
        edgecolors="none",
        color="#567F9D",
    )

    lo = min(
        d["observed_RI"].min(),
        d["predicted_RI"].min(),
    )

    hi = max(
        d["observed_RI"].max(),
        d["predicted_RI"].max(),
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.0,
        color="black",
    )

    y = d["observed_RI"].to_numpy(float)
    p_ = d["predicted_RI"].to_numpy(float)

    mse = np.mean((y - p_) ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(y - p_))
    r2 = 1 - np.sum((y - p_) ** 2) / np.sum((y - y.mean()) ** 2)

    ax.text(
        0.04,
        0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.set_xlabel("Observed RI")
    ax.set_ylabel("Predicted RI")

    style_axis(ax)
    fig.tight_layout()

    save_figure(
        fig,
        MAIN_OOF_OUT / f"{model}_OOF_Observed_vs_Predicted",
    )


# =============================================================================
# 11. SHAP 贡献率 95% CI
# =============================================================================
def draw_shap_ci(
    contribution: pd.DataFrame,
    out_file: Path,
    model_name: str,
) -> None:

    d = contribution.sort_values(
        "contribution_percent",
        ascending=True,
    ).reset_index(drop=True)

    y = np.arange(len(d))
    x = d["contribution_percent"].to_numpy(float)

    xerr = np.vstack([
        x - d["CI_lower_2.5"].to_numpy(float),
        d["CI_upper_97.5"].to_numpy(float) - x,
    ])

    fig, ax = plt.subplots(figsize=SHAP_CI_SIZE)

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        markersize=6,
        capsize=4,
        linewidth=1.2,
        color="#527D9A",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        d["feature"].astype(str)
    )

    ax.set_xlabel("Relative contribution (%)")
    ax.set_ylabel("")

    for yi, xi in zip(y, x):
        ax.text(
            xi,
            yi + 0.20,
            f"{xi:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    style_axis(ax)
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.22,
        color=COLOR_GRID,
    )

    fig.tight_layout()
    save_figure(fig, out_file)


# =============================================================================
# 12. SHAP 条形单图
# =============================================================================
def draw_shap_bar(
    contribution: pd.DataFrame,
    out_file: Path,
    model_name: str,
) -> None:

    d = contribution.sort_values(
        "contribution_percent",
        ascending=False,
    ).reset_index(drop=True)

    y = np.arange(len(d))
    vals = d["contribution_percent"].to_numpy(float)

    cmap = plt.get_cmap(SHAP_BAR_CMAP)
    colors = cmap(
        np.linspace(0.22, 0.90, len(d))
    )

    fig, ax = plt.subplots(figsize=SHAP_BAR_SIZE)

    bars = ax.barh(
        y,
        vals,
        height=0.72,
        color=colors,
        edgecolor="none",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        d["feature"].astype(str)
    )

    ax.invert_yaxis()

    # 去掉最上面变量之上的多余纵轴：
    # 上边界直接贴到第一根横条的上边缘，因此左侧纵轴不会再向上多出一截。
    first_bar_top = bars[0].get_y()
    ax.set_ylim(
        len(d) - 0.5,
        first_bar_top,
    )

    maxv = float(vals.max())

    ax.set_xlim(
        0,
        maxv * 1.20,
    )

    ax.set_xlabel(
        "Relative mean |SHAP| importance (%)"
    )

    for bar, value in zip(bars, vals):

        ax.text(
            value + maxv * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=BAR_VALUE_SIZE,
        )

    style_axis(ax)
    fig.tight_layout()

    save_figure(fig, out_file)


# =============================================================================
# 13. SHAP beeswarm 单图
# =============================================================================
def draw_shap_beeswarm(
    contribution: pd.DataFrame,
    obs: pd.DataFrame,
    out_file: Path,
    model_name: str,
) -> None:

    if shap is None:
        print("[跳过] shap 未安装，无法重绘 beeswarm。")
        return

    ordered, X, shap_values = load_shap_arrays(
        contribution,
        obs,
    )

    fig = plt.figure(figsize=SHAP_BEE_SIZE)

    ax = fig.add_axes([
        0.14,
        0.14,
        0.70,
        0.78,
    ])

    plt.sca(ax)

    before = set(fig.axes)

    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_type="dot",
        max_display=len(ordered),
        sort=False,
        cmap=SHAP_BEE_CMAP,
        plot_size=None,
    )

    created = [
        a for a in fig.axes
        if a not in before
    ]

    cbar = created[-1] if created else None

    # 主轴加宽，减少右侧空白
    ax.set_position([
        0.14,
        0.14,
        0.70,
        0.78,
    ])

    if cbar is not None:
        cbar.set_position([
            0.855,
            0.17,
            0.016,
            0.70,
        ])

        cbar.set_ylabel(
            "Feature value",
            fontsize=9,
        )

    ax.set_xlabel(
        "SHAP value (impact on model output)"
    )

    ax.set_ylabel("Features")

    style_beeswarm_axis(ax)

    save_figure(
        fig,
        out_file,
        tight=False,
    )


# =============================================================================
# 14. SHAP 条形 + beeswarm 组合
# =============================================================================
def draw_shap_bar_beeswarm_combo(
    contribution: pd.DataFrame,
    obs: pd.DataFrame,
    out_file: Path,
    model_name: str,
) -> None:

    if shap is None:
        return

    d = contribution.sort_values(
        "contribution_percent",
        ascending=False,
    ).reset_index(drop=True)

    ordered, X, shap_values = load_shap_arrays(
        contribution,
        obs,
    )

    fig = plt.figure(figsize=SHAP_COMBO_SIZE)

    # --------------------
    # 左侧条形图
    # --------------------
    ax1 = fig.add_axes([
        0.055,
        0.14,
        0.39,
        0.76,
    ])

    y = np.arange(len(d))
    vals = d["contribution_percent"].to_numpy(float)

    cmap = plt.get_cmap(SHAP_BAR_CMAP)

    colors = cmap(
        np.linspace(0.22, 0.90, len(d))
    )

    bars = ax1.barh(
        y,
        vals,
        height=0.72,
        color=colors,
        edgecolor="none",
    )

    ax1.set_yticks(y)
    ax1.set_yticklabels(
        d["feature"].astype(str)
    )

    ax1.invert_yaxis()

    # TMX 上方不再保留任何多余纵轴：
    # 纵轴顶部与第一根横条上边缘完全齐平。
    first_bar_top = bars[0].get_y()
    ax1.set_ylim(
        len(d) - 0.5,
        first_bar_top,
    )

    maxv = float(vals.max())

    ax1.set_xlim(
        0,
        maxv * 1.22,
    )

    ax1.set_xlabel(
        "Relative mean |SHAP| importance (%)"
    )

    for bar, value in zip(bars, vals):

        ax1.text(
            value + maxv * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=BAR_VALUE_SIZE,
        )

    style_axis(ax1)

    # --------------------
    # 右侧 beeswarm
    # --------------------
    ax2 = fig.add_axes([
        0.505,
        0.14,
        0.36,
        0.76,
    ])

    plt.sca(ax2)

    before = set(fig.axes)

    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_type="dot",
        max_display=len(ordered),
        sort=False,
        cmap=SHAP_BEE_CMAP,
        plot_size=None,
    )

    created = [
        a for a in fig.axes
        if a not in before
    ]

    cbar = created[-1] if created else None

    ax2.set_position([
        0.505,
        0.14,
        0.36,
        0.76,
    ])

    if cbar is not None:

        cbar.set_position([
            0.882,
            0.16,
            0.014,
            0.71,
        ])

        cbar.set_ylabel(
            "Feature value",
            fontsize=9,
        )

    ax2.set_xlabel(
        "SHAP value (impact on model output)"
    )

    ax2.set_ylabel("Features")

    style_beeswarm_axis(ax2)

    # 子图编号统一放在两个子图内容区上方：
    # (a) 不再压到 TMX 横条或纵轴上；
    # (a) 与 (b) 使用完全相同的 y 坐标，保证水平对齐。
    panel_y = 0.925

    fig.text(
        0.055,
        panel_y,
        "(a)",
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
        zorder=500,
    )

    fig.text(
        0.505,
        panel_y,
        "(b)",
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
        zorder=500,
    )

    save_figure(
        fig,
        out_file,
        tight=False,
    )


# =============================================================================
# 15. SHAP rose
# =============================================================================
def rose_text_rotation(angle_degrees: float) -> float:
    angle = float(angle_degrees) % 360.0
    rotation = 90.0 - angle

    while rotation > 90.0:
        rotation -= 180.0

    while rotation < -90.0:
        rotation += 180.0

    if 120.0 <= angle <= 240.0:
        rotation = float(np.clip(rotation, -15.0, 15.0))
    else:
        rotation = float(np.clip(rotation, -30.0, 30.0))

    return rotation


def draw_rose_axis(
    ax,
    contribution: pd.DataFrame,
) -> None:

    d = contribution.sort_values(
        "contribution_percent",
        ascending=False,
    ).reset_index(drop=True)

    vals = d["contribution_percent"].to_numpy(float)
    raw = d["overall_mean_abs_SHAP_repeat_averaged"].to_numpy(float)
    names = d["feature"].astype(str).tolist()

    n = len(d)

    theta = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False,
    )

    width = 2 * np.pi / n * 0.82

    vmax = float(vals.max())
    vmin = float(vals.min())

    inner = max(
        15.0,
        vmax * 0.62,
    )

    gap = max(
        0.55,
        vmax * 0.018,
    )

    bottom = inner + gap

    cmap = plt.get_cmap("Spectral_r")

    norm = mcolors.Normalize(
        vmin=vmin,
        vmax=vmax if vmax != vmin else vmin + 1e-9,
    )

    ax.bar(
        theta,
        vals,
        width=width,
        bottom=bottom,
        color=cmap(norm(vals)),
        edgecolor="black",
        linewidth=0.9,
        zorder=20,
    )

    circ = np.linspace(
        0,
        2 * np.pi,
        800,
    )

    ax.fill_between(
        circ,
        0,
        inner,
        color="white",
        zorder=40,
    )

    ax.plot(
        circ,
        np.full_like(circ, inner),
        color="black",
        linewidth=1.0,
        zorder=60,
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_ylim(
        0,
        bottom + vmax + max(10, vmax * 0.26),
    )

    ax.set_axis_off()

    for i, (ang, val, rawv, name) in enumerate(
        zip(theta, vals, raw, names)
    ):

        txt = ax.text(
            ang,
            bottom + val * 0.50,
            f"{rawv:.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9.0,
            fontweight="bold",
            zorder=100,
        )

        txt.set_path_effects([
            pe.withStroke(
                linewidth=1.4,
                foreground="black",
            )
        ])

        deg = np.degrees(ang)

        lab = ax.text(
            ang,
            bottom + val + max(4.8, vmax * 0.13),
            f"{name}\n{val:.2f}%",
            ha="center",
            va="center",
            rotation=rose_text_rotation(deg),
            rotation_mode="anchor",
            fontsize=10.2,
            fontweight="bold",
            zorder=120,
            clip_on=False,
        )

        lab.set_path_effects([
            pe.withStroke(
                linewidth=1.8,
                foreground="white",
            )
        ])


def draw_shap_rose(
    contribution: pd.DataFrame,
    out_file: Path,
) -> None:

    fig = plt.figure(
        figsize=SHAP_ROSE_SIZE
    )

    ax = fig.add_axes(
        [0.03, 0.03, 0.94, 0.94],
        projection="polar",
    )

    draw_rose_axis(
        ax,
        contribution,
    )

    save_figure(
        fig,
        out_file,
        tight=False,
    )


# =============================================================================
# 16. SHAP rose + beeswarm
# =============================================================================
def draw_shap_rose_bee(
    contribution: pd.DataFrame,
    obs: pd.DataFrame,
    out_file: Path,
) -> None:

    if shap is None:
        return

    ordered, X, shap_values = load_shap_arrays(
        contribution,
        obs,
    )

    fig = plt.figure(
        figsize=SHAP_ROSE_BEE_SIZE
    )

    axr = fig.add_axes(
        [0.015, 0.035, 0.47, 0.92],
        projection="polar",
    )

    draw_rose_axis(
        axr,
        contribution,
    )

    axbee = fig.add_axes(
        [0.54, 0.16, 0.32, 0.70]
    )

    plt.sca(axbee)

    before = set(fig.axes)

    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_type="dot",
        max_display=len(ordered),
        sort=False,
        cmap=SHAP_BEE_CMAP,
        plot_size=None,
    )

    created = [
        a for a in fig.axes
        if a not in before
    ]

    cbar = created[-1] if created else None

    axbee.set_position(
        [0.54, 0.16, 0.32, 0.70]
    )

    if cbar is not None:
        cbar.set_position(
            [0.885, 0.17, 0.014, 0.68]
        )

    axbee.set_xlabel(
        "SHAP value (impact on model output)"
    )

    axbee.set_ylabel("Features")

    style_axis(axbee)

    fig.text(
        0.02,
        0.965,
        "(a)",
        fontsize=PANEL_LABEL_SIZE + 1,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
    )

    fig.text(
        0.515,
        0.965,
        "(b)",
        fontsize=PANEL_LABEL_SIZE + 1,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
    )

    save_figure(
        fig,
        out_file,
        tight=False,
    )


# =============================================================================
# 17. SHAP dependence
# =============================================================================
def add_binned_smooth_line(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    bins: int = 40,
) -> None:

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 20:
        return

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    edges = np.linspace(
        0,
        len(x),
        bins + 1,
    ).astype(int)

    xx = []
    yy = []

    for i in range(bins):
        s = edges[i]
        e = edges[i + 1]

        if e > s:
            xx.append(
                np.mean(x[s:e])
            )
            yy.append(
                np.mean(y[s:e])
            )

    ax.plot(
        xx,
        yy,
        color="#D66B61",
        linewidth=1.9,
        label="Binned smooth curve",
    )


def draw_shap_dependence(
    contribution: pd.DataFrame,
    obs: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    top_n: int = 6,
) -> None:

    top_features = (
        contribution
        .sort_values("contribution_percent", ascending=False)["feature"]
        .astype(str)
        .tolist()[:top_n]
    )

    # 2 × 3 组合图
    fig, axes = plt.subplots(
        2,
        3,
        figsize=SHAP_DEP_COMBO_SIZE,
    )

    axes = axes.ravel()

    for i, feat in enumerate(top_features):

        ax = axes[i]

        x = obs[feat].to_numpy(float)

        y = obs[
            f"mean_signed_SHAP_{feat}"
        ].to_numpy(float)

        ax.scatter(
            x,
            y,
            s=15,
            alpha=0.50,
            color="#5E87A3",
            edgecolors="none",
        )

        add_binned_smooth_line(
            ax,
            x,
            y,
            bins=40,
        )

        ax.axhline(
            0,
            color=COLOR_ZERO,
            linestyle="-.",
            linewidth=0.8,
        )

        set_feature_xaxis(
            ax,
            feat,
            y=-0.115,
        )

        ax.set_ylabel("SHAP value")
        add_panel_label(ax, i)
        style_axis(ax)

    fig.subplots_adjust(
        left=0.065,
        right=0.955,
        bottom=0.12,
        top=0.96,
        wspace=0.30,
        hspace=0.38,
    )

    save_figure(
        fig,
        out_dir / f"{prefix}_SHAP_Dependence_Top6_Combo",
        tight=False,
    )

    # 单图
    for i, feat in enumerate(top_features):

        fig, ax = plt.subplots(
            figsize=SHAP_DEP_SINGLE_SIZE
        )

        x = obs[feat].to_numpy(float)
        y = obs[
            f"mean_signed_SHAP_{feat}"
        ].to_numpy(float)

        ax.scatter(
            x,
            y,
            s=15,
            alpha=0.50,
            color="#5E87A3",
            edgecolors="none",
        )

        add_binned_smooth_line(
            ax,
            x,
            y,
            bins=40,
        )

        ax.axhline(
            0,
            color=COLOR_ZERO,
            linestyle="-.",
            linewidth=0.8,
        )

        set_feature_xaxis(
            ax,
            feat,
            y=-0.115,
        )

        ax.set_ylabel("SHAP value")
        style_axis(ax)

        fig.tight_layout()

        save_figure(
            fig,
            out_dir / f"{prefix}_SHAP_Dependence_{feat}",
        )


# =============================================================================
# 18. 统一 SHAP 全套图
# =============================================================================
def redraw_shap_suite(
    contribution: pd.DataFrame,
    obs: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    model_name: str,
    dependence_top_n: int = 6,
) -> None:

    ensure_dir(out_dir)

    draw_shap_ci(
        contribution,
        out_dir / f"{prefix}_SHAP_Contribution_95CI",
        model_name,
    )

    draw_shap_bar(
        contribution,
        out_dir / f"{prefix}_SHAP_Bar_Single",
        model_name,
    )

    draw_shap_beeswarm(
        contribution,
        obs,
        out_dir / f"{prefix}_SHAP_Beeswarm_Single",
        model_name,
    )

    draw_shap_bar_beeswarm_combo(
        contribution,
        obs,
        out_dir / f"{prefix}_SHAP_Bar_Beeswarm_Combo",
        model_name,
    )

    draw_shap_rose(
        contribution,
        out_dir / f"{prefix}_SHAP_Rose_Single",
    )

    draw_shap_rose_bee(
        contribution,
        obs,
        out_dir / f"{prefix}_SHAP_Rose_Beeswarm_Combo",
    )

    draw_shap_dependence(
        contribution,
        obs,
        out_dir,
        prefix,
        top_n=dependence_top_n,
    )


# =============================================================================
# 19. ALE 单图（2.5th–97.5th percentile interval）
# =============================================================================
def draw_ale_single(
    ale_summary: pd.DataFrame,
    feature: str,
    out_file: Path,
) -> None:

    g = ale_summary[
        ale_summary["feature"].astype(str) == feature
    ].sort_values("x")

    if g.empty:
        return

    style = get_feature_style(feature)

    fig, ax = plt.subplots(
        figsize=ALE_SINGLE_SIZE
    )

    ax.fill_between(
        g["x"].to_numpy(float),
        g["q2.5"].to_numpy(float),
        g["q97.5"].to_numpy(float),
        color=style["fill"],
        alpha=CI_ALPHA,
        linewidth=0,
    )

    ax.plot(
        g["x"].to_numpy(float),
        g["mean_ALE"].to_numpy(float),
        color=style["line"],
        linewidth=ALE_LINE_WIDTH,
    )

    ax.axhline(
        0,
        color=COLOR_ZERO,
        linestyle="-.",
        linewidth=ZERO_LINE_WIDTH,
    )

    # 不再在图顶部重复变量名
    set_feature_xaxis(
        ax,
        feature,
        y=-0.115,
    )

    ax.set_ylabel("ALE effect")
    ax.yaxis.set_major_formatter(
        FuncFormatter(format_three_decimal)
    )

    # 每个子图都有图例
    handles = [
        Patch(
            facecolor=style["fill"],
            edgecolor="none",
            alpha=0.85,
            label=ALE_INTERVAL_LABEL,
        ),
        Line2D(
            [0],
            [0],
            color=style["line"],
            linewidth=ALE_LINE_WIDTH,
            label="Mean ALE",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        columnspacing=0.7,
    )

    style_axis(ax)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.18,
        color=COLOR_GRID,
    )

    fig.tight_layout()

    save_figure(
        fig,
        out_file,
    )


# =============================================================================
# 20. ALE 2 × 3 六子图（2.5th–97.5th percentile interval）
# =============================================================================
def draw_ale_top6_combo(
    ale_summary: pd.DataFrame,
    features: Sequence[str],
    out_file: Path,
    word_ready: bool = False,
) -> None:

    features = list(features)[:6]

    text_scale = ALE_WORD_TEXT_SCALE if word_ready else 1.0
    combo_size = ALE_WORD_COMBO_SIZE if word_ready else ALE_COMBO_SIZE

    fig, axes = plt.subplots(
        2,
        3,
        figsize=combo_size,
    )

    axes = axes.ravel()

    for i, feature in enumerate(features):

        ax = axes[i]

        g = ale_summary[
            ale_summary["feature"].astype(str) == feature
        ].sort_values("x")

        if g.empty:
            continue

        style = get_feature_style(feature)

        x = g["x"].to_numpy(float)
        mean = g["mean_ALE"].to_numpy(float)
        low = g["q2.5"].to_numpy(float)
        high = g["q97.5"].to_numpy(float)

        ax.fill_between(
            x,
            low,
            high,
            color=style["fill"],
            alpha=CI_ALPHA,
            linewidth=0,
        )

        ax.plot(
            x,
            mean,
            color=style["line"],
            linewidth=ALE_LINE_WIDTH,
        )

        ax.axhline(
            0,
            color=COLOR_ZERO,
            linestyle="-.",
            linewidth=ZERO_LINE_WIDTH,
        )

        if word_ready and feature in ALE_WORD_REFERENCE_TICKS:
            # set_xticks / set_yticks 默认会扩展坐标范围，因此先保存、后恢复
            # 原数据范围，只保留旧版在可视区域内的刻度。
            original_xlim = ax.get_xlim()
            original_ylim = ax.get_ylim()
            reference_ticks = ALE_WORD_REFERENCE_TICKS[feature]
            ax.set_xticks(reference_ticks["x"])
            ax.set_yticks(reference_ticks["y"])
            ax.set_xlim(original_xlim)
            ax.set_ylim(original_ylim)
            # 仅统一显示格式，刻度位置和纵轴范围保持旧版不变。
            ax.yaxis.set_major_formatter(
                FuncFormatter(format_three_decimal)
            )
            # 只在纵轴顶部增加少量视觉留白；原有数据范围、刻度位置不变。
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.075)

        # 顶部不显示重复名称
        ax.set_title("")

        set_feature_xaxis(
            ax,
            feature,
            y=-0.115,
            font_size=AXIS_LABEL_SIZE * text_scale,
        )

        ax.set_ylabel(
            "ALE effect",
            fontsize=AXIS_LABEL_SIZE * text_scale,
        )

        ax.tick_params(
            axis="both",
            labelsize=TICK_SIZE * text_scale,
        )

        # 每个子图都独立有图例
        handles = [
            Patch(
                facecolor=style["fill"],
                edgecolor="none",
                alpha=0.85,
                label=ALE_INTERVAL_LABEL,
            ),
            Line2D(
                [0],
                [0],
                color=style["line"],
                linewidth=ALE_LINE_WIDTH,
                label="Mean ALE",
            ),
        ]

        # 每个子图保留与自身曲线、置信区间同色的独立图例。
        legend = ax.legend(
            handles=handles,
            loc="upper center",
            # Word 版图例留在子图上部；不透明底板确保阴影不会透入图例。
            bbox_to_anchor=(0.5, 0.915 if word_ready else 1.03),
            ncol=2,
            frameon=word_ready,
            facecolor="white",
            edgecolor="none",
            framealpha=1.0 if word_ready else None,
            fontsize=LEGEND_SIZE * (1.0 if word_ready else 1.0),
            handlelength=1.5,
            handletextpad=0.35,
            columnspacing=0.65,
            borderaxespad=0,
        )

        if word_ready:
            # 编号位于图例上方，二者保持独立空间。
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(1.0)
            legend.set_zorder(1000)

        add_panel_label(
            ax,
            i,
            y=1.035 if word_ready else 0.985,
            # 与 SHAP 组合图在 Word 等宽插入后的视觉字号对齐。
            fontsize=(
                ALE_WORD_PANEL_LABEL_SIZE
                if word_ready else PANEL_LABEL_SIZE
            ),
        )
        style_axis(ax)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.18,
            color=COLOR_GRID,
        )

    fig.subplots_adjust(
        left=0.075 if word_ready else 0.065,
        right=0.975 if word_ready else 0.955,
        bottom=0.11 if word_ready else 0.12,
        top=0.93 if word_ready else 0.95,
        wspace=0.32 if word_ready else 0.30,
        hspace=0.54 if word_ready else 0.42,
    )

    save_figure(
        fig,
        out_file,
        tight=False,
        dpi=WORD_ALE_DPI,
    )


# =============================================================================
# 21. ALE 全套
# =============================================================================
def redraw_ale_suite(
    ale_summary: pd.DataFrame,
    features: Sequence[str],
    out_dir: Path,
    prefix: str,
    word_ready_combo: bool = False,
    combo_suffix: str = "",
) -> None:

    ensure_dir(out_dir)

    features = list(features)[:6]

    draw_ale_top6_combo(
        ale_summary,
        features,
        out_dir / f"{prefix}_ALE_Top6_2x3{combo_suffix}",
        word_ready=word_ready_combo,
    )

    for feature in features:

        draw_ale_single(
            ale_summary,
            feature,
            out_dir / f"{prefix}_ALE_{feature}_Single",
        )


# =============================================================================
# 22. Main vs sensitivity 性能图
# =============================================================================
def draw_performance_comparison(
    perf_table: pd.DataFrame,
    out_file: Path,
    sensitivity_label: str,
) -> None:

    d = perf_table[
        perf_table["Metric"].isin(
            ["Test_R2", "Test_RMSE", "Test_MAE"]
        )
    ].copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=SENS_PERF_SIZE,
    )

    for i, (_, row) in enumerate(
        d.iterrows()
    ):

        ax = axes[i]

        vals = [
            row["Main_mean"],
            row["Sensitivity_mean"],
        ]

        errs = [
            row["Main_SD"],
            row["Sensitivity_SD"],
        ]

        ax.bar(
            [0, 1],
            vals,
            yerr=errs,
            capsize=4,
            color=[COLOR_MAIN, COLOR_SENS],
            edgecolor="black",
            linewidth=0.5,
            width=0.62,
        )

        ax.set_xticks([0, 1])

        ax.set_xticklabels([
            "Main",
            sensitivity_label,
        ])

        ax.set_ylabel(
            str(row["Metric"]).replace("Test_", "")
        )

        add_panel_label(ax, i)
        style_axis(ax)

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.19,
        top=0.96,
        wspace=0.32,
    )

    save_figure(
        fig,
        out_file,
        tight=False,
    )


# =============================================================================
# 23. SHAP contribution comparison
# =============================================================================
def draw_shap_contribution_comparison(
    main_contrib: pd.DataFrame,
    sens_contrib: pd.DataFrame,
    features: Sequence[str],
    out_file: Path,
    sensitivity_label: str,
) -> None:

    ma = main_contrib.set_index("feature")
    se = sens_contrib.set_index("feature")

    feats = [
        f for f in features
        if f in ma.index and f in se.index
    ]

    x = np.arange(len(feats))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(8.6, 4.8)
    )

    ax.bar(
        x - width / 2,
        [
            ma.loc[f, "contribution_percent"]
            for f in feats
        ],
        width,
        label="Main",
        color=COLOR_MAIN,
        edgecolor="black",
        linewidth=0.45,
    )

    ax.bar(
        x + width / 2,
        [
            se.loc[f, "contribution_percent"]
            for f in feats
        ],
        width,
        label=sensitivity_label,
        color=COLOR_SENS,
        edgecolor="black",
        linewidth=0.45,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(feats)

    ax.set_ylabel(
        "Relative contribution (%)"
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    style_axis(ax)
    fig.tight_layout()

    save_figure(
        fig,
        out_file,
    )


# =============================================================================
# 24. Main vs No-Year ALE
# =============================================================================
def draw_ale_main_vs_sensitivity(
    main_summary: pd.DataFrame,
    sens_summary: pd.DataFrame,
    features: Sequence[str],
    out_file: Path,
    sensitivity_label: str,
) -> None:

    features = list(features)[:6]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=ALE_COMPARE_SIZE,
    )

    axes = axes.ravel()

    for i, feat in enumerate(features):

        ax = axes[i]

        a = main_summary[
            main_summary["feature"].astype(str) == feat
        ].sort_values("x")

        b = sens_summary[
            sens_summary["feature"].astype(str) == feat
        ].sort_values("x")

        style = get_feature_style(feat)

        if len(a):
            ax.plot(
                a["x"],
                a["mean_ALE"],
                color=style["line"],
                linewidth=ALE_LINE_WIDTH,
            )

        if len(b):
            ax.plot(
                b["x"],
                b["mean_ALE"],
                color=style["compare"],
                linewidth=COMPARE_LINE_WIDTH,
                linestyle="--",
            )

        ax.axhline(
            0,
            color=COLOR_ZERO,
            linestyle="-.",
            linewidth=ZERO_LINE_WIDTH,
        )

        ax.set_title("")

        set_feature_xaxis(
            ax,
            feat,
            y=-0.115,
        )

        ax.set_ylabel("ALE effect")

        handles = [
            Line2D(
                [0],
                [0],
                color=style["line"],
                linewidth=ALE_LINE_WIDTH,
                label="Main",
            ),
            Line2D(
                [0],
                [0],
                color=style["compare"],
                linewidth=COMPARE_LINE_WIDTH,
                linestyle="--",
                label=sensitivity_label,
            ),
        ]

        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            frameon=False,
            fontsize=LEGEND_SIZE,
            handlelength=1.5,
            columnspacing=0.7,
        )

        # (a) 的 No-Year 虚线从左上角起始，标签移至坐标轴上方，
        # 避免与曲线重合；其余子图位置保持不变。
        if i == 0:
            add_panel_label(ax, i, x=0.012, y=1.018)
        else:
            add_panel_label(ax, i)
        style_axis(ax)

    fig.subplots_adjust(
        left=0.065,
        right=0.955,
        bottom=0.12,
        top=0.95,
        wspace=0.30,
        hspace=0.42,
    )

    save_figure(
        fig,
        out_file,
        tight=False,
    )


# =============================================================================
# 25. 01 主分析所有图
# =============================================================================
def redraw_all_main_figures() -> None:

    model = load_selected_model()

    print("[01] 模型性能图")
    redraw_main_model_performance_single()
    redraw_main_model_combo()

    print("[01] OOF prediction 图")
    redraw_main_oof_observed_predicted()

    print("[01] SHAP 全套图")
    contribution = load_main_contribution()
    obs = load_main_obs_shap()

    redraw_shap_suite(
        contribution,
        obs,
        MAIN_SHAP_OUT,
        prefix=model,
        model_name=model,
        dependence_top_n=6,
    )

    print("[01] ALE Top-6 全套图")
    ale = load_main_ale()
    top6 = load_main_top6()

    redraw_ale_suite(
        ale,
        top6,
        MAIN_ALE_OUT,
        prefix=model,
        word_ready_combo=True,
        combo_suffix="_SHAPVisualLabelAligned",
    )


# =============================================================================
# 26. 02-A No-Year 全部图
# =============================================================================
def redraw_noyear_figures() -> None:

    model = load_selected_model()

    perf_path = (
        SENS_RUN
        / "01_noYear"
        / "NoYear_vs_Main_performance.csv"
    )

    perf = optional_csv(perf_path)

    if perf is not None:
        draw_performance_comparison(
            perf,
            NOYEAR_PERF_OUT / "NoYear_vs_Main_Performance",
            "No-Year",
        )

    # No-Year SHAP 本身
    ny_contrib_path = (
        SENS_RUN
        / "01_noYear"
        / "oof_shap"
        / "SHAP_contribution_bootstrap_CI_and_stability.csv"
    )

    ny_obs_path = (
        SENS_RUN
        / "01_noYear"
        / "oof_shap"
        / "OOF_SHAP_mean_across_5_repeats_per_observation.csv"
    )

    ny_contrib = optional_csv(ny_contrib_path)
    ny_obs = optional_csv(ny_obs_path)

    if ny_contrib is not None and ny_obs is not None:

        redraw_shap_suite(
            ny_contrib,
            ny_obs,
            NOYEAR_SHAP_OUT,
            prefix=f"NoYear_{model}",
            model_name=model,
            dependence_top_n=6,
        )

        draw_shap_contribution_comparison(
            load_main_contribution(),
            ny_contrib,
            ["TMX", "SR", "TMN", "SM", "PRE", "GI", "VPD"],
            NOYEAR_SHAP_OUT / "NoYear_vs_Main_SHAP_Contribution",
            "No-Year",
        )

    # No-Year ALE
    ny_ale_path = (
        SENS_RUN
        / "01_noYear"
        / "ALE"
        / "ALE_summary_95_empirical_interval.csv"
    )

    ny_ale = optional_csv(ny_ale_path)

    if ny_ale is not None:

        top6 = load_main_top6()

        redraw_ale_suite(
            ny_ale,
            top6,
            NOYEAR_ALE_OUT,
            prefix=f"NoYear_{model}",
        )

        draw_ale_main_vs_sensitivity(
            load_main_ale(),
            ny_ale,
            top6,
            NOYEAR_ALE_OUT / "NoYear_vs_Main_ALE_Top6_2x3",
            "No-Year",
        )


# =============================================================================
# 27. 02-B 100 km Block CV
# =============================================================================
def redraw_block100_figures() -> None:

    perf_path = (
        SENS_RUN
        / "02_100km_BlockCV"
        / "Block100_vs_Main_performance_and_delta.csv"
    )

    perf = optional_csv(perf_path)

    if perf is not None:

        draw_performance_comparison(
            perf,
            BLOCK_PERF_OUT / "Block100_vs_Main_Performance",
            "100 km",
        )


# =============================================================================
# 28. 02-C Temperature Tmean + DTR
# =============================================================================
def redraw_temperature_figures() -> None:

    model = load_selected_model()

    perf_path = (
        SENS_RUN
        / "03_temperature_Tmean_DTR"
        / "Temperature_vs_Main_performance.csv"
    )

    perf = optional_csv(perf_path)

    if perf is not None:

        draw_performance_comparison(
            perf,
            TEMP_PERF_OUT / "Temperature_vs_Main_Performance",
            "Tmean+DTR",
        )

    # 温度敏感性 SHAP 本身
    temp_contrib_path = (
        SENS_RUN
        / "03_temperature_Tmean_DTR"
        / "oof_shap"
        / "SHAP_contribution_bootstrap_CI_and_stability.csv"
    )

    temp_obs_path = (
        SENS_RUN
        / "03_temperature_Tmean_DTR"
        / "oof_shap"
        / "OOF_SHAP_mean_across_5_repeats_per_observation.csv"
    )

    temp_contrib = optional_csv(
        temp_contrib_path
    )

    temp_obs = optional_csv(
        temp_obs_path
    )

    if temp_contrib is not None and temp_obs is not None:

        redraw_shap_suite(
            temp_contrib,
            temp_obs,
            TEMP_SHAP_OUT,
            prefix=f"TmeanDTR_{model}",
            model_name=model,
            dependence_top_n=6,
        )

        # 非温度变量贡献对比
        draw_shap_contribution_comparison(
            load_main_contribution(),
            temp_contrib,
            ["SM", "PRE", "SR", "VPD", "GI"],
            TEMP_SHAP_OUT / "Temperature_nonTemp_SHAP_Contribution",
            "Tmean+DTR",
        )

    # 温度组总贡献
    group_path = (
        SENS_RUN
        / "03_temperature_Tmean_DTR"
        / "Temperature_group_contribution_comparison.csv"
    )

    group = optional_csv(group_path)

    if group is not None and len(group):

        row = group.iloc[0]

        vals = [
            float(
                row[
                    "Main_temperature_group_contribution_percent"
                ]
            ),
            float(
                row[
                    "Sensitivity_temperature_group_contribution_percent"
                ]
            ),
        ]

        labels = [
            "TMX + TMN",
            "Tmean + DTR",
        ]

        fig, ax = plt.subplots(
            figsize=(5.8, 4.4)
        )

        bars = ax.bar(
            [0, 1],
            vals,
            color=[
                "#C16C52",
                "#6888A5",
            ],
            edgecolor="black",
            linewidth=0.5,
            width=0.62,
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)

        ax.set_ylabel(
            "Temperature-group contribution (%)"
        )

        ymax = max(vals)

        ax.set_ylim(
            0,
            ymax * 1.16,
        )

        for b, v in zip(bars, vals):

            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + ymax * 0.025,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        style_axis(ax)
        fig.tight_layout()

        save_figure(
            fig,
            TEMP_SHAP_OUT / "Temperature_Group_Contribution",
        )

    # 温度敏感性 ALE
    temp_ale_path = (
        SENS_RUN
        / "03_temperature_Tmean_DTR"
        / "ALE"
        / "ALE_summary_95_empirical_interval.csv"
    )

    temp_ale = optional_csv(
        temp_ale_path
    )

    if temp_ale is not None:

        # 根据敏感性自己的 SHAP 排名自动取前 6
        if temp_contrib is not None:

            temp_top6 = (
                temp_contrib
                .sort_values(
                    "contribution_percent",
                    ascending=False,
                )["feature"]
                .astype(str)
                .tolist()[:6]
            )

        else:

            temp_top6 = (
                temp_ale["feature"]
                .drop_duplicates()
                .astype(str)
                .tolist()[:6]
            )

        redraw_ale_suite(
            temp_ale,
            temp_top6,
            TEMP_ALE_OUT,
            prefix=f"TmeanDTR_{model}",
        )


# =============================================================================
# 29. 02 三敏感性总体性能汇总图
# =============================================================================
def redraw_all_sensitivity_performance_overview() -> None:

    files = [
        (
            "No-Year",
            SENS_RUN
            / "01_noYear"
            / "NoYear_vs_Main_performance.csv",
        ),
        (
            "100 km",
            SENS_RUN
            / "02_100km_BlockCV"
            / "Block100_vs_Main_performance_and_delta.csv",
        ),
        (
            "Tmean+DTR",
            SENS_RUN
            / "03_temperature_Tmean_DTR"
            / "Temperature_vs_Main_performance.csv",
        ),
    ]

    rows = []

    for module, p in files:

        d = optional_csv(p)

        if d is None:
            continue

        d = d.copy()
        d["Module"] = module
        rows.append(d)

    if not rows:
        return

    all_perf = pd.concat(
        rows,
        ignore_index=True,
    )

    metrics = [
        "Test_R2",
        "Test_RMSE",
        "Test_MAE",
    ]

    modules = [
        "No-Year",
        "100 km",
        "Tmean+DTR",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=SENS_ALL_PERF_SIZE,
    )

    for i, metric in enumerate(metrics):

        ax = axes[i]

        dm = all_perf[
            all_perf["Metric"] == metric
        ]

        x = np.arange(len(modules))
        width = 0.34

        main_means = []
        main_sds = []
        sens_means = []
        sens_sds = []

        for module in modules:

            row = dm[
                dm["Module"] == module
            ]

            if len(row):

                r = row.iloc[0]

                main_means.append(
                    float(r["Main_mean"])
                )

                main_sds.append(
                    float(r["Main_SD"])
                )

                sens_means.append(
                    float(r["Sensitivity_mean"])
                )

                sens_sds.append(
                    float(r["Sensitivity_SD"])
                )

            else:
                main_means.append(np.nan)
                main_sds.append(np.nan)
                sens_means.append(np.nan)
                sens_sds.append(np.nan)

        ax.bar(
            x - width / 2,
            main_means,
            width,
            yerr=main_sds,
            capsize=3,
            color=COLOR_MAIN,
            edgecolor="black",
            linewidth=0.45,
            label="Main",
        )

        ax.bar(
            x + width / 2,
            sens_means,
            width,
            yerr=sens_sds,
            capsize=3,
            color=COLOR_SENS,
            edgecolor="black",
            linewidth=0.45,
            label="Sensitivity",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            modules,
            rotation=15,
            ha="right",
        )

        ax.set_ylabel(
            metric.replace("Test_", "")
        )

        add_panel_label(
            ax,
            i,
        )

        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.24,
        top=0.87,
        wspace=0.32,
    )

    save_figure(
        fig,
        SENS_OUT / "00_Sensitivity_Performance_Overview",
        tight=False,
    )


# =============================================================================
# 30. 绘图数据快照
# =============================================================================
def export_plot_data_snapshot() -> None:

    ensure_dir(
        PLOT_DATA_OUT
    )

    files = {
        "01_Main_Model_Performance.csv":
            MAIN_RUN
            / "03_model_comparison"
            / "model_summary_mean_SD.csv",

        "01_Main_SHAP_Contribution.csv":
            MAIN_RUN
            / "04_oof_shap"
            / "SHAP_contribution_bootstrap_CI_and_stability.csv",

        "01_Main_SHAP_Observation_Mean.csv":
            MAIN_RUN
            / "04_oof_shap"
            / "OOF_SHAP_mean_across_5_repeats_per_observation.csv",

        "01_Main_ALE_Top6.csv":
            MAIN_RUN
            / "05_ALE_Top6"
            / "ALE_summary_95_empirical_interval.csv",

        "02_NoYear_Performance.csv":
            SENS_RUN
            / "01_noYear"
            / "NoYear_vs_Main_performance.csv",

        "02_NoYear_SHAP.csv":
            SENS_RUN
            / "01_noYear"
            / "oof_shap"
            / "SHAP_contribution_bootstrap_CI_and_stability.csv",

        "02_NoYear_ALE.csv":
            SENS_RUN
            / "01_noYear"
            / "ALE"
            / "ALE_summary_95_empirical_interval.csv",

        "02_Block100_Performance.csv":
            SENS_RUN
            / "02_100km_BlockCV"
            / "Block100_vs_Main_performance_and_delta.csv",

        "02_Temperature_Performance.csv":
            SENS_RUN
            / "03_temperature_Tmean_DTR"
            / "Temperature_vs_Main_performance.csv",

        "02_Temperature_SHAP.csv":
            SENS_RUN
            / "03_temperature_Tmean_DTR"
            / "oof_shap"
            / "SHAP_contribution_bootstrap_CI_and_stability.csv",

        "02_Temperature_ALE.csv":
            SENS_RUN
            / "03_temperature_Tmean_DTR"
            / "ALE"
            / "ALE_summary_95_empirical_interval.csv",
    }

    for new_name, src in files.items():

        if src.exists():

            try:

                pd.read_csv(src).to_csv(
                    PLOT_DATA_OUT / new_name,
                    index=False,
                    encoding="utf-8-sig",
                )

            except Exception as exc:

                print(
                    f"[提示] 无法复制 {src.name}: {exc}"
                )


# =============================================================================
# 31. 创建完整目录结构
# =============================================================================
def create_output_structure() -> None:

    folders = [
        REDRAW_ROOT,
        MAIN_OUT,
        MAIN_MODEL_OUT,
        MAIN_OOF_OUT,
        MAIN_SHAP_OUT,
        MAIN_ALE_OUT,
        SENS_OUT,
        NOYEAR_OUT,
        NOYEAR_PERF_OUT,
        NOYEAR_SHAP_OUT,
        NOYEAR_ALE_OUT,
        BLOCK_OUT,
        BLOCK_PERF_OUT,
        TEMP_OUT,
        TEMP_PERF_OUT,
        TEMP_SHAP_OUT,
        TEMP_ALE_OUT,
        PLOT_DATA_OUT,
    ]

    for folder in folders:
        ensure_dir(folder)


# =============================================================================
# 32. 运行前检查
# =============================================================================
def preflight_check() -> None:

    print("=" * 90)
    print("03_Redraw_Final_Figures.py")
    print("01 + 02 全图件独立重绘版")
    print("不训练模型、不运行 Optuna、不重新计算 SHAP/ALE。")
    print("=" * 90)

    if not MAIN_RUN.exists():

        raise FileNotFoundError(
            f"主分析目录不存在：{MAIN_RUN}"
        )

    if not SENS_RUN.exists():

        raise FileNotFoundError(
            f"敏感性目录不存在：{SENS_RUN}"
        )

    print(
        f"01 主分析：{MAIN_RUN}"
    )

    print(
        f"02 敏感性：{SENS_RUN}"
    )

    print(
        f"最终模型：{load_selected_model()}"
    )

    print(
        f"主分析 ALE Top-6：{load_main_top6()}"
    )

    print(
        f"重绘根目录：{REDRAW_ROOT}"
    )

    print("=" * 90)


# =============================================================================
# 33. 主程序
# =============================================================================
def main() -> None:

    preflight_check()

    create_output_structure()

    export_plot_data_snapshot()

    # -------------------------------------------------------------------------
    # 01
    # -------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("开始重绘 01 主分析图件")
    print("=" * 90)

    redraw_all_main_figures()

    # -------------------------------------------------------------------------
    # 02-A
    # -------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("开始重绘 02-A No-Year 图件")
    print("=" * 90)

    redraw_noyear_figures()

    # -------------------------------------------------------------------------
    # 02-B
    # -------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("开始重绘 02-B 100 km Block CV 图件")
    print("=" * 90)

    redraw_block100_figures()

    # -------------------------------------------------------------------------
    # 02-C
    # -------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("开始重绘 02-C Tmean+DTR 图件")
    print("=" * 90)

    redraw_temperature_figures()

    # -------------------------------------------------------------------------
    # 02 总汇总
    # -------------------------------------------------------------------------
    redraw_all_sensitivity_performance_overview()

    print()
    print("=" * 90)
    print("全部重绘完成。")
    print(f"输出根目录：{REDRAW_ROOT}")
    print()
    print("目录结构：")
    print("Final_Publication_Figures/")
    print("├── 01_Main_Analysis/")
    print("│   ├── 01_Model_Performance/")
    print("│   ├── 02_OOF_Prediction/")
    print("│   ├── 03_SHAP/")
    print("│   └── 04_ALE/")
    print("└── 02_Sensitivity_Analyses/")
    print("    ├── 01_NoYear/")
    print("    ├── 02_100km_BlockCV/")
    print("    └── 03_Temperature_Tmean_DTR/")
    print()
    print("01 / 02 原始结果没有被覆盖。")
    print("=" * 90)


if __name__ == "__main__":
    main()
