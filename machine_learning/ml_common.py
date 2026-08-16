# -*- coding: utf-8 -*-
"""
川西北草地生态系统韧性机器学习分析 —— 公共函数模块

本文件不要单独运行。
由以下脚本自动调用：
1) 01_Main_ML_NestedCV_OOFSHAP_ALE.py
2) 02_Sensitivity_Analyses.py
3) 04_SHAP_Interaction_SR_SM.py

03_Redraw_Final_Figures.py 只读取 01/02 已完成结果并独立重绘图件，
不依赖本模块中的训练函数。

核心固定原则
------------
- 正式生态驱动变量：SM、PRE、TMN、TMX、SR、VPD、GI。
- 主模型额外加入 Year 作为时间协助变量。
- 主模型比较：OLS、精确 RBF-SVR、RF、LightGBM、XGBoost。
- 5 次重复 × 5 折 Nested Group CV；主分析按 patch_id1 分组。
- 外层重复种子：42、43、44、45、46。
- Optuna-TPE 直接最大化内层平均验证 R²，不使用 SD 惩罚项。
- XGBoost 内层 Early Stopping。
- 最终解释模型由 25 个外层任务平均 Test R² 自动决定，绝不预设 RF。
- Top-2 性能比较：repeated k-fold corrected t-test (r=5, k=5)，并输出 Wilcoxon 补充诊断。
- 正式解释：25 个外层模型 OOF-SHAP；Year 从 7 个生态驱动贡献率中剔除后重新归一化为 100%。
- SHAP 稳定性：25 模型 mean|SHAP|±SD、Kendall's W、patch_id1 cluster bootstrap 95% CI。
- 非线性主分析：ALE；PDP 不作为正式主分析。
- 图件：Times New Roman、600 dpi；变量名位于横轴中央，单位单独位于横轴最右端。

说明
----
用户已经完成 VIF/Pearson 前期变量筛选，因此本最终建模程序不重复进行变量筛选。
程序只保留必要的运行安全检查（字段、重复、Inf、分组泄漏等）。
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import math
import os
import platform
import re
import string
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import rankdata

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import shap
from joblib import Parallel, delayed


# =============================================================================
# 0. 固定配置
# =============================================================================
# GitHub 公共复现版目录约定：
# repo_root/
# ├── data/modeling_data.csv
# ├── machine_learning/*.py
# └── output/machine_learning/
#
# 如需在本机使用其他数据或输出位置，可设置环境变量：
#   NW_SICHUAN_DATA_FILE
#   NW_SICHUAN_OUTPUT_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "modeling_data.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "machine_learning"

DATA_FILE_PATH = Path(
    os.environ.get("NW_SICHUAN_DATA_FILE", str(DEFAULT_DATA_FILE))
).expanduser()

OUTPUT_ROOT_PATH = Path(
    os.environ.get("NW_SICHUAN_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))
).expanduser()

TARGET = "RI"
GROUP_COLUMN = "patch_id1"
BLOCK100_COLUMN = "Block100"
YEAR_COLUMN = "Year"
SAMPLE_ID_COLUMN = "sample_id"

DRIVER_FEATURES = ["SM", "PRE", "TMN", "TMX", "SR", "VPD", "GI"]
MAIN_MODEL_FEATURES = DRIVER_FEATURES + [YEAR_COLUMN]

RUN_MODELS = ["OLS", "SVR", "RF", "LightGBM", "XGBoost"]
TREE_MODELS = {"RF", "LightGBM", "XGBoost"}

REPEAT_SEEDS = [42, 43, 44, 45, 46]
N_REPEATS = 5
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 5

RANDOM_STATE = 42
CPU_COUNT = os.cpu_count() or 4
MODEL_N_JOBS = max(1, min(12, CPU_COUNT - 1 if CPU_COUNT > 1 else 1))

OPTUNA_TRIALS = {
    "OLS": 0,
    "SVR": 10,  # 精确 RBF-SVR；10 次 Optuna-TPE trial / 外层任务
    "RF": 20,
    "LightGBM": 25,
    "XGBoost": 35,
}

# -----------------------------------------------------------------------------
# 精确 RBF-SVR：正式主分析仍使用 sklearn.svm.SVR(kernel="rbf")，不使用 Nyström。
# 仅做两类计算优化：
# 1) 在仍具有较宽覆盖的前提下，收紧极端搜索区间，减少特别慢的高 C / 高 gamma 组合；
# 2) 5 个内层 Group-CV 折采用 2 进程并行。统计设计仍然是完整 5 折，不减少任何折。
# -----------------------------------------------------------------------------
SVR_C_RANGE = (0.01, 30.0)
SVR_EPSILON_RANGE = (0.001, 0.15)
SVR_GAMMA_RANGE = (1e-4, 0.50)

# 2 个并行 SVR 进程，每个最多使用约 2 GB libsvm kernel cache。
# 对 16 GB 级笔记本比 2 × 4096 MB 更稳；若机器内存很大可后续再上调。
SVR_CACHE_SIZE_MB = 2048
SVR_INNER_PARALLEL_JOBS = 2

# 改过 SVR 搜索空间后，必须与上午旧 Optuna study 隔离。
# 其他模型 study 名不变，因此其上午已经完成的 trial / task 可以继续复用。
SVR_STUDY_TAG = "EXACTRBF_V2_C30_G050_T10"

# XGBoost Early Stopping
XGB_MAX_ESTIMATORS = 5000
XGB_EARLY_STOPPING_ROUNDS = 150
XGB_MIN_FINAL_ESTIMATORS = 100
XGB_MAX_FINAL_ESTIMATORS = 5000

# SHAP
SHAP_CLUSTER_BOOTSTRAP_N = 1000
SHAP_BOOTSTRAP_SEED = 2026
SHAP_BACKGROUND_SAMPLES = 50
# 每个 OOF-SHAP 外层折在独立进程中运行。TreeSHAP 单折通常只占一个 CPU 核；
# 3 路并行适配本机 16 核 / 32 GB 内存，同时保留充足的系统余量。
SHAP_OUTER_PARALLEL_JOBS = 3
# None = 对最终模型的每个外层测试折完整解释。若极端情况下 SVR 成为最终模型，计算可能非常慢；
# 若机器资源确实不足，可显式设置一个数值，但论文必须如实说明下采样。
SHAP_NON_TREE_MAX_SAMPLES_PER_FOLD: Optional[int] = None

# ALE
ALE_TOP_N = 6
ALE_N_BINS = 20
ALE_COMMON_GRID_N = 60
ALE_Q_RANGE = (0.02, 0.98)
ALE_TURN_MIN_PROPORTION = 0.60

# 图件
FIG_DPI = 600

DISPLAY_NAME_MAP = {
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

# 单位严格按当前论文表 1 约定。图中显示为横轴最右端括号形式。
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


# =============================================================================
# 1. Matplotlib / 文件 / JSON 通用工具
# =============================================================================
def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


configure_matplotlib()


def display_name(feature: str) -> str:
    return DISPLAY_NAME_MAP.get(feature, str(feature))


def feature_unit(feature: str) -> str:
    return FEATURE_UNITS.get(feature, "")


def feature_label_with_unit(feature: str) -> str:
    unit = feature_unit(feature)
    return f"{display_name(feature)} ({unit})" if unit else display_name(feature)


def set_feature_xaxis_label_with_unit(
    ax,
    feature: str,
    fontsize: float = 12,
    unit_fontsize: Optional[float] = None,
    unit_y: float = -0.115,
) -> None:
    """变量名居中；单位单独放横轴最右端，格式如 (°C)。"""
    if unit_fontsize is None:
        unit_fontsize = fontsize
    ax.set_xlabel(display_name(feature), fontsize=fontsize, fontfamily="Times New Roman")
    unit = feature_unit(feature)
    if unit:
        ax.text(
            1.0,
            unit_y,
            f"({unit})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=unit_fontsize,
            fontfamily="Times New Roman",
            clip_on=False,
            zorder=500,
        )


def panel_label(index: int) -> str:
    letters = string.ascii_lowercase
    if index < 26:
        return f"({letters[index]})"
    return f"({letters[(index // 26) - 1]}{letters[index % 26]})"


def add_panel_label(ax, label: str = "(a)", x: float = 0.018, y: float = 0.982, fontsize: int = 14) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        fontfamily="Times New Roman",
        zorder=500,
        clip_on=False,
    )


def force_times_new_roman(ax) -> None:
    try:
        ax.title.set_fontfamily("Times New Roman")
        ax.xaxis.label.set_fontfamily("Times New Roman")
        ax.yaxis.label.set_fontfamily("Times New Roman")
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontfamily("Times New Roman")
        legend = ax.get_legend()
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontfamily("Times New Roman")
        for txt in ax.texts:
            txt.set_fontfamily("Times New Roman")
    except Exception:
        pass


def force_figure_times_new_roman(fig) -> None:
    for ax in fig.get_axes():
        force_times_new_roman(ax)


def get_cmap_safely(name: str):
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    return cm.get_cmap(name)


def sanitize_filename(name: Any) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def safe_json_dump(obj: Any, output_file: Path | str) -> None:
    output_file = Path(output_file)
    ensure_dir(output_file.parent)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)


def safe_json_load(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=json_default)


def save_figure_formats(
    fig,
    base_file: Path | str,
    dpi: int = FIG_DPI,
    fixed_canvas: bool = False,
    pad_inches: float = 0.12,
) -> None:
    base_file = Path(base_file).with_suffix("")
    ensure_dir(base_file.parent)
    kwargs = {"facecolor": "white"}
    if fixed_canvas:
        kwargs["bbox_inches"] = None
    else:
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"] = pad_inches

    fig.savefig(str(base_file) + ".png", dpi=dpi, **kwargs)
    try:
        fig.savefig(
            str(base_file) + ".tif",
            dpi=dpi,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
            **kwargs,
        )
    except Exception:
        fig.savefig(str(base_file) + ".tif", dpi=dpi, format="tiff", **kwargs)
    fig.savefig(str(base_file) + ".pdf", **kwargs)
    fig.savefig(str(base_file) + ".svg", **kwargs)
    plt.close(fig)
    gc.collect()


def read_csv_safely(path: Path | str) -> pd.DataFrame:
    last_error = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise last_error


def hash_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sqlite_url(path: Path | str) -> str:
    p = Path(path).resolve()
    return "sqlite:///" + str(p).replace("\\", "/")


def setup_logger(log_file: Path | str, name: str = "NW_Sichuan_ML") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def _format_duration(seconds: Optional[float]) -> str:
    """将耗时格式化为适合控制台进度条阅读的短文本。"""
    if seconds is None or not np.isfinite(seconds):
        return "--"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@dataclass
class ConsoleProgress:
    """轻量、无额外依赖的 PyCharm 控制台进度条。

    仅在任务完成、跳过或阶段开始时打印一行；不会向长耗时的 SHAP
    内部计算伪造百分比。断点续跑时已完成任务会作为初始进度显示。
    """
    label: str
    total: int
    completed_at_start: int = 0
    width: int = 24

    def __post_init__(self) -> None:
        self.completed_at_start = max(0, min(int(self.completed_at_start), int(self.total)))
        self.started_at = datetime.now()

    def show(self, completed: int, current: str = "") -> None:
        completed = max(0, min(int(completed), int(self.total)))
        ratio = completed / self.total if self.total else 1.0
        filled = int(round(self.width * ratio))
        # 使用 ASCII，避免 Windows 某些 GBK 控制台不能编码 Unicode 方块字符。
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = (datetime.now() - self.started_at).total_seconds()
        newly_finished = completed - self.completed_at_start
        eta = None
        if newly_finished > 0 and completed < self.total:
            eta = elapsed / newly_finished * (self.total - completed)
        text = (
            f"[{self.label}] {bar} {completed}/{self.total} "
            f"({ratio * 100:5.1f}%) | 已用 {_format_duration(elapsed)}"
        )
        if eta is not None:
            text += f" | 估计剩余 {_format_duration(eta)}"
        if current:
            text += f" | {current}"
        print(text, flush=True)


def export_software_versions(output_folder: Path | str) -> pd.DataFrame:
    packages = [
        "numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
        "xgboost", "lightgbm", "optuna", "shap", "openpyxl", "joblib"
    ]
    rows = [{"package": "python", "version": platform.python_version()}]
    for p in packages:
        try:
            version = importlib_metadata.version(p)
        except Exception:
            version = "NOT_INSTALLED"
        rows.append({"package": p, "version": version})
    out = pd.DataFrame(rows)
    out.to_csv(Path(output_folder) / "software_versions.csv", index=False, encoding="utf-8-sig")
    return out


def export_file_hashes(files: Dict[str, Path | str], output_folder: Path | str) -> pd.DataFrame:
    rows = []
    for label, path in files.items():
        p = Path(path)
        rows.append({
            "label": label,
            "file": str(p),
            "sha256": hash_file(p) if p.exists() and p.is_file() else "NOT_FOUND",
        })
    out = pd.DataFrame(rows)
    out.to_csv(Path(output_folder) / "file_hashes.csv", index=False, encoding="utf-8-sig")
    return out


def create_or_resume_run_directory(
    output_root: Path | str,
    prefix: str,
    active_pointer_name: str,
    complete_flag_name: str = "RUN_COMPLETE.flag",
) -> Tuple[Path, bool]:
    """若上次未完成则续跑同一目录；若已完成则创建新时间戳目录。"""
    root = ensure_dir(output_root)
    active_pointer = root / active_pointer_name
    if active_pointer.exists():
        try:
            candidate = Path(active_pointer.read_text(encoding="utf-8").strip())
            if candidate.exists() and not (candidate / complete_flag_name).exists():
                return candidate, True
        except Exception:
            pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(root / f"{prefix}_{stamp}")
    active_pointer.write_text(str(run_dir), encoding="utf-8")
    return run_dir, False


def mark_run_complete(
    run_dir: Path | str,
    output_root: Path | str,
    latest_pointer_name: str,
    active_pointer_name: str,
    complete_flag_name: str = "RUN_COMPLETE.flag",
) -> None:
    run_dir = Path(run_dir)
    root = Path(output_root)
    (run_dir / complete_flag_name).write_text(datetime.now().isoformat(), encoding="utf-8")
    (root / latest_pointer_name).write_text(str(run_dir), encoding="utf-8")
    active = root / active_pointer_name
    if active.exists():
        try:
            active.unlink()
        except Exception:
            pass


# =============================================================================
# 2. 输入表读取与必要运行安全检查（不是变量筛选模块）
# =============================================================================
def load_model_data(
    csv_path: Path | str,
    required_features: Sequence[str],
    require_block100: bool = True,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到输入表：{csv_path}")

    df = read_csv_safely(csv_path)
    df.columns = df.columns.astype(str).str.strip()
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = [TARGET, GROUP_COLUMN, YEAR_COLUMN] + list(required_features)
    if require_block100:
        required.append(BLOCK100_COLUMN)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入表缺少必要字段：{missing}\n当前字段：{df.columns.tolist()}")

    df = df.copy().reset_index(drop=True)
    df[SAMPLE_ID_COLUMN] = np.arange(len(df), dtype=int)
    df[GROUP_COLUMN] = df[GROUP_COLUMN].astype(str).str.strip()
    if BLOCK100_COLUMN in df.columns:
        df[BLOCK100_COLUMN] = df[BLOCK100_COLUMN].astype(str).str.strip()

    numeric_cols = list(dict.fromkeys([TARGET, YEAR_COLUMN] + list(required_features)))
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 目标、年份、分组字段不能缺失。
    if df[[TARGET, YEAR_COLUMN]].isna().any().any():
        bad = df[df[[TARGET, YEAR_COLUMN]].isna().any(axis=1)][[SAMPLE_ID_COLUMN, TARGET, YEAR_COLUMN]]
        raise ValueError(f"RI 或 Year 存在缺失，共 {len(bad)} 行。请先检查输入表。")
    if (df[GROUP_COLUMN].str.len() == 0).any():
        raise ValueError("patch_id1 存在空字符串。")
    if require_block100 and (df[BLOCK100_COLUMN].str.len() == 0).any():
        raise ValueError("Block100 存在空字符串。")

    # patch_id1-Year 必须唯一。
    dup = df.duplicated([GROUP_COLUMN, YEAR_COLUMN], keep=False)
    if dup.any():
        raise ValueError(
            f"发现 {int(dup.sum())} 行 patch_id1-Year 重复记录。最终建模前必须解决重复。"
        )

    # Inf 检查。生态驱动变量缺失允许存在，后续严格折内中位数填补。
    for c in numeric_cols:
        arr = df[c].to_numpy(dtype=float)
        if np.isinf(arr).any():
            raise ValueError(f"字段 {c} 存在 Inf/-Inf。")

    # 常数列检查。
    const_cols = [c for c in required_features if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        raise ValueError(f"以下正式建模变量为常数列：{const_cols}")

    # 每个 20 km 格网应只对应一个 Block100。
    if require_block100:
        n_block_per_patch = df.groupby(GROUP_COLUMN)[BLOCK100_COLUMN].nunique()
        bad_patch = n_block_per_patch[n_block_per_patch != 1]
        if len(bad_patch):
            raise ValueError(f"有 {len(bad_patch)} 个 patch_id1 对应多个 Block100。")

    return df


def input_structure_summary(df: pd.DataFrame, model_features: Sequence[str]) -> Dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "n_patch_id1": int(df[GROUP_COLUMN].nunique()),
        "n_block100": int(df[BLOCK100_COLUMN].nunique()) if BLOCK100_COLUMN in df.columns else None,
        "year_min": int(df[YEAR_COLUMN].min()),
        "year_max": int(df[YEAR_COLUMN].max()),
        "n_years": int(df[YEAR_COLUMN].nunique()),
        "patch_year_duplicates": int(df.duplicated([GROUP_COLUMN, YEAR_COLUMN]).sum()),
        "missing_counts": {c: int(df[c].isna().sum()) for c in [TARGET] + list(model_features)},
    }


# =============================================================================
# 3. 重复分组划分与 split manifest
# =============================================================================
def build_repeated_group_splits(
    df: pd.DataFrame,
    group_column: str,
    seeds: Sequence[int] = REPEAT_SEEDS,
    n_splits: int = OUTER_N_SPLITS,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    group_series = df[group_column].astype(str)
    unique_groups = np.array(sorted(group_series.unique()))
    if len(unique_groups) < n_splits:
        raise ValueError(f"唯一分组数 {len(unique_groups)} < n_splits={n_splits}")

    tasks: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []
    group_manifest_rows: List[Dict[str, Any]] = []

    for repeat_id, seed in enumerate(seeds, start=1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        group_to_test_fold: Dict[str, int] = {}

        for fold_id, (g_train_idx, g_test_idx) in enumerate(kf.split(unique_groups), start=1):
            train_groups = set(unique_groups[g_train_idx])
            test_groups = set(unique_groups[g_test_idx])
            if not train_groups.isdisjoint(test_groups):
                raise RuntimeError("外层 group split 出现交集。")

            train_mask = group_series.isin(train_groups).to_numpy()
            test_mask = group_series.isin(test_groups).to_numpy()
            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)

            tasks.append({
                "repeat": repeat_id,
                "fold": fold_id,
                "seed": int(seed),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_group_count": len(train_groups),
                "test_group_count": len(test_groups),
                "train_sample_count": len(train_idx),
                "test_sample_count": len(test_idx),
                "group_column": group_column,
            })

            for g in sorted(train_groups):
                group_manifest_rows.append({
                    "repeat": repeat_id, "fold": fold_id, "seed": int(seed),
                    "group_column": group_column, "group_id": g, "role": "train"
                })
            for g in sorted(test_groups):
                group_manifest_rows.append({
                    "repeat": repeat_id, "fold": fold_id, "seed": int(seed),
                    "group_column": group_column, "group_id": g, "role": "test"
                })
                group_to_test_fold[g] = fold_id

        for row_idx, g in zip(df[SAMPLE_ID_COLUMN].to_numpy(), group_series.to_numpy()):
            assignment_rows.append({
                SAMPLE_ID_COLUMN: int(row_idx),
                "repeat": repeat_id,
                "seed": int(seed),
                "test_fold": int(group_to_test_fold[g]),
                "group_column": group_column,
                "group_id": g,
            })

    return tasks, pd.DataFrame(assignment_rows), pd.DataFrame(group_manifest_rows)


def load_outer_tasks_from_assignment(
    df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    group_column: str,
) -> List[Dict[str, Any]]:
    tasks = []
    sid_to_pos = pd.Series(np.arange(len(df), dtype=int), index=df[SAMPLE_ID_COLUMN].astype(int)).to_dict()
    for repeat_id in sorted(assignment_df["repeat"].unique()):
        rep = assignment_df[assignment_df["repeat"] == repeat_id]
        seed = int(rep["seed"].iloc[0])
        for fold_id in sorted(rep["test_fold"].unique()):
            test_sids = rep.loc[rep["test_fold"] == fold_id, SAMPLE_ID_COLUMN].astype(int).to_numpy()
            test_idx = np.array([sid_to_pos[int(sid)] for sid in test_sids], dtype=int)
            test_mask = np.zeros(len(df), dtype=bool)
            test_mask[test_idx] = True
            train_idx = np.flatnonzero(~test_mask)
            train_groups = set(df.iloc[train_idx][group_column].astype(str))
            test_groups = set(df.iloc[test_idx][group_column].astype(str))
            if not train_groups.isdisjoint(test_groups):
                raise RuntimeError(f"载入 manifest 后 R{repeat_id}F{fold_id} 出现 group 泄漏。")
            tasks.append({
                "repeat": int(repeat_id),
                "fold": int(fold_id),
                "seed": seed,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_group_count": len(train_groups),
                "test_group_count": len(test_groups),
                "train_sample_count": len(train_idx),
                "test_sample_count": len(test_idx),
                "group_column": group_column,
            })
    return tasks


def random_group_kfold_indices(groups: Sequence[Any], n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups_s = pd.Series(groups).astype(str).reset_index(drop=True)
    unique_groups = np.array(sorted(groups_s.unique()))
    if len(unique_groups) < n_splits:
        raise ValueError(f"内层唯一分组数 {len(unique_groups)} < n_splits={n_splits}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    splits = []
    for g_tr, g_va in kf.split(unique_groups):
        tr_groups = set(unique_groups[g_tr])
        va_groups = set(unique_groups[g_va])
        tr_idx = np.flatnonzero(groups_s.isin(tr_groups).to_numpy())
        va_idx = np.flatnonzero(groups_s.isin(va_groups).to_numpy())
        if not tr_groups.isdisjoint(va_groups):
            raise RuntimeError("内层 group split 出现交集。")
        splits.append((tr_idx, va_idx))
    return splits


def validate_outer_tasks(tasks: Sequence[Dict[str, Any]], df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for task in tasks:
        tr = task["train_idx"]
        te = task["test_idx"]
        tr_groups = set(df.iloc[tr][group_column].astype(str))
        te_groups = set(df.iloc[te][group_column].astype(str))
        overlap = tr_groups.intersection(te_groups)
        rows.append({
            "repeat": task["repeat"],
            "fold": task["fold"],
            "seed": task["seed"],
            "group_column": group_column,
            "train_groups": len(tr_groups),
            "test_groups": len(te_groups),
            "train_samples": len(tr),
            "test_samples": len(te),
            "overlap_groups": len(overlap),
            "test_train_sample_ratio": len(te) / len(tr),
        })
        if overlap:
            raise RuntimeError(f"R{task['repeat']}F{task['fold']} 存在 {len(overlap)} 个 group 泄漏。")
    return pd.DataFrame(rows)


# =============================================================================
# 4. 模型、搜索空间和 Optuna-TPE
# =============================================================================
def search_space_table() -> pd.DataFrame:
    rows = [
        ["OLS", "No tuning", "—", "—", "baseline", 0],
        ["SVR", "C", SVR_C_RANGE[0], SVR_C_RANGE[1], "float(log)", OPTUNA_TRIALS["SVR"]],
        ["SVR", "epsilon", SVR_EPSILON_RANGE[0], SVR_EPSILON_RANGE[1], "float(log)", OPTUNA_TRIALS["SVR"]],
        ["SVR", "gamma", SVR_GAMMA_RANGE[0], SVR_GAMMA_RANGE[1], "float(log)", OPTUNA_TRIALS["SVR"]],
        ["RF", "n_estimators", 400, 1400, "int(step=100)", OPTUNA_TRIALS["RF"]],
        ["RF", "max_depth", 5, 24, "int", OPTUNA_TRIALS["RF"]],
        ["RF", "min_samples_split", 2, 24, "int", OPTUNA_TRIALS["RF"]],
        ["RF", "min_samples_leaf", 2, 16, "int", OPTUNA_TRIALS["RF"]],
        ["RF", "max_features", "sqrt/log2/0.6/0.8/1.0", "—", "categorical", OPTUNA_TRIALS["RF"]],
        ["LightGBM", "n_estimators", 500, 2400, "int(step=100)", OPTUNA_TRIALS["LightGBM"]],
        ["LightGBM", "learning_rate", 0.005, 0.06, "float(log)", OPTUNA_TRIALS["LightGBM"]],
        ["LightGBM", "max_depth", 4, 11, "int", OPTUNA_TRIALS["LightGBM"]],
        ["LightGBM", "num_leaves", 15, 127, "int", OPTUNA_TRIALS["LightGBM"]],
        ["LightGBM", "min_child_samples", 10, 120, "int", OPTUNA_TRIALS["LightGBM"]],
        ["LightGBM", "subsample", 0.60, 0.95, "float", OPTUNA_TRIALS["LightGBM"]],
        ["LightGBM", "colsample_bytree", 0.55, 0.95, "float", OPTUNA_TRIALS["LightGBM"]],
        ["LightGBM", "reg_lambda", 0.1, 50.0, "float(log)", OPTUNA_TRIALS["LightGBM"]],
        ["LightGBM", "reg_alpha", 1e-6, 10.0, "float(log)", OPTUNA_TRIALS["LightGBM"]],
        ["XGBoost", "learning_rate", 0.005, 0.06, "float(log)", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "max_depth", 3, 9, "int", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "min_child_weight", 5, 50, "int", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "subsample", 0.60, 0.95, "float", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "colsample_bytree", 0.55, 0.95, "float", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "colsample_bynode", 0.55, 1.00, "float", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "gamma", 1e-5, 2.0, "float(log)", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "reg_lambda", 0.5, 100.0, "float(log)", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "reg_alpha", 1e-6, 10.0, "float(log)", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "max_bin", 128, 512, "int(step=64)", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "grow_policy", "depthwise/lossguide", "—", "categorical", OPTUNA_TRIALS["XGBoost"]],
        ["XGBoost", "max_leaves", 16, 256, "int(log), only lossguide", OPTUNA_TRIALS["XGBoost"]],
    ]
    return pd.DataFrame(rows, columns=["Model", "Hyperparameter", "Lower_or_Options", "Upper", "Type", "Optuna_trials_per_outer_task"])


def export_search_space(output_folder: Path | str) -> pd.DataFrame:
    df = search_space_table()
    out = Path(output_folder)
    ensure_dir(out)
    df.to_csv(out / "Hyperparameter_Search_Space.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "Hyperparameter_Search_Space.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="SearchSpace", index=False)
    return df


def sanitize_params(model_name: str, params: Dict[str, Any] | None) -> Dict[str, Any]:
    p = dict(params or {})
    if model_name == "SVR":
        for k in ["C", "epsilon", "gamma"]:
            if k in p:
                p[k] = float(p[k])
    elif model_name == "RF":
        for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
            if k in p:
                p[k] = int(round(float(p[k])))
    elif model_name == "LightGBM":
        for k in ["n_estimators", "max_depth", "num_leaves", "min_child_samples"]:
            if k in p:
                p[k] = int(round(float(p[k])))
        for k in ["learning_rate", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"]:
            if k in p:
                p[k] = float(p[k])
    elif model_name == "XGBoost":
        for k in ["max_depth", "min_child_weight", "max_leaves", "max_bin"]:
            if k in p:
                p[k] = int(round(float(p[k])))
        for k in ["learning_rate", "subsample", "colsample_bytree", "colsample_bynode", "gamma", "reg_lambda", "reg_alpha"]:
            if k in p:
                p[k] = float(p[k])
    return p


def suggest_params(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    if model_name == "SVR":
        return {
            "C": trial.suggest_float("C", SVR_C_RANGE[0], SVR_C_RANGE[1], log=True),
            "epsilon": trial.suggest_float("epsilon", SVR_EPSILON_RANGE[0], SVR_EPSILON_RANGE[1], log=True),
            "gamma": trial.suggest_float("gamma", SVR_GAMMA_RANGE[0], SVR_GAMMA_RANGE[1], log=True),
        }
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1400, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 16),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.6, 0.8, 1.0]),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2400, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.06, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 11),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.60, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        }
    if model_name == "XGBoost":
        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        max_leaves = 0
        if grow_policy == "lossguide":
            max_leaves = trial.suggest_int("max_leaves", 16, 256, log=True)
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.06, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.60, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.55, 1.00),
            "gamma": trial.suggest_float("gamma", 1e-5, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
            "grow_policy": grow_policy,
            "max_leaves": max_leaves,
        }
    if model_name == "OLS":
        return {}
    raise ValueError(f"未知模型：{model_name}")


def build_pipeline_model(
    model_name: str,
    params: Dict[str, Any] | None,
    random_state: int,
    model_n_jobs: Optional[int] = None,
) -> Pipeline:
    params = sanitize_params(model_name, params)
    runtime_n_jobs = MODEL_N_JOBS if model_n_jobs is None else max(1, int(model_n_jobs))
    if model_name == "OLS":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
    if model_name == "SVR":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=params["C"],
                epsilon=params["epsilon"],
                gamma=params["gamma"],
                cache_size=SVR_CACHE_SIZE_MB,
                shrinking=True,
            )),
        ])
    if model_name == "RF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=random_state,
                n_jobs=runtime_n_jobs,
            )),
        ])
    if model_name == "LightGBM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                objective="regression",
                random_state=random_state,
                n_jobs=runtime_n_jobs,
                verbosity=-1,
                bagging_freq=1,
                **params,
            )),
        ])
    raise ValueError(f"build_pipeline_model 不支持 {model_name}")


def build_xgb_model(
    params: Dict[str, Any],
    n_estimators: int,
    random_state: int,
    early_stopping_rounds: Optional[int] = None,
    model_n_jobs: Optional[int] = None,
) -> XGBRegressor:
    params = sanitize_params("XGBoost", params)
    kwargs = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=MODEL_N_JOBS if model_n_jobs is None else max(1, int(model_n_jobs)),
        tree_method="hist",
        verbosity=0,
        **params,
    )
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return XGBRegressor(**kwargs)


@dataclass
class XGBModelBundle:
    imputer: SimpleImputer
    model: XGBRegressor
    feature_names: List[str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = pd.DataFrame(X, columns=self.feature_names).copy()
        arr = self.imputer.transform(Xf[self.feature_names])
        return pd.DataFrame(arr, columns=self.feature_names, index=Xf.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.transform(X))


def fit_xgb_with_early_stopping(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int,
) -> Tuple[SimpleImputer, XGBRegressor, int]:
    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train)
    Xva = imputer.transform(X_valid)
    model = build_xgb_model(params, XGB_MAX_ESTIMATORS, random_state, XGB_EARLY_STOPPING_ROUNDS)
    model.fit(Xtr, y_train, eval_set=[(Xva, y_valid)], verbose=False)
    best_iteration = getattr(model, "best_iteration", None)
    n_est = XGB_MAX_ESTIMATORS if best_iteration is None else int(best_iteration) + 1
    n_est = int(np.clip(n_est, XGB_MIN_FINAL_ESTIMATORS, XGB_MAX_FINAL_ESTIMATORS))
    return imputer, model, n_est


def evaluate_model(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": mse,
        "MSE_x1e3": mse * 1000.0,
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _evaluate_one_generic_fold(
    model_name: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    tr: np.ndarray,
    va: np.ndarray,
    random_state: int,
) -> Tuple[float, float, float, float]:
    """
    评价一个内层折。

    说明：
    - 模型本身、样本划分、缺失值处理与评价指标均不改变；
    - 单独拆成顶层函数，是为了让 Windows + joblib(loky) 能稳定并行 SVR 内层折。
    """
    model = build_pipeline_model(model_name, params, random_state)
    X_tr = X.iloc[tr]
    X_va = X.iloc[va]
    y_tr = y.iloc[tr]
    y_va = y.iloc[va]

    model.fit(X_tr, y_tr)
    p_tr = model.predict(X_tr)
    p_va = model.predict(X_va)

    result = (
        float(r2_score(y_tr, p_tr)),
        float(r2_score(y_va, p_va)),
        float(mean_squared_error(y_tr, p_tr)),
        float(mean_squared_error(y_va, p_va)),
    )

    del model
    gc.collect()
    return result


def evaluate_generic_params(
    model_name: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    base_seed: int,
) -> Dict[str, Any]:
    """
    OLS / SVR / RF / LightGBM 的内层交叉验证评价。

    精确 RBF-SVR 仍完整计算 5 个 inner folds，只把原来的串行执行
    改为最多 2 个折同时运行；折数和样本不变，SVR 调参预算为 10 trials / 外层任务。
    """
    if model_name == "SVR":
        fold_results = Parallel(
            n_jobs=SVR_INNER_PARALLEL_JOBS,
            backend="loky",
            batch_size=1,
            pre_dispatch=SVR_INNER_PARALLEL_JOBS,
        )(
            delayed(_evaluate_one_generic_fold)(
                model_name=model_name,
                params=params,
                X=X,
                y=y,
                tr=tr,
                va=va,
                random_state=base_seed + i,
            )
            for i, (tr, va) in enumerate(inner_splits, start=1)
        )
    else:
        # 其它模型维持原执行方式，避免与其自身 n_jobs 形成嵌套并行。
        fold_results = [
            _evaluate_one_generic_fold(
                model_name=model_name,
                params=params,
                X=X,
                y=y,
                tr=tr,
                va=va,
                random_state=base_seed + i,
            )
            for i, (tr, va) in enumerate(inner_splits, start=1)
        ]

    tr_r2 = [x[0] for x in fold_results]
    va_r2 = [x[1] for x in fold_results]
    tr_mse = [x[2] for x in fold_results]
    va_mse = [x[3] for x in fold_results]

    return {
        "mean_train_r2": float(np.mean(tr_r2)),
        "mean_validation_r2": float(np.mean(va_r2)),
        "std_validation_r2": float(np.std(va_r2, ddof=1)) if len(va_r2) > 1 else 0.0,
        "mean_train_mse": float(np.mean(tr_mse)),
        "mean_validation_mse": float(np.mean(va_mse)),
        "std_validation_mse": float(np.std(va_mse, ddof=1)) if len(va_mse) > 1 else 0.0,
        "mean_gap": float(np.mean(tr_r2) - np.mean(va_r2)),
        # 最终老师意见：直接最大化内层平均 R²，不再使用 λ×SD 惩罚。
        "objective_score": float(np.mean(va_r2)),
    }


def evaluate_xgb_params(
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    base_seed: int,
) -> Dict[str, Any]:
    tr_r2, va_r2, tr_mse, va_mse, n_estimators = [], [], [], [], []
    for i, (tr, va) in enumerate(inner_splits, start=1):
        imp, model, n_est = fit_xgb_with_early_stopping(
            params, X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], base_seed + i
        )
        p_tr = model.predict(imp.transform(X.iloc[tr]))
        p_va = model.predict(imp.transform(X.iloc[va]))
        tr_r2.append(r2_score(y.iloc[tr], p_tr))
        va_r2.append(r2_score(y.iloc[va], p_va))
        tr_mse.append(mean_squared_error(y.iloc[tr], p_tr))
        va_mse.append(mean_squared_error(y.iloc[va], p_va))
        n_estimators.append(int(n_est))
        del model, imp
        gc.collect()
    return {
        "mean_train_r2": float(np.mean(tr_r2)),
        "mean_validation_r2": float(np.mean(va_r2)),
        "std_validation_r2": float(np.std(va_r2, ddof=1)) if len(va_r2) > 1 else 0.0,
        "mean_train_mse": float(np.mean(tr_mse)),
        "mean_validation_mse": float(np.mean(va_mse)),
        "std_validation_mse": float(np.std(va_mse, ddof=1)) if len(va_mse) > 1 else 0.0,
        "mean_gap": float(np.mean(tr_r2) - np.mean(va_r2)),
        "objective_score": float(np.mean(va_r2)),
        "fold_best_estimators": n_estimators,
        "median_best_estimators": int(np.median(n_estimators)),
        "mean_best_estimators": float(np.mean(n_estimators)),
    }


def tune_model_nested(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    repeat_id: int,
    fold_id: int,
    outer_seed: int,
    tuning_folder: Path,
    sqlite_db: Path,
    study_prefix: str,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ensure_dir(tuning_folder)
    inner_seed = int(outer_seed * 1000 + fold_id * 37 + 17)
    inner_splits = random_group_kfold_indices(groups_train, INNER_N_SPLITS, inner_seed)

    if model_name == "OLS":
        result = evaluate_generic_params("OLS", {}, X_train, y_train, groups_train, inner_splits, inner_seed)
        safe_json_dump({"best_params": {}, "inner_cv": result, "tuning": "No tuning; linear baseline"}, tuning_folder / "best_result.json")
        return {}, result

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if model_name == "SVR":
        # 新的精确 RBF-SVR 搜索空间使用独立 study 名，避免与上午旧范围 trial 混合。
        study_name = f"{study_prefix}_{model_name}_{SVR_STUDY_TAG}_R{repeat_id:02d}_F{fold_id:02d}"
        logger.info(
            f"SVR R{repeat_id}F{fold_id}: exact RBF；{OPTUNA_TRIALS['SVR']} trials；"
            f"C={SVR_C_RANGE[0]}–{SVR_C_RANGE[1]}，"
            f"epsilon={SVR_EPSILON_RANGE[0]}–{SVR_EPSILON_RANGE[1]}，"
            f"gamma={SVR_GAMMA_RANGE[0]}–{SVR_GAMMA_RANGE[1]}；"
            f"inner {INNER_N_SPLITS}-fold，{SVR_INNER_PARALLEL_JOBS} 路并行。"
        )
    else:
        study_name = f"{study_prefix}_{model_name}_R{repeat_id:02d}_F{fold_id:02d}"
    storage = sqlite_url(sqlite_db)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=inner_seed),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(model_name, trial)
        if model_name == "XGBoost":
            res = evaluate_xgb_params(params, X_train, y_train, groups_train, inner_splits, inner_seed)
            trial.set_user_attr("fold_best_estimators", res["fold_best_estimators"])
            trial.set_user_attr("median_best_estimators", res["median_best_estimators"])
        else:
            res = evaluate_generic_params(model_name, params, X_train, y_train, groups_train, inner_splits, inner_seed)
        for k in ["mean_train_r2", "mean_validation_r2", "std_validation_r2", "mean_train_mse", "mean_validation_mse", "std_validation_mse", "mean_gap"]:
            trial.set_user_attr(k, res[k])
        return res["objective_score"]

    target_trials = OPTUNA_TRIALS[model_name]
    complete_trials = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, target_trials - complete_trials)
    if remaining:
        logger.info(f"{model_name} R{repeat_id}F{fold_id}: Optuna 继续/开始 {remaining} 个 trial（目标总数 {target_trials}）。")

        callbacks = []
        if model_name == "SVR":
            def _svr_trial_progress(study_obj: optuna.Study, trial_obj: optuna.trial.FrozenTrial) -> None:
                n_complete_now = sum(
                    t.state == optuna.trial.TrialState.COMPLETE for t in study_obj.trials
                )
                duration_s = trial_obj.duration.total_seconds() if trial_obj.duration is not None else float("nan")
                if trial_obj.state == optuna.trial.TrialState.COMPLETE:
                    logger.info(
                        f"SVR R{repeat_id}F{fold_id}: trial {n_complete_now}/{target_trials} 完成；"
                        f"本 trial {duration_s/60.0:.1f} min；"
                        f"当前 best inner R²={study_obj.best_value:.4f}"
                    )
                else:
                    logger.warning(
                        f"SVR R{repeat_id}F{fold_id}: 一个 trial 状态={trial_obj.state.name}；"
                        f"已完成 {n_complete_now}/{target_trials}。"
                    )
            callbacks.append(_svr_trial_progress)

        # Optuna trial 本身保持串行，避免出现“trial 并行 × inner-fold 并行”的内存爆炸。
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
            callbacks=callbacks,
        )
    else:
        logger.info(f"{model_name} R{repeat_id}F{fold_id}: Optuna 已完成 {complete_trials}/{target_trials}，直接读取。")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(tuning_folder / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    best_params = sanitize_params(model_name, study.best_params)
    if model_name == "XGBoost":
        final_cv = evaluate_xgb_params(best_params, X_train, y_train, groups_train, inner_splits, inner_seed)
    else:
        final_cv = evaluate_generic_params(model_name, best_params, X_train, y_train, groups_train, inner_splits, inner_seed)
    safe_json_dump({
        "study_name": study_name,
        "best_value": float(study.best_value),
        "best_params": best_params,
        "inner_cv": final_cv,
        "target_trials": target_trials,
        "complete_trials": int(sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)),
    }, tuning_folder / "best_result.json")
    return best_params, final_cv


def fit_outer_model(
    model_name: str,
    params: Dict[str, Any],
    final_n_estimators: Optional[int],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    feature_names: Sequence[str],
    model_n_jobs: Optional[int] = None,
):
    if model_name == "XGBoost":
        if final_n_estimators is None or (isinstance(final_n_estimators, float) and np.isnan(final_n_estimators)):
            raise ValueError("XGBoost 外层最终模型缺少 final_n_estimators。")
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_train)
        model = build_xgb_model(
            params, int(final_n_estimators), random_state,
            early_stopping_rounds=None, model_n_jobs=model_n_jobs,
        )
        model.fit(Xtr, y_train, verbose=False)
        return XGBModelBundle(imp, model, list(feature_names))
    model = build_pipeline_model(model_name, params, random_state, model_n_jobs=model_n_jobs)
    model.fit(X_train, y_train)
    return model


# =============================================================================
# 5. Nested CV 主循环、断点续跑、模型汇总
# =============================================================================
def run_repeated_nested_cv(
    df: pd.DataFrame,
    features: Sequence[str],
    group_column: str,
    outer_tasks: Sequence[Dict[str, Any]],
    model_names: Sequence[str],
    output_folder: Path | str,
    checkpoint_folder: Path | str,
    sqlite_db: Path | str,
    study_prefix: str,
    logger: logging.Logger,
    progress_label: str = "Nested CV",
) -> pd.DataFrame:
    output_folder = ensure_dir(output_folder)
    checkpoint_folder = ensure_dir(checkpoint_folder)
    sqlite_db = Path(sqlite_db)
    X_all = df[list(features)].copy()
    y_all = df[TARGET].copy()
    groups_all = df[group_column].astype(str).copy()

    all_results: List[Dict[str, Any]] = []
    total = len(outer_tasks) * len(model_names)
    counter = 0
    precompleted = 0
    for task in outer_tasks:
        r, f = int(task["repeat"]), int(task["fold"])
        for model_name in model_names:
            task_dir = output_folder / "tasks" / model_name / f"repeat_{r:02d}" / f"fold_{f:02d}"
            if (task_dir / "TASK_COMPLETE.flag").exists() and (task_dir / "task_result.json").exists() and (task_dir / "test_predictions.csv").exists():
                precompleted += 1
    progress = ConsoleProgress(progress_label, total, completed_at_start=precompleted)
    progress.show(precompleted, "resuming checkpoint scan")
    completed_count = precompleted

    for task in outer_tasks:
        r, f, seed = int(task["repeat"]), int(task["fold"]), int(task["seed"])
        tr_idx, te_idx = task["train_idx"], task["test_idx"]
        X_train, X_test = X_all.iloc[tr_idx].copy(), X_all.iloc[te_idx].copy()
        y_train, y_test = y_all.iloc[tr_idx].copy(), y_all.iloc[te_idx].copy()
        g_train, g_test = groups_all.iloc[tr_idx].copy(), groups_all.iloc[te_idx].copy()
        if not set(g_train).isdisjoint(set(g_test)):
            raise RuntimeError(f"R{r}F{f} group 泄漏。")

        for model_name in model_names:
            counter += 1
            task_dir = ensure_dir(output_folder / "tasks" / model_name / f"repeat_{r:02d}" / f"fold_{f:02d}")
            tuning_dir = ensure_dir(task_dir / "tuning")
            done_flag = task_dir / "TASK_COMPLETE.flag"
            result_json = task_dir / "task_result.json"
            test_pred_file = task_dir / "test_predictions.csv"
            train_pred_file = task_dir / "train_predictions.csv"

            if done_flag.exists() and result_json.exists() and test_pred_file.exists():
                result = safe_json_load(result_json)
                all_results.append(result)
                logger.info(f"[{counter}/{total}] 跳过已完成：{model_name} R{r}F{f}")
                continue

            logger.info(f"[{counter}/{total}] 开始：{model_name} R{r}F{f}")
            best_params, inner_cv = tune_model_nested(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                groups_train=g_train.reset_index(drop=True),
                repeat_id=r,
                fold_id=f,
                outer_seed=seed,
                tuning_folder=tuning_dir,
                sqlite_db=sqlite_db,
                study_prefix=study_prefix,
                logger=logger,
            )

            final_n_estimators = None
            if model_name == "XGBoost":
                final_n_estimators = int(np.clip(
                    inner_cv["median_best_estimators"], XGB_MIN_FINAL_ESTIMATORS, XGB_MAX_FINAL_ESTIMATORS
                ))

            model = fit_outer_model(
                model_name, best_params, final_n_estimators,
                X_train, y_train, random_state=seed + f + r * 100, feature_names=features
            )
            p_train = model.predict(X_train)
            p_test = model.predict(X_test)
            train_m = evaluate_model(y_train, p_train)
            test_m = evaluate_model(y_test, p_test)

            train_out = pd.DataFrame({
                SAMPLE_ID_COLUMN: df.iloc[tr_idx][SAMPLE_ID_COLUMN].to_numpy(),
                GROUP_COLUMN: df.iloc[tr_idx][GROUP_COLUMN].astype(str).to_numpy(),
                YEAR_COLUMN: df.iloc[tr_idx][YEAR_COLUMN].to_numpy(),
                "observed_RI": y_train.to_numpy(),
                "predicted_RI": p_train,
                "residual": y_train.to_numpy() - p_train,
                "repeat": r,
                "fold": f,
                "Model": model_name,
                "dataset": "train",
                "CV_group_column": group_column,
                "CV_group_id": df.iloc[tr_idx][group_column].astype(str).to_numpy(),
            })
            test_out = pd.DataFrame({
                SAMPLE_ID_COLUMN: df.iloc[te_idx][SAMPLE_ID_COLUMN].to_numpy(),
                GROUP_COLUMN: df.iloc[te_idx][GROUP_COLUMN].astype(str).to_numpy(),
                YEAR_COLUMN: df.iloc[te_idx][YEAR_COLUMN].to_numpy(),
                "observed_RI": y_test.to_numpy(),
                "predicted_RI": p_test,
                "residual": y_test.to_numpy() - p_test,
                "repeat": r,
                "fold": f,
                "Model": model_name,
                "dataset": "test",
                "CV_group_column": group_column,
                "CV_group_id": df.iloc[te_idx][group_column].astype(str).to_numpy(),
            })
            train_out.to_csv(train_pred_file, index=False, encoding="utf-8-sig")
            test_out.to_csv(test_pred_file, index=False, encoding="utf-8-sig")

            result = {
                "Model": model_name,
                "Repeat": r,
                "Outer_Fold": f,
                "Outer_Seed": seed,
                "Group_Column": group_column,
                "Train_Group_Count": int(g_train.nunique()),
                "Test_Group_Count": int(g_test.nunique()),
                "Train_Sample_Count": int(len(tr_idx)),
                "Test_Sample_Count": int(len(te_idx)),
                "Test_Train_Sample_Ratio": float(len(te_idx) / len(tr_idx)),
                "Train_R2": train_m["R2"],
                "Train_MSE": train_m["MSE"],
                "Train_MSE_x1e3": train_m["MSE_x1e3"],
                "Train_RMSE": train_m["RMSE"],
                "Train_MAE": train_m["MAE"],
                "Inner_CV_R2_mean": inner_cv["mean_validation_r2"],
                "Inner_CV_R2_std": inner_cv["std_validation_r2"],
                "Inner_CV_MSE_mean": inner_cv["mean_validation_mse"],
                "Inner_CV_MSE_std": inner_cv["std_validation_mse"],
                "Inner_CV_Train_R2_mean": inner_cv["mean_train_r2"],
                "Inner_CV_Gap": inner_cv["mean_gap"],
                "Test_R2": test_m["R2"],
                "Test_MSE": test_m["MSE"],
                "Test_MSE_x1e3": test_m["MSE_x1e3"],
                "Test_RMSE": test_m["RMSE"],
                "Test_MAE": test_m["MAE"],
                "Train_Test_R2_Gap": train_m["R2"] - test_m["R2"],
                "XGB_Final_n_estimators": final_n_estimators,
                "Best_Params_JSON": safe_json_dumps(best_params),
            }
            safe_json_dump(result, result_json)
            done_flag.write_text(datetime.now().isoformat(), encoding="utf-8")
            all_results.append(result)
            completed_count += 1
            progress.show(completed_count, f"completed {model_name} R{r}F{f}")
            del model
            gc.collect()

            pd.DataFrame(all_results).to_csv(checkpoint_folder / "all_outer_scores_checkpoint.csv", index=False, encoding="utf-8-sig")

    results = pd.DataFrame(all_results).sort_values(["Model", "Repeat", "Outer_Fold"]).reset_index(drop=True)
    results.to_csv(output_folder / "all_outer_scores.csv", index=False, encoding="utf-8-sig")
    return results


def summarize_model_performance(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in results.groupby("Model"):
        row = {"Model": model, "Outer_Task_Count": len(g)}
        for col in ["Test_R2", "Test_MSE", "Test_MSE_x1e3", "Test_RMSE", "Test_MAE", "Train_R2", "Train_Test_R2_Gap", "Inner_CV_R2_mean"]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_SD"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        row["Test_R2_min"] = float(g["Test_R2"].min())
        row["Test_R2_max"] = float(g["Test_R2"].max())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["Test_R2_mean", "Test_RMSE_mean", "Test_MAE_mean", "Train_Test_R2_Gap_mean"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)
    summary.insert(0, "Model_Rank", np.arange(1, len(summary) + 1))
    summary["Selected_for_Interpretation"] = np.where(summary["Model_Rank"] == 1, "Yes", "No")
    summary["Test_R2_mean_SD"] = summary.apply(lambda r: f"{r['Test_R2_mean']:.3f} ± {r['Test_R2_SD']:.3f}", axis=1)
    summary["Test_MSE_x1e3_mean_SD"] = summary.apply(lambda r: f"{r['Test_MSE_x1e3_mean']:.3f} ± {r['Test_MSE_x1e3_SD']:.3f}", axis=1)
    summary["Test_RMSE_mean_SD"] = summary.apply(lambda r: f"{r['Test_RMSE_mean']:.3f} ± {r['Test_RMSE_SD']:.3f}", axis=1)
    summary["Test_MAE_mean_SD"] = summary.apply(lambda r: f"{r['Test_MAE_mean']:.3f} ± {r['Test_MAE_SD']:.3f}", axis=1)
    return summary


def export_best_params(results: pd.DataFrame, output_folder: Path | str, file_stem: str = "Best_Hyperparameters_25_outer_tasks") -> pd.DataFrame:
    rows = []
    param_keys = set()
    parsed = []
    for _, r in results.iterrows():
        params = json.loads(r["Best_Params_JSON"]) if isinstance(r["Best_Params_JSON"], str) else {}
        parsed.append(params)
        param_keys.update(params.keys())
    param_keys = sorted(param_keys)
    for (_, r), params in zip(results.iterrows(), parsed):
        row = {
            "Model": r["Model"], "Repeat": int(r["Repeat"]), "Outer_Fold": int(r["Outer_Fold"]),
            "Inner_CV_R2_mean": r["Inner_CV_R2_mean"], "Test_R2": r["Test_R2"],
            "XGB_Final_n_estimators": r.get("XGB_Final_n_estimators", np.nan),
            "Tuning_Status": "No tuning; baseline" if r["Model"] == "OLS" else "Optuna-TPE",
        }
        for k in param_keys:
            row[k] = params.get(k, np.nan)
        rows.append(row)
    out_df = pd.DataFrame(rows).sort_values(["Model", "Repeat", "Outer_Fold"])
    out = ensure_dir(output_folder)
    out_df.to_csv(out / f"{file_stem}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / f"{file_stem}.xlsx", engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="AllModels", index=False)
        for model, g in out_df.groupby("Model"):
            g.to_excel(writer, sheet_name=sanitize_filename(model)[:31], index=False)
    return out_df


# =============================================================================
# 6. Top-2 repeated k-fold corrected t-test + Wilcoxon
# =============================================================================
def corrected_repeated_kfold_t_test(
    results: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "Test_R2",
    r: int = N_REPEATS,
    k: int = OUTER_N_SPLITS,
) -> Dict[str, Any]:
    a = results[results["Model"] == model_a][["Repeat", "Outer_Fold", metric, "Test_Train_Sample_Ratio"]].copy()
    b = results[results["Model"] == model_b][["Repeat", "Outer_Fold", metric]].copy()
    m = a.merge(b, on=["Repeat", "Outer_Fold"], suffixes=("_A", "_B"))
    if len(m) != r * k:
        raise ValueError(f"Top-2 配对任务应为 {r*k}，实际 {len(m)}。")
    d = m[f"{metric}_A"].to_numpy(float) - m[f"{metric}_B"].to_numpy(float)
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    mean_ratio = float(m["Test_Train_Sample_Ratio"].mean())
    correction = (1.0 / (r * k)) + mean_ratio
    se = math.sqrt(correction * (sd_d ** 2)) if sd_d > 0 else 0.0
    t_stat = mean_d / se if se > 0 else (np.inf if mean_d > 0 else -np.inf if mean_d < 0 else 0.0)
    dfree = r * k - 1
    p_value = float(2 * stats.t.sf(abs(t_stat), df=dfree)) if np.isfinite(t_stat) else 0.0

    try:
        wil = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        wil_stat, wil_p = float(wil.statistic), float(wil.pvalue)
    except Exception:
        wil_stat, wil_p = np.nan, np.nan

    return {
        "Model_A": model_a,
        "Model_B": model_b,
        "Metric": metric,
        "r_repeats": r,
        "k_folds": k,
        "n_paired_tasks": len(d),
        "Mean_A_minus_B": mean_d,
        "SD_difference": sd_d,
        "Mean_n_test_over_n_train": mean_ratio,
        "Correction_factor": correction,
        "Corrected_SE": se,
        "Corrected_t": float(t_stat),
        "df": dfree,
        "Corrected_t_two_sided_p": p_value,
        "Wilcoxon_statistic": wil_stat,
        "Wilcoxon_two_sided_p": wil_p,
        "Primary_Test": "Repeated k-fold corrected t-test",
        "Supplementary_Test": "Wilcoxon signed-rank",
    }


def compare_top2_models(results: pd.DataFrame, summary: pd.DataFrame, output_folder: Path | str) -> pd.DataFrame:
    if len(summary) < 2:
        raise ValueError("模型不足 2 个，无法 Top-2 比较。")
    a = str(summary.iloc[0]["Model"])
    b = str(summary.iloc[1]["Model"])
    result = corrected_repeated_kfold_t_test(results, a, b, metric="Test_R2")
    out = pd.DataFrame([result])
    out.to_csv(Path(output_folder) / "Top2_RepeatedKFold_Corrected_t_and_Wilcoxon.csv", index=False, encoding="utf-8-sig")
    return out


# =============================================================================
# 7. 模型性能图
# =============================================================================
def plot_model_performance(summary: pd.DataFrame, figure_folder: Path | str) -> None:
    figure_folder = ensure_dir(figure_folder)
    specs = [
        ("Test_R2_mean", "Test_R2_SD", "Nested CV Test R²", True),
        ("Test_MSE_x1e3_mean", "Test_MSE_x1e3_SD", "Nested CV Test MSE (×10⁻³)", False),
        ("Test_RMSE_mean", "Test_RMSE_SD", "Nested CV Test RMSE", False),
        ("Test_MAE_mean", "Test_MAE_SD", "Nested CV Test MAE", False),
        ("Train_Test_R2_Gap_mean", "Train_Test_R2_Gap_SD", "Mean Train–Test R² gap", False),
    ]
    for idx, (metric, sdcol, ylabel, higher) in enumerate(specs):
        d = summary.sort_values(metric, ascending=not higher).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        x = np.arange(len(d))
        cmap = get_cmap_safely("viridis")
        colors = cmap(np.linspace(0.22, 0.86, len(d)))
        bars = ax.bar(x, d[metric], color=colors, edgecolor="black", linewidth=0.7, width=0.68)
        if sdcol in d.columns:
            ax.errorbar(x, d[metric], yerr=d[sdcol], fmt="none", ecolor="black", capsize=4, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(d["Model"], rotation=18, ha="right")
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, d[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
        add_panel_label(ax, panel_label(idx), x=0.01, y=0.99)
        force_figure_times_new_roman(fig)
        fig.tight_layout()
        save_figure_formats(fig, figure_folder / f"Model_Comparison_{metric}")

    # R² + RMSE 组合图
    d = summary.sort_values("Test_R2_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(d)); width = 0.34
    fig, ax1 = plt.subplots(figsize=(9.4, 5.8)); ax2 = ax1.twinx()
    cmap = get_cmap_safely("viridis")
    colors = cmap(np.linspace(0.25, 0.8, 2))
    b1 = ax1.bar(x-width/2, d["Test_R2_mean"], width, yerr=d["Test_R2_SD"], capsize=3, color=colors[0], edgecolor="black", linewidth=0.6, label="Test R²")
    b2 = ax2.bar(x+width/2, d["Test_RMSE_mean"], width, yerr=d["Test_RMSE_SD"], capsize=3, color=colors[1], edgecolor="black", linewidth=0.6, alpha=0.82, label="Test RMSE")
    ax1.set_xticks(x); ax1.set_xticklabels(d["Model"], rotation=18, ha="right")
    ax1.set_xlabel("Model"); ax1.set_ylabel("Test R²"); ax2.set_ylabel("Test RMSE")
    ax1.set_title("Nested Group CV model comparison")
    ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    handles = [b1, b2]; labels = ["Test R²", "Test RMSE"]
    ax1.legend(handles, labels, frameon=False, loc="best")
    add_panel_label(ax1, "(a)")
    force_figure_times_new_roman(fig); fig.tight_layout()
    save_figure_formats(fig, figure_folder / "Model_Comparison_TestR2_RMSE_Combo")


def plot_selected_oof_observed_predicted(
    results_folder: Path | str,
    selected_model: str,
    figure_folder: Path | str,
) -> pd.DataFrame:
    files = sorted(Path(results_folder).glob(f"tasks/{selected_model}/repeat_*/fold_*/test_predictions.csv"))
    if not files:
        return pd.DataFrame()
    allp = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    avg = allp.groupby(SAMPLE_ID_COLUMN, as_index=False).agg(
        observed_RI=("observed_RI", "first"),
        predicted_RI=("predicted_RI", "mean"),
        prediction_SD=("predicted_RI", "std"),
        patch_id1=(GROUP_COLUMN, "first"),
        Year=(YEAR_COLUMN, "first"),
    )
    avg.to_csv(Path(results_folder) / f"{selected_model}_OOF_predictions_mean_across_5_repeats.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(avg["observed_RI"], avg["predicted_RI"], s=13, alpha=0.42, edgecolors="none")
    lo = min(avg["observed_RI"].min(), avg["predicted_RI"].min())
    hi = max(avg["observed_RI"].max(), avg["predicted_RI"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="black")
    m = evaluate_model(avg["observed_RI"], avg["predicted_RI"])
    ax.text(0.04, 0.95, f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.3f}\nMAE = {m['MAE']:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=10)
    ax.set_xlabel("Observed RI"); ax.set_ylabel("Predicted RI"); ax.set_title(f"{selected_model}: repeated OOF prediction")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    add_panel_label(ax, "(a)"); force_figure_times_new_roman(fig); fig.tight_layout()
    save_figure_formats(fig, Path(figure_folder) / f"{selected_model}_OOF_Observed_vs_Predicted")
    return avg


# =============================================================================
# 8. SHAP：25 外层 OOF、稳定性、Kendall W、cluster bootstrap
# =============================================================================
def _imputed_original_frame(fitted_model, X: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    if isinstance(fitted_model, XGBModelBundle):
        return fitted_model.transform(X)
    if isinstance(fitted_model, Pipeline):
        imp = fitted_model.named_steps.get("imputer")
        if imp is None:
            return X[list(feature_names)].copy()
        arr = imp.transform(X[list(feature_names)])
        return pd.DataFrame(arr, columns=list(feature_names), index=X.index)
    return X[list(feature_names)].copy()


def _extract_shap_values(result) -> np.ndarray:
    values = result.values if hasattr(result, "values") else result
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"SHAP 维度异常：{values.shape}")
    return values


def compute_shap_for_fold(
    model_name: str,
    fitted_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: Sequence[str],
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, str]:
    feature_names = list(feature_names)
    X_eval = X_test.copy()
    if model_name == "SVR" and SHAP_NON_TREE_MAX_SAMPLES_PER_FOLD is not None and len(X_eval) > SHAP_NON_TREE_MAX_SAMPLES_PER_FOLD:
        X_eval = X_eval.sample(SHAP_NON_TREE_MAX_SAMPLES_PER_FOLD, random_state=random_state).sort_index()
    X_plot = _imputed_original_frame(fitted_model, X_eval, feature_names)

    if model_name in TREE_MODELS:
        core = fitted_model.model if isinstance(fitted_model, XGBModelBundle) else fitted_model.named_steps["model"]
        explainer = shap.TreeExplainer(core)
        try:
            result = explainer(X_plot)
        except Exception:
            result = explainer.shap_values(X_plot)
        values = _extract_shap_values(result)
        method = "TreeExplainer/TreeSHAP"
    elif model_name == "OLS":
        imp = fitted_model.named_steps["imputer"]
        scaler = fitted_model.named_steps["scaler"]
        core = fitted_model.named_steps["model"]
        Xtr_i = imp.transform(X_train[feature_names]); Xev_i = imp.transform(X_eval[feature_names])
        Xtr_s = scaler.transform(Xtr_i); Xev_s = scaler.transform(Xev_i)
        rng = np.random.default_rng(random_state)
        bsize = min(SHAP_BACKGROUND_SAMPLES, len(Xtr_s))
        bidx = rng.choice(len(Xtr_s), size=bsize, replace=False)
        explainer = shap.LinearExplainer(core, Xtr_s[bidx])
        try:
            result = explainer(Xev_s)
        except Exception:
            result = explainer.shap_values(Xev_s)
        values = _extract_shap_values(result)
        method = "LinearExplainer"
    elif model_name == "SVR":
        Xtr_plot = _imputed_original_frame(fitted_model, X_train, feature_names)
        background = Xtr_plot.sample(min(SHAP_BACKGROUND_SAMPLES, len(Xtr_plot)), random_state=random_state)
        def pred_fn(arr):
            frame = pd.DataFrame(arr, columns=feature_names)
            return fitted_model.predict(frame)
        explainer = shap.Explainer(pred_fn, background, algorithm="permutation", feature_names=feature_names, seed=random_state)
        result = explainer(X_plot, max_evals=2 * len(feature_names) + 1, batch_size=32)
        values = _extract_shap_values(result)
        method = "PermutationExplainer"
    else:
        raise ValueError(model_name)

    if values.shape[1] != len(feature_names):
        raise ValueError(f"SHAP 特征数 {values.shape[1]} != {len(feature_names)}")
    return X_plot, values, method


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W，行=judge/model，列=item/feature；支持 ties。"""
    ranks = np.asarray(rank_matrix, dtype=float)
    if ranks.ndim != 2:
        raise ValueError("rank_matrix 必须二维")
    m, n = ranks.shape
    Rj = ranks.sum(axis=0)
    Rbar = m * (n + 1) / 2.0
    S = float(np.sum((Rj - Rbar) ** 2))
    tie_correction = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_correction += float(np.sum(counts**3 - counts))
    denom = m**2 * (n**3 - n) - m * tie_correction
    return float(12 * S / denom) if denom > 0 else np.nan


def _cluster_bootstrap_contribution(
    observation_summary: pd.DataFrame,
    driver_features: Sequence[str],
    group_column: str = GROUP_COLUMN,
    B: int = SHAP_CLUSTER_BOOTSTRAP_N,
    seed: int = SHAP_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    abs_cols = [f"mean_abs_SHAP_{f}" for f in driver_features]
    # 每个格网先聚合 sum 和 n；bootstrap 抽中格网时纳入该格网全部年份。
    gsum = observation_summary.groupby(group_column)[abs_cols].sum()
    gcount = observation_summary.groupby(group_column).size()
    groups = gsum.index.to_numpy()
    rng = np.random.default_rng(seed)
    contribs = np.zeros((B, len(driver_features)), dtype=float)
    for b in range(B):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sum_abs = gsum.loc[sampled].to_numpy().sum(axis=0)
        total_n = gcount.loc[sampled].to_numpy().sum()
        mean_abs = sum_abs / total_n
        denom = mean_abs.sum()
        contribs[b, :] = mean_abs / denom * 100.0
    rows = []
    for j, f in enumerate(driver_features):
        rows.append({
            "feature": f,
            "bootstrap_B": B,
            "CI_lower_2.5": float(np.quantile(contribs[:, j], 0.025)),
            "CI_upper_97.5": float(np.quantile(contribs[:, j], 0.975)),
            "bootstrap_mean": float(np.mean(contribs[:, j])),
            "bootstrap_SD": float(np.std(contribs[:, j], ddof=1)),
        })
    return pd.DataFrame(rows)


def _oof_shap_paths(fold_folder: Path, repeat_id: int, fold_id: int) -> Tuple[Path, Path, Path]:
    stem = f"repeat_{repeat_id:02d}_fold_{fold_id:02d}"
    return (
        fold_folder / f"{stem}_SHAP.csv",
        fold_folder / f"{stem}_importance.csv",
        fold_folder / f"{stem}.done",
    )


def _run_one_oof_shap_fold(
    df: pd.DataFrame,
    features: Sequence[str],
    selected_model: str,
    task: Dict[str, Any],
    result_row: Dict[str, Any],
    fold_folder: str,
) -> Dict[str, Any]:
    """计算一个外层 OOF-SHAP 折，并在全部输出落盘后写完成标记。

    此函数必须保持在模块顶层，以便 Windows 的多进程 worker 安全调用。
    """
    r, f, seed = int(task["repeat"]), int(task["fold"]), int(task["seed"])
    fold_folder_path = Path(fold_folder)
    fold_file, fold_imp_file, done_file = _oof_shap_paths(fold_folder_path, r, f)

    params_raw = result_row.get("Best_Params_JSON", "{}")
    params = json.loads(params_raw) if isinstance(params_raw, str) else {}
    n_est = result_row.get("XGB_Final_n_estimators", np.nan)
    n_est = None if pd.isna(n_est) else int(n_est)
    tr, te = task["train_idx"], task["test_idx"]
    X_all = df[list(features)].copy()
    y_all = df[TARGET].copy()
    X_train, X_test = X_all.iloc[tr].copy(), X_all.iloc[te].copy()
    y_train = y_all.iloc[tr].copy()

    # 多个 SHAP 折并行时，重建解释模型只使用一个核，避免 RF/XGBoost 的
    # 内部 n_jobs 与外层进程数相乘造成过度抢占。随机种子与模型参数不变。
    model = fit_outer_model(
        selected_model, params, n_est, X_train, y_train,
        seed + r * 100 + f, features, model_n_jobs=1,
    )
    X_plot, values, method = compute_shap_for_fold(
        selected_model, model, X_train, X_test, features, seed + r * 1000 + f,
    )

    ids = X_plot.index.to_numpy(dtype=int)
    out = pd.DataFrame({
        SAMPLE_ID_COLUMN: df.loc[ids, SAMPLE_ID_COLUMN].to_numpy(),
        GROUP_COLUMN: df.loc[ids, GROUP_COLUMN].astype(str).to_numpy(),
        YEAR_COLUMN: df.loc[ids, YEAR_COLUMN].to_numpy(),
        "repeat": r,
        "fold": f,
    })
    if BLOCK100_COLUMN in df.columns:
        out[BLOCK100_COLUMN] = df.loc[ids, BLOCK100_COLUMN].astype(str).to_numpy()
    for feat in features:
        out[feat] = X_plot[feat].to_numpy()
    for j, feat in enumerate(features):
        out[f"SHAP_{feat}"] = values[:, j]

    fold_imp = [
        {
            "repeat": r,
            "fold": f,
            "feature": feat,
            "mean_abs_SHAP": float(np.mean(np.abs(values[:, j]))),
            "mean_signed_SHAP": float(np.mean(values[:, j])),
            "SHAP_method": method,
            "n_test_samples_explained": len(values),
        }
        for j, feat in enumerate(features)
    ]

    # 临时文件避免中途停止时留下貌似完整的 CSV；只有三个最终文件齐备后
    # 才写 .done，因此下一次运行可以可靠地跳过或重算该折。
    pid = os.getpid()
    shap_tmp = fold_file.with_name(f"{fold_file.stem}.partial_{pid}.csv")
    imp_tmp = fold_imp_file.with_name(f"{fold_imp_file.stem}.partial_{pid}.csv")
    out.to_csv(shap_tmp, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_imp).to_csv(imp_tmp, index=False, encoding="utf-8-sig")
    os.replace(shap_tmp, fold_file)
    os.replace(imp_tmp, fold_imp_file)
    done_file.write_text(datetime.now().isoformat(), encoding="utf-8")
    del model
    gc.collect()
    return {"repeat": r, "fold": f, "method": method}


def run_oof_shap(
    df: pd.DataFrame,
    features: Sequence[str],
    driver_features: Sequence[str],
    selected_model: str,
    outer_tasks: Sequence[Dict[str, Any]],
    model_results: pd.DataFrame,
    output_folder: Path | str,
    logger: logging.Logger,
    group_column_for_training: str = GROUP_COLUMN,
    parallel_jobs: int = SHAP_OUTER_PARALLEL_JOBS,
    progress_label: str = "OOF-SHAP",
) -> Dict[str, Any]:
    output_folder = ensure_dir(output_folder)
    fold_folder = ensure_dir(output_folder / "fold_shap")
    X_all = df[list(features)].copy(); y_all = df[TARGET].copy()
    result_lookup = model_results[model_results["Model"] == selected_model].set_index(["Repeat", "Outer_Fold"])
    shap_files = []
    importance_rows = []
    method_names = set()
    tasks_to_compute = list(outer_tasks)

    # 多进程路径：每个 worker 独立重建一个外层模型并写入本折专属文件。
    # 已完成折始终以 .done + 两个 CSV 为准，因此可安全断点续跑。
    requested_workers = max(1, int(parallel_jobs))
    if requested_workers > 1:
        pending_folds: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        completed_count = 0
        for task in outer_tasks:
            r, f = int(task["repeat"]), int(task["fold"])
            fold_file, fold_imp_file, done_file = _oof_shap_paths(fold_folder, r, f)
            if done_file.exists() and fold_file.exists() and fold_imp_file.exists():
                completed_count += 1
            else:
                pending_folds.append((task, result_lookup.loc[(r, f)].to_dict()))

        progress = ConsoleProgress(progress_label, len(outer_tasks), completed_at_start=completed_count)
        progress.show(completed_count, "checkpoint scan complete")
        if pending_folds:
            workers = min(requested_workers, len(pending_folds))
            logger.info(
                f"OOF-SHAP parallel mode: {workers} worker process(es), {len(pending_folds)} unfinished fold(s)."
            )
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _run_one_oof_shap_fold,
                        df,
                        list(features),
                        selected_model,
                        task,
                        result_row,
                        str(fold_folder),
                    )
                    for task, result_row in pending_folds
                ]
                for future in as_completed(futures):
                    result = future.result()
                    completed_count += 1
                    logger.info(f"OOF-SHAP completed: R{result['repeat']}F{result['fold']}")
                    progress.show(completed_count, f"completed R{result['repeat']}F{result['fold']}")

        # 由下面的汇总代码统一从完成检查点读取，不再进入旧的串行循环。
        shap_files = []
        importance_rows = []
        method_names = set()
        for task in outer_tasks:
            r, f = int(task["repeat"]), int(task["fold"])
            fold_file, fold_imp_file, done_file = _oof_shap_paths(fold_folder, r, f)
            if not (done_file.exists() and fold_file.exists() and fold_imp_file.exists()):
                raise RuntimeError(f"OOF-SHAP fold R{r}F{f} did not create a complete checkpoint.")
            shap_files.append(fold_file)
            fold_records = pd.read_csv(fold_imp_file).to_dict("records")
            importance_rows.extend(fold_records)
            method_names.update(str(record.get("SHAP_method", "unknown")) for record in fold_records)
        tasks_to_compute = []

    for task in tasks_to_compute:
        r, f, seed = int(task["repeat"]), int(task["fold"]), int(task["seed"])
        fold_file = fold_folder / f"repeat_{r:02d}_fold_{f:02d}_SHAP.csv"
        fold_imp_file = fold_folder / f"repeat_{r:02d}_fold_{f:02d}_importance.csv"
        done = fold_folder / f"repeat_{r:02d}_fold_{f:02d}.done"
        if done.exists() and fold_file.exists() and fold_imp_file.exists():
            logger.info(f"OOF-SHAP 跳过已完成 R{r}F{f}")
            shap_files.append(fold_file)
            importance_rows.extend(pd.read_csv(fold_imp_file).to_dict("records"))
            continue

        row = result_lookup.loc[(r, f)]
        params = json.loads(row["Best_Params_JSON"]) if isinstance(row["Best_Params_JSON"], str) else {}
        n_est = row.get("XGB_Final_n_estimators", np.nan)
        n_est = None if pd.isna(n_est) else int(n_est)
        tr, te = task["train_idx"], task["test_idx"]
        X_train, X_test = X_all.iloc[tr].copy(), X_all.iloc[te].copy()
        y_train = y_all.iloc[tr].copy()
        model = fit_outer_model(selected_model, params, n_est, X_train, y_train, seed + r * 100 + f, features)
        X_plot, values, method = compute_shap_for_fold(selected_model, model, X_train, X_test, features, seed + r * 1000 + f)
        method_names.add(method)

        # X_plot index 对应原始 df 行位置；df 的 index 与 sample_id 一致（load_model_data 后 reset）。
        ids = X_plot.index.to_numpy(dtype=int)
        out = pd.DataFrame({
            SAMPLE_ID_COLUMN: df.loc[ids, SAMPLE_ID_COLUMN].to_numpy(),
            GROUP_COLUMN: df.loc[ids, GROUP_COLUMN].astype(str).to_numpy(),
            YEAR_COLUMN: df.loc[ids, YEAR_COLUMN].to_numpy(),
            "repeat": r,
            "fold": f,
        })
        if BLOCK100_COLUMN in df.columns:
            out[BLOCK100_COLUMN] = df.loc[ids, BLOCK100_COLUMN].astype(str).to_numpy()
        for feat in features:
            out[feat] = X_plot[feat].to_numpy()
        for j, feat in enumerate(features):
            out[f"SHAP_{feat}"] = values[:, j]
        out.to_csv(fold_file, index=False, encoding="utf-8-sig")
        shap_files.append(fold_file)

        fold_imp = []
        for j, feat in enumerate(features):
            fold_imp.append({
                "repeat": r, "fold": f, "feature": feat,
                "mean_abs_SHAP": float(np.mean(np.abs(values[:, j]))),
                "mean_signed_SHAP": float(np.mean(values[:, j])),
                "SHAP_method": method,
                "n_test_samples_explained": len(values),
            })
        pd.DataFrame(fold_imp).to_csv(fold_imp_file, index=False, encoding="utf-8-sig")
        importance_rows.extend(fold_imp)
        done.write_text(datetime.now().isoformat(), encoding="utf-8")
        del model
        gc.collect()

    all_shap = pd.concat([pd.read_csv(p) for p in shap_files], ignore_index=True)

    # OOF 覆盖完整性：正常正式设置下，每个 repeat 的 5 个测试折应恰好覆盖全体观测一次，
    # 每个观测在 5 个 repeat 中最终拥有 5 套真正的 OOF-SHAP。
    coverage_rows = []
    repeat_ids = sorted({int(t["repeat"]) for t in outer_tasks})
    full_explanation_expected = not (selected_model == "SVR" and SHAP_NON_TREE_MAX_SAMPLES_PER_FOLD is not None)
    for rr in repeat_ids:
        gr = all_shap[all_shap["repeat"] == rr]
        coverage_rows.append({
            "repeat": rr,
            "rows": len(gr),
            "unique_sample_ids": int(gr[SAMPLE_ID_COLUMN].nunique()),
            "expected_unique_sample_ids": len(df),
            "complete": bool(gr[SAMPLE_ID_COLUMN].nunique() == len(df)) if full_explanation_expected else "Not required due to configured SVR SHAP subsampling",
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(output_folder / "OOF_SHAP_coverage_check.csv", index=False, encoding="utf-8-sig")
    if full_explanation_expected:
        if not all(coverage_df["unique_sample_ids"] == len(df)):
            raise RuntimeError("至少一个 repeat 的 OOF-SHAP 未完整覆盖全体观测。")
        sample_repeat_counts = all_shap.groupby(SAMPLE_ID_COLUMN)["repeat"].nunique()
        if not (sample_repeat_counts == len(repeat_ids)).all():
            raise RuntimeError("至少一个观测没有获得每个 repeat 各 1 套 OOF-SHAP。")

    all_shap.to_csv(output_folder / "OOF_SHAP_all_25_outer_models.csv", index=False, encoding="utf-8-sig")
    fold_imp_df = pd.DataFrame(importance_rows)
    fold_imp_df.to_csv(output_folder / "SHAP_importance_25models.csv", index=False, encoding="utf-8-sig")

    # 每个 repeat 拼接 5 折后的 OOF importance/contribution。
    repeat_rows = []
    for r, g in all_shap.groupby("repeat"):
        mean_abs = {f: float(np.mean(np.abs(g[f"SHAP_{f}"].to_numpy()))) for f in features}
        denom = sum(mean_abs[f] for f in driver_features)
        for feat in features:
            repeat_rows.append({
                "repeat": int(r), "feature": feat, "mean_abs_SHAP": mean_abs[feat],
                "driver_contribution_percent": mean_abs[feat] / denom * 100.0 if feat in driver_features else np.nan,
            })
    repeat_imp = pd.DataFrame(repeat_rows)
    repeat_imp.to_csv(output_folder / "SHAP_importance_and_contribution_by_repeat.csv", index=False, encoding="utf-8-sig")

    # 同一观测 5 次 signed SHAP 与 abs SHAP 分别平均。
    agg_dict = {GROUP_COLUMN: "first", YEAR_COLUMN: "first"}
    if BLOCK100_COLUMN in all_shap.columns:
        agg_dict[BLOCK100_COLUMN] = "first"
    for feat in features:
        agg_dict[feat] = "first"
        agg_dict[f"SHAP_{feat}"] = "mean"
    obs_signed = all_shap.groupby(SAMPLE_ID_COLUMN, as_index=False).agg(agg_dict)
    obs_signed = obs_signed.rename(columns={f"SHAP_{f}": f"mean_signed_SHAP_{f}" for f in features})

    abs_temp = all_shap[[SAMPLE_ID_COLUMN] + [f"SHAP_{f}" for f in features]].copy()
    for feat in features:
        abs_temp[f"SHAP_{feat}"] = abs_temp[f"SHAP_{feat}"].abs()
    obs_abs = abs_temp.groupby(SAMPLE_ID_COLUMN, as_index=False).mean().rename(columns={f"SHAP_{f}": f"mean_abs_SHAP_{f}" for f in features})
    obs_summary = obs_signed.merge(obs_abs, on=SAMPLE_ID_COLUMN, how="inner")
    obs_summary.to_csv(output_folder / "OOF_SHAP_mean_across_5_repeats_per_observation.csv", index=False, encoding="utf-8-sig")

    # 25 外层模型 mean|SHAP| mean±SD + mean rank。
    driver_fold = fold_imp_df[fold_imp_df["feature"].isin(driver_features)].copy()
    pivot = driver_fold.pivot_table(index=["repeat", "fold"], columns="feature", values="mean_abs_SHAP")
    ordered_drivers = list(driver_features)
    pivot = pivot[ordered_drivers]
    rank_matrix = np.vstack([rankdata(-row, method="average") for row in pivot.to_numpy()])
    W_all = kendalls_w(rank_matrix)
    rank_df = pd.DataFrame(rank_matrix, columns=ordered_drivers, index=pivot.index).reset_index()
    rank_df.to_csv(output_folder / "SHAP_rankings_25models.csv", index=False, encoding="utf-8-sig")

    kendall_rows = [{"scope": "All 25 outer models", "repeat": "All", "n_rankings": len(rank_matrix), "Kendalls_W": W_all}]
    for r in sorted(rank_df["repeat"].unique()):
        rr = rank_df[rank_df["repeat"] == r][ordered_drivers].to_numpy(float)
        kendall_rows.append({"scope": "Within repeat (5 folds)", "repeat": int(r), "n_rankings": len(rr), "Kendalls_W": kendalls_w(rr)})
    kendall_df = pd.DataFrame(kendall_rows)
    kendall_df.to_csv(output_folder / "Kendall_W_SHAP_rank_stability.csv", index=False, encoding="utf-8-sig")

    fold_stats = driver_fold.groupby("feature")["mean_abs_SHAP"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "mean_abs_SHAP_25models_mean", "std": "mean_abs_SHAP_25models_SD"})
    mean_rank = rank_df[ordered_drivers].mean(axis=0).rename("mean_rank_25models").reset_index().rename(columns={"index": "feature"})

    # 最终贡献率：先对每个观测的 5 次 abs SHAP 取平均，再对全部观测求 mean。
    overall_abs = {f: float(obs_summary[f"mean_abs_SHAP_{f}"].mean()) for f in driver_features}
    denom = sum(overall_abs.values())
    overall_contrib = pd.DataFrame({
        "feature": list(driver_features),
        "overall_mean_abs_SHAP_repeat_averaged": [overall_abs[f] for f in driver_features],
        "contribution_percent": [overall_abs[f] / denom * 100.0 for f in driver_features],
    })
    boot = _cluster_bootstrap_contribution(obs_summary, driver_features, GROUP_COLUMN, SHAP_CLUSTER_BOOTSTRAP_N, SHAP_BOOTSTRAP_SEED)
    repeat_stats = repeat_imp[repeat_imp["feature"].isin(driver_features)].groupby("feature")["driver_contribution_percent"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "repeat_contribution_mean", "std": "repeat_contribution_SD"})
    contribution = overall_contrib.merge(boot, on="feature").merge(fold_stats, on="feature").merge(mean_rank, on="feature").merge(repeat_stats, on="feature")
    contribution["Kendalls_W_all_25"] = W_all
    contribution = contribution.sort_values("contribution_percent", ascending=False).reset_index(drop=True)
    contribution.insert(0, "rank", np.arange(1, len(contribution) + 1))
    contribution.to_csv(output_folder / "SHAP_contribution_bootstrap_CI_and_stability.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_folder / "SHAP_summary_tables.xlsx", engine="openpyxl") as writer:
        contribution.to_excel(writer, sheet_name="Contribution_CI", index=False)
        fold_imp_df.to_excel(writer, sheet_name="Importance_25models", index=False)
        repeat_imp.to_excel(writer, sheet_name="Repeat_importance", index=False)
        kendall_df.to_excel(writer, sheet_name="Kendall_W", index=False)

    return {
        "all_shap": all_shap,
        "observation_summary": obs_summary,
        "fold_importance": fold_imp_df,
        "repeat_importance": repeat_imp,
        "contribution": contribution,
        "kendall": kendall_df,
        "shap_method": ", ".join(sorted(method_names)) if method_names else "loaded_from_checkpoint",
    }


# =============================================================================
# 9. SHAP 图：贡献率CI、条形图、蜂群、玫瑰、dependence 单图+组合图
# =============================================================================
def plot_shap_contribution_ci(contribution: pd.DataFrame, figure_folder: Path | str, model_name: str) -> None:
    d = contribution.sort_values("contribution_percent", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    x = d["contribution_percent"].to_numpy()
    xerr = np.vstack([x - d["CI_lower_2.5"].to_numpy(), d["CI_upper_97.5"].to_numpy() - x])
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.errorbar(x, y, xerr=xerr, fmt="o", markersize=6, capsize=4, linewidth=1.3)
    ax.set_yticks(y); ax.set_yticklabels([display_name(f) for f in d["feature"]])
    ax.set_xlabel("Relative contribution (%)"); ax.set_ylabel("")
    ax.set_title(f"{model_name}–SHAP relative contribution")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.7)
    for yi, xi in zip(y, x):
        ax.text(xi, yi + 0.20, f"{xi:.2f}%", ha="center", va="bottom", fontsize=9)
    add_panel_label(ax, "(a)"); force_figure_times_new_roman(fig); fig.tight_layout()
    save_figure_formats(fig, Path(figure_folder) / f"{model_name}_SHAP_Contribution_95CI")


def _ordered_beeswarm_arrays(obs_summary: pd.DataFrame, contribution: pd.DataFrame, driver_features: Sequence[str]):
    ordered = contribution.sort_values("contribution_percent", ascending=False)["feature"].tolist()
    X = obs_summary[ordered].copy()
    X.columns = [display_name(f) for f in ordered]
    shap_values = np.column_stack([obs_summary[f"mean_signed_SHAP_{f}"].to_numpy() for f in ordered])
    return ordered, X, shap_values


def draw_shap_bar(ax, contribution: pd.DataFrame, title: str) -> None:
    d = contribution.sort_values("contribution_percent", ascending=False).reset_index(drop=True)
    y = np.arange(len(d)); vals = d["contribution_percent"].to_numpy()
    cmap = get_cmap_safely("viridis")
    colors = cmap(np.linspace(0.25, 0.90, len(d)))
    bars = ax.barh(y, vals, color=colors, edgecolor="none", height=0.74)
    ax.set_yticks(y); ax.set_yticklabels([display_name(f) for f in d["feature"]]); ax.invert_yaxis()
    ax.set_xlabel("Relative mean |SHAP| importance (%)"); ax.set_title(title, pad=10)
    maxv = max(vals) if len(vals) else 1
    ax.set_xlim(0, maxv * 1.22)
    for b, v in zip(bars, vals):
        ax.text(v + maxv*0.015, b.get_y()+b.get_height()/2, f"{v:.2f}%", va="center", fontsize=9.5, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    force_times_new_roman(ax)


def _style_beeswarm_axis(ax) -> None:
    ax.set_title("OOF-SHAP Summary Plot", fontsize=15, pad=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    force_times_new_roman(ax)


def _relocate_shap_colorbar(fig, main_ax, position) -> None:
    for a in fig.get_axes():
        if a is main_ax:
            continue
        bbox = a.get_position()
        if bbox.width < 0.10:
            a.set_position(position)
            a.set_ylabel("Feature value", fontsize=10, fontfamily="Times New Roman")
            for t in a.get_yticklabels():
                t.set_fontfamily("Times New Roman")
            break


def rose_text_rotation(angle_degrees: float) -> float:
    angle = float(angle_degrees) % 360.0
    rotation = 90.0 - angle
    while rotation > 90.0: rotation -= 180.0
    while rotation < -90.0: rotation += 180.0
    if 120.0 <= angle <= 240.0:
        rotation = float(np.clip(rotation, -15.0, 15.0))
    else:
        rotation = float(np.clip(rotation, -30.0, 30.0))
    return rotation


def draw_adaptive_rose(ax, contribution: pd.DataFrame, cmap_name: str = "Spectral_r") -> None:
    d = contribution.sort_values("contribution_percent", ascending=False).reset_index(drop=True)
    vals = d["contribution_percent"].to_numpy(float)
    raw = d["overall_mean_abs_SHAP_repeat_averaged"].to_numpy(float)
    names = [display_name(f) for f in d["feature"]]
    n = len(d); theta = np.linspace(0, 2*np.pi, n, endpoint=False); width = 2*np.pi/n*0.82
    vmax, vmin = float(vals.max()), float(vals.min())
    inner = max(15.0, vmax*0.62); gap = max(0.55, vmax*0.018); bottom = inner+gap
    cmap = get_cmap_safely(cmap_name); norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmax != vmin else vmin+1e-9)
    ax.bar(theta, vals, width=width, bottom=bottom, color=cmap(norm(vals)), edgecolor="black", linewidth=1.0, zorder=20)
    circ = np.linspace(0, 2*np.pi, 800); ax.fill_between(circ, 0, inner, color="white", zorder=40); ax.plot(circ, np.full_like(circ, inner), color="black", linewidth=1.2, zorder=60)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_ylim(0, bottom+vmax+max(11, vmax*0.28)); ax.set_axis_off()
    for i, (ang, val, rawv, name) in enumerate(zip(theta, vals, raw, names)):
        txt = ax.text(ang, bottom+val*0.50, f"{rawv:.3f}", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold", zorder=100)
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])
        deg = np.degrees(ang); extra = max(5.0, vmax*0.14) + (2.0 if i == 0 else 0)
        lab = ax.text(ang, bottom+val+extra, f"{name}\n{val:.2f}%", ha="center", va="center", rotation=rose_text_rotation(deg), rotation_mode="anchor", fontsize=11, fontweight="bold", zorder=120, clip_on=False)
        lab.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = ax.inset_axes([0.475, 0.375, 0.045, 0.25], transform=ax.transAxes, zorder=200)
    cb = ax.figure.colorbar(sm, cax=cax, orientation="vertical"); cb.set_ticks(np.linspace(vmin, vmax, 3)); cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f")); cb.set_label("Importance (%)", fontsize=9)
    for t in cb.ax.get_yticklabels(): t.set_fontfamily("Times New Roman")


def add_binned_smooth_line(ax, x: Sequence[float], y: Sequence[float], bins: int = 40) -> None:
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y); x = x[valid]; y = y[valid]
    if len(x) < 20: return
    order = np.argsort(x); x = x[order]; y = y[order]
    edges = np.linspace(0, len(x), bins+1).astype(int)
    xx, yy = [], []
    for i in range(bins):
        s, e = edges[i], edges[i+1]
        if e > s:
            xx.append(np.mean(x[s:e])); yy.append(np.mean(y[s:e]))
    ax.plot(xx, yy, color="lightcoral", linewidth=2.0, label="Binned smooth curve")


def plot_shap_outputs(
    shap_result: Dict[str, Any],
    driver_features: Sequence[str],
    model_name: str,
    main_figure_folder: Path | str,
    supplementary_figure_folder: Path | str,
) -> List[str]:
    main_figure_folder = ensure_dir(main_figure_folder); supplementary_figure_folder = ensure_dir(supplementary_figure_folder)
    contrib = shap_result["contribution"]; obs = shap_result["observation_summary"]
    plot_shap_contribution_ci(contrib, main_figure_folder, model_name)
    ordered, X_bee, shap_bee = _ordered_beeswarm_arrays(obs, contrib, driver_features)

    # 标准横向条形图单图
    fig, ax = plt.subplots(figsize=(8.5, 5.8)); draw_shap_bar(ax, contrib, f"{model_name}–SHAP Feature Importance"); add_panel_label(ax, "(a)"); force_figure_times_new_roman(fig); fig.tight_layout(); save_figure_formats(fig, supplementary_figure_folder / f"{model_name}_SHAP_Bar_Single")

    # 蜂群单图
    fig = plt.figure(figsize=(10.0, 7.0)); ax = fig.add_axes([0.13, 0.13, 0.69, 0.78]); plt.sca(ax)
    shap.summary_plot(shap_bee, X_bee, show=False, plot_type="dot", max_display=len(ordered), sort=False, cmap=get_cmap_safely("Spectral_r"), plot_size=None)
    _relocate_shap_colorbar(fig, ax, [0.86, 0.13, 0.018, 0.78]); _style_beeswarm_axis(ax); add_panel_label(ax, "(a)"); force_figure_times_new_roman(fig); save_figure_formats(fig, main_figure_folder / f"{model_name}_OOF_SHAP_Beeswarm")

    # 条形 + 蜂群组合图
    fig = plt.figure(figsize=(18.5, 7.2)); axb = fig.add_axes([0.055, 0.14, 0.39, 0.73]); draw_shap_bar(axb, contrib, f"{model_name}–SHAP Feature Importance")
    axbee = fig.add_axes([0.565, 0.15, 0.305, 0.71]); plt.sca(axbee)
    shap.summary_plot(shap_bee, X_bee, show=False, plot_type="dot", max_display=len(ordered), sort=False, cmap=get_cmap_safely("Spectral_r"), plot_size=None)
    _relocate_shap_colorbar(fig, axbee, [0.92, 0.15, 0.012, 0.71]); _style_beeswarm_axis(axbee)
    fig.text(0.055, 0.95, "(a)", fontsize=16, fontweight="bold", fontfamily="Times New Roman"); fig.text(0.555, 0.95, "(b)", fontsize=16, fontweight="bold", fontfamily="Times New Roman")
    force_figure_times_new_roman(fig); save_figure_formats(fig, main_figure_folder / f"{model_name}_SHAP_Bar_Beeswarm_Combo", fixed_canvas=True)

    # 玫瑰单图 + 玫瑰蜂群组合图
    fig = plt.figure(figsize=(8.8, 8.8)); axr = fig.add_axes([0.025, 0.025, 0.95, 0.95], projection="polar"); draw_adaptive_rose(axr, contrib); add_panel_label(axr, "(a)", x=-0.01, y=1.01, fontsize=16); force_figure_times_new_roman(fig); save_figure_formats(fig, supplementary_figure_folder / f"{model_name}_SHAP_Rose_Single", fixed_canvas=True)
    fig = plt.figure(figsize=(19.5, 8.6)); axr = fig.add_axes([0.015, 0.025, 0.50, 0.95], projection="polar"); draw_adaptive_rose(axr, contrib)
    axbee = fig.add_axes([0.57, 0.15, 0.315, 0.71]); plt.sca(axbee); shap.summary_plot(shap_bee, X_bee, show=False, plot_type="dot", max_display=len(ordered), sort=False, cmap=get_cmap_safely("Spectral_r"), plot_size=None); _relocate_shap_colorbar(fig, axbee, [0.925, 0.15, 0.012, 0.71]); _style_beeswarm_axis(axbee)
    fig.text(0.02, 0.965, "(a)", fontsize=17, fontweight="bold", fontfamily="Times New Roman"); fig.text(0.555, 0.965, "(b)", fontsize=17, fontweight="bold", fontfamily="Times New Roman"); force_figure_times_new_roman(fig); save_figure_formats(fig, supplementary_figure_folder / f"{model_name}_SHAP_Rose_Beeswarm_Combo", fixed_canvas=True)

    # dependence：Top6，单图 + 2×3 组合图；单位位于横轴最右端。
    top_features = contrib.sort_values("contribution_percent", ascending=False)["feature"].tolist()[:ALE_TOP_N]
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.6)); axes = axes.reshape(-1)
    for i, feat in enumerate(top_features):
        ax = axes[i]; x = obs[feat].to_numpy(); y = obs[f"mean_signed_SHAP_{feat}"].to_numpy()
        ax.scatter(x, y, s=18, alpha=0.62, label="OOF-SHAP values"); add_binned_smooth_line(ax, x, y, 40); ax.axhline(0, color="black", linestyle="-.", linewidth=1.0, label="SHAP = 0")
        ax.set_title(display_name(feat), pad=8); set_feature_xaxis_label_with_unit(ax, feat, 12, 11, -0.105); ax.set_ylabel("SHAP value"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.legend(fontsize=8, frameon=True, loc="best"); add_panel_label(ax, panel_label(i)); force_times_new_roman(ax)
    for i in range(len(top_features), 6): fig.delaxes(axes[i])
    fig.tight_layout(); force_figure_times_new_roman(fig); save_figure_formats(fig, supplementary_figure_folder / f"{model_name}_SHAP_Dependence_Top6_Combo")

    for i, feat in enumerate(top_features):
        fig, ax = plt.subplots(figsize=(6.5, 5.0)); x = obs[feat].to_numpy(); y = obs[f"mean_signed_SHAP_{feat}"].to_numpy(); ax.scatter(x, y, s=18, alpha=0.62, label="OOF-SHAP values"); add_binned_smooth_line(ax, x, y, 40); ax.axhline(0, color="black", linestyle="-.", linewidth=1.0, label="SHAP = 0"); ax.set_title(display_name(feat), pad=8); set_feature_xaxis_label_with_unit(ax, feat, 12, 11, -0.105); ax.set_ylabel("SHAP value"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.legend(fontsize=8, frameon=True, loc="best"); add_panel_label(ax, panel_label(i)); force_figure_times_new_roman(fig); fig.tight_layout(); save_figure_formats(fig, supplementary_figure_folder / f"{model_name}_SHAP_Dependence_{feat}")

    return top_features


# =============================================================================
# 10. ALE：25 条外层曲线、经验不确定性带、转折稳定性
# =============================================================================
def calculate_ale_1d(
    fitted_model,
    X_eval: pd.DataFrame,
    feature: str,
    n_bins: int = ALE_N_BINS,
    q_range: Tuple[float, float] = ALE_Q_RANGE,
) -> pd.DataFrame:
    x = pd.to_numeric(X_eval[feature], errors="coerce").to_numpy(float)
    valid = np.isfinite(x)
    X = X_eval.loc[valid].copy(); x = x[valid]
    if len(x) < max(30, n_bins * 2):
        return pd.DataFrame(columns=["x", "ALE", "count", "lower_edge", "upper_edge"])
    qlo, qhi = np.quantile(x, q_range)
    inrange = (x >= qlo) & (x <= qhi)
    X = X.loc[inrange].copy(); x = x[inrange]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["x", "ALE", "count", "lower_edge", "upper_edge"])
    bins = np.searchsorted(edges, x, side="right") - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    local_effects, counts = [], []
    for k in range(len(edges) - 1):
        idx = np.where(bins == k)[0]
        counts.append(len(idx))
        if len(idx) == 0:
            local_effects.append(np.nan)
            continue
        Xk = X.iloc[idx].copy(); Xlow = Xk.copy(); Xhigh = Xk.copy()
        Xlow[feature] = edges[k]; Xhigh[feature] = edges[k+1]
        diff = fitted_model.predict(Xhigh) - fitted_model.predict(Xlow)
        local_effects.append(float(np.mean(diff)))
    local_effects = np.asarray(local_effects, float); counts = np.asarray(counts, int)
    # 对极少数空 bin 线性补齐局部差值。
    if np.isnan(local_effects).any():
        valid_idx = np.where(np.isfinite(local_effects))[0]
        if len(valid_idx) < 2:
            return pd.DataFrame(columns=["x", "ALE", "count", "lower_edge", "upper_edge"])
        local_effects = np.interp(np.arange(len(local_effects)), valid_idx, local_effects[valid_idx])
    cumulative = np.cumsum(local_effects)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weights = counts / counts.sum()
    centered = cumulative - np.sum(cumulative * weights)
    return pd.DataFrame({"x": centers, "ALE": centered, "count": counts, "lower_edge": edges[:-1], "upper_edge": edges[1:]})


def _smooth_curve(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    valid = np.isfinite(y)
    if valid.sum() < 5:
        return y
    yf = y.copy()
    yf[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), yf[valid])
    n = len(yf)
    window = min(11, n if n % 2 == 1 else n - 1)
    if window < 5:
        return yf
    if window % 2 == 0: window -= 1
    try:
        return savgol_filter(yf, window_length=window, polyorder=min(3, window-2), mode="interp")
    except Exception:
        return yf


def detect_ale_transitions(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y); x = x[valid]; y = y[valid]
    if len(x) < 7:
        return {"zero_x": np.nan, "zero_direction": "none", "slope_turn_x": np.nan, "slope_turn_type": "none"}
    ys = _smooth_curve(y)
    # 零交叉：选择局部斜率绝对值最大的交叉。
    zero_candidates = []
    for i in range(len(x)-1):
        if ys[i] == 0:
            zx = x[i]
        elif ys[i] * ys[i+1] < 0:
            zx = x[i] + (x[i+1]-x[i]) * (-ys[i]) / (ys[i+1]-ys[i])
        else:
            continue
        direction = "negative_to_positive" if ys[i] <= 0 < ys[i+1] else "positive_to_negative"
        slope = abs((ys[i+1]-ys[i]) / max(abs(x[i+1]-x[i]), 1e-12))
        zero_candidates.append((slope, zx, direction))
    if zero_candidates:
        _, zero_x, zero_dir = max(zero_candidates, key=lambda z: z[0])
    else:
        zero_x, zero_dir = np.nan, "none"

    dy = np.gradient(ys, x); ddy = np.gradient(dy, x)
    lo, hi = np.quantile(x, [0.10, 0.90])
    turn_candidates = []
    for i in range(1, len(x)-1):
        if not (lo <= x[i] <= hi): continue
        if dy[i-1] * dy[i+1] < 0:
            ttype = "local_max" if dy[i-1] > 0 and dy[i+1] < 0 else "local_min"
            turn_candidates.append((abs(ddy[i]), x[i], ttype))
    if turn_candidates:
        _, turn_x, turn_type = max(turn_candidates, key=lambda z: z[0])
    else:
        turn_x, turn_type = np.nan, "none"
    return {"zero_x": float(zero_x) if np.isfinite(zero_x) else np.nan, "zero_direction": zero_dir, "slope_turn_x": float(turn_x) if np.isfinite(turn_x) else np.nan, "slope_turn_type": turn_type}


def run_ale_25models(
    df: pd.DataFrame,
    features: Sequence[str],
    top_features: Sequence[str],
    selected_model: str,
    outer_tasks: Sequence[Dict[str, Any]],
    model_results: pd.DataFrame,
    output_folder: Path | str,
    logger: logging.Logger,
    progress_label: str = "ALE",
) -> Dict[str, pd.DataFrame]:
    output_folder = ensure_dir(output_folder); curve_folder = ensure_dir(output_folder / "fold_curves")
    X_all = df[list(features)].copy(); y_all = df[TARGET].copy()
    lookup = model_results[model_results["Model"] == selected_model].set_index(["Repeat", "Outer_Fold"])
    raw_files = []
    precompleted = 0
    for task in outer_tasks:
        r, f = int(task["repeat"]), int(task["fold"])
        outfile, done = (
            curve_folder / f"repeat_{r:02d}_fold_{f:02d}_ALE.csv",
            curve_folder / f"repeat_{r:02d}_fold_{f:02d}.done",
        )
        if done.exists() and outfile.exists():
            precompleted += 1
    progress = ConsoleProgress(progress_label, len(outer_tasks), completed_at_start=precompleted)
    progress.show(precompleted, "resuming checkpoint scan")
    completed_count = precompleted

    for task in outer_tasks:
        r, f, seed = int(task["repeat"]), int(task["fold"]), int(task["seed"])
        outfile = curve_folder / f"repeat_{r:02d}_fold_{f:02d}_ALE.csv"; done = curve_folder / f"repeat_{r:02d}_fold_{f:02d}.done"
        if done.exists() and outfile.exists():
            raw_files.append(outfile); logger.info(f"ALE 跳过已完成 R{r}F{f}"); continue
        row = lookup.loc[(r, f)]; params = json.loads(row["Best_Params_JSON"]) if isinstance(row["Best_Params_JSON"], str) else {}; n_est = row.get("XGB_Final_n_estimators", np.nan); n_est = None if pd.isna(n_est) else int(n_est)
        tr, te = task["train_idx"], task["test_idx"]
        model = fit_outer_model(selected_model, params, n_est, X_all.iloc[tr], y_all.iloc[tr], seed+r*100+f, features)
        fold_curves = []
        for feat in top_features:
            cur = calculate_ale_1d(model, X_all.iloc[te].copy(), feat, ALE_N_BINS, ALE_Q_RANGE)
            cur.insert(0, "feature", feat); cur.insert(0, "fold", f); cur.insert(0, "repeat", r)
            fold_curves.append(cur)
        pd.concat(fold_curves, ignore_index=True).to_csv(outfile, index=False, encoding="utf-8-sig"); raw_files.append(outfile); done.write_text(datetime.now().isoformat(), encoding="utf-8")
        completed_count += 1
        progress.show(completed_count, f"completed R{r}F{f}")
        del model; gc.collect()

    raw = pd.concat([pd.read_csv(p) for p in raw_files], ignore_index=True)
    raw.to_csv(output_folder / "ALE_curves_25models_raw.csv", index=False, encoding="utf-8-sig")

    interp_rows = []; threshold_rows = []
    for feat in top_features:
        clean = pd.to_numeric(df[feat], errors="coerce").dropna().to_numpy(float)
        qlo, qhi = np.quantile(clean, ALE_Q_RANGE); common_x = np.linspace(qlo, qhi, ALE_COMMON_GRID_N)
        for (r, f), g in raw[raw["feature"] == feat].groupby(["repeat", "fold"]):
            g = g.sort_values("x"); xi = g["x"].to_numpy(float); yi = g["ALE"].to_numpy(float)
            if len(xi) < 2: continue
            interp = np.interp(common_x, xi, yi, left=np.nan, right=np.nan)
            # np.interp 不支持 nan left/right in some versions as intended; 显式屏蔽范围外。
            interp[common_x < xi.min()] = np.nan; interp[common_x > xi.max()] = np.nan
            for xv, yv in zip(common_x, interp):
                interp_rows.append({"repeat": int(r), "fold": int(f), "feature": feat, "x": xv, "ALE": yv})
            trans = detect_ale_transitions(common_x, interp)
            threshold_rows.append({"repeat": int(r), "fold": int(f), "feature": feat, **trans})

    interp_df = pd.DataFrame(interp_rows); interp_df.to_csv(output_folder / "ALE_curves_25models_common_grid.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for (feat, xval), g in interp_df.groupby(["feature", "x"]):
        vals = g["ALE"].dropna().to_numpy(float)
        if len(vals):
            summary_rows.append({"feature": feat, "x": xval, "mean_ALE": float(np.mean(vals)), "q2.5": float(np.quantile(vals, 0.025)), "q97.5": float(np.quantile(vals, 0.975)), "n_curves": len(vals)})
    summary = pd.DataFrame(summary_rows); summary.to_csv(output_folder / "ALE_summary_95_empirical_interval.csv", index=False, encoding="utf-8-sig")

    threshold = pd.DataFrame(threshold_rows); threshold.to_csv(output_folder / "ALE_thresholds_each_outer_model.csv", index=False, encoding="utf-8-sig")
    stability_rows = []
    for feat, g in threshold.groupby("feature"):
        n = len(g)
        for kind, value_col, dir_col in [("zero_crossing", "zero_x", "zero_direction"), ("slope_turn", "slope_turn_x", "slope_turn_type")]:
            valid = g[np.isfinite(g[value_col]) & (g[dir_col] != "none")]
            if len(valid):
                dominant = valid[dir_col].value_counts().index[0]
                same = valid[valid[dir_col] == dominant]
                prop = len(same) / n
                vals = same[value_col].to_numpy(float)
                stability_rows.append({
                    "feature": feat, "transition_type": kind, "dominant_direction_or_type": dominant,
                    "n_outer_models": n, "n_same_direction_detected": len(same), "stable_proportion": prop,
                    "position_median": float(np.median(vals)), "position_Q25": float(np.quantile(vals, 0.25)), "position_Q75": float(np.quantile(vals, 0.75)),
                    "position_Q2.5": float(np.quantile(vals, 0.025)), "position_Q97.5": float(np.quantile(vals, 0.975)),
                    "interpretation_label": "stable transition interval" if prop >= ALE_TURN_MIN_PROPORTION else "potential transition interval",
                    "predefined_min_stable_proportion": ALE_TURN_MIN_PROPORTION,
                })
            else:
                stability_rows.append({"feature": feat, "transition_type": kind, "dominant_direction_or_type": "none", "n_outer_models": n, "n_same_direction_detected": 0, "stable_proportion": 0.0, "position_median": np.nan, "position_Q25": np.nan, "position_Q75": np.nan, "position_Q2.5": np.nan, "position_Q97.5": np.nan, "interpretation_label": "no consistent transition detected", "predefined_min_stable_proportion": ALE_TURN_MIN_PROPORTION})
    stability = pd.DataFrame(stability_rows); stability.to_csv(output_folder / "ALE_threshold_stability.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_folder / "ALE_summary_tables.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="ALE_summary", index=False); threshold.to_excel(writer, sheet_name="Each_model_threshold", index=False); stability.to_excel(writer, sheet_name="Threshold_stability", index=False)
    return {"raw": raw, "common": interp_df, "summary": summary, "threshold_each": threshold, "threshold_stability": stability}


def plot_ale_outputs(ale_result: Dict[str, pd.DataFrame], top_features: Sequence[str], model_name: str, main_figure_folder: Path | str, supplementary_figure_folder: Path | str, prefix: str = "") -> None:
    summary = ale_result["summary"]; main_figure_folder = ensure_dir(main_figure_folder); supplementary_figure_folder = ensure_dir(supplementary_figure_folder)
    # 2×3 组图（Top-6）
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.6)); axes = axes.reshape(-1)
    for i, feat in enumerate(top_features):
        ax = axes[i]; g = summary[summary["feature"] == feat].sort_values("x")
        ax.fill_between(g["x"].to_numpy(), g["q2.5"].to_numpy(), g["q97.5"].to_numpy(), alpha=0.25, linewidth=0, label="95% empirical uncertainty interval")
        ax.plot(g["x"], g["mean_ALE"], linewidth=2.1, label="Mean ALE"); ax.axhline(0, color="black", linestyle="-.", linewidth=0.9)
        ax.set_title(display_name(feat), pad=8); set_feature_xaxis_label_with_unit(ax, feat, 12, 11, -0.105); ax.set_ylabel("ALE effect"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.6); add_panel_label(ax, panel_label(i)); force_times_new_roman(ax)
    for i in range(len(top_features), 6): fig.delaxes(axes[i])
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); force_figure_times_new_roman(fig); save_figure_formats(fig, main_figure_folder / f"{prefix}{model_name}_ALE_Top6_Combo")

    for i, feat in enumerate(top_features):
        g = summary[summary["feature"] == feat].sort_values("x"); fig, ax = plt.subplots(figsize=(6.5, 5.0)); ax.fill_between(g["x"].to_numpy(), g["q2.5"].to_numpy(), g["q97.5"].to_numpy(), alpha=0.25, linewidth=0, label="95% empirical uncertainty interval"); ax.plot(g["x"], g["mean_ALE"], linewidth=2.1, label="Mean ALE"); ax.axhline(0, color="black", linestyle="-.", linewidth=0.9); ax.set_title(display_name(feat), pad=8); set_feature_xaxis_label_with_unit(ax, feat, 12, 11, -0.105); ax.set_ylabel("ALE effect"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.6); ax.legend(frameon=False, fontsize=9); add_panel_label(ax, panel_label(i)); force_figure_times_new_roman(fig); fig.tight_layout(); save_figure_formats(fig, supplementary_figure_folder / f"{prefix}{model_name}_ALE_{feat}_Single")


# =============================================================================
# 11. 敏感性比较辅助函数
# =============================================================================
def compare_main_vs_sensitivity_performance(
    main_results: pd.DataFrame,
    sensitivity_results: pd.DataFrame,
    selected_model: str,
    sensitivity_name: str,
) -> pd.DataFrame:
    main = main_results[main_results["Model"] == selected_model].copy()
    sens = sensitivity_results[sensitivity_results["Model"] == selected_model].copy()
    rows = []
    for metric in ["Test_R2", "Test_RMSE", "Test_MAE", "Test_MSE_x1e3"]:
        rows.append({
            "Sensitivity": sensitivity_name,
            "Metric": metric,
            "Main_mean": float(main[metric].mean()),
            "Main_SD": float(main[metric].std(ddof=1)),
            "Sensitivity_mean": float(sens[metric].mean()),
            "Sensitivity_SD": float(sens[metric].std(ddof=1)),
            "Sensitivity_minus_Main": float(sens[metric].mean() - main[metric].mean()),
            "Main_minus_Sensitivity": float(main[metric].mean() - sens[metric].mean()),
        })
    return pd.DataFrame(rows)


def compare_shap_contributions(
    main_contrib: pd.DataFrame,
    sens_contrib: pd.DataFrame,
    common_features: Sequence[str],
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = main_contrib.set_index("feature"); b = sens_contrib.set_index("feature")
    rows = []
    for feat in common_features:
        if feat in a.index and feat in b.index:
            rows.append({
                "feature": feat,
                "main_contribution_percent": float(a.loc[feat, "contribution_percent"]),
                "sensitivity_contribution_percent": float(b.loc[feat, "contribution_percent"]),
                "difference_sensitivity_minus_main": float(b.loc[feat, "contribution_percent"] - a.loc[feat, "contribution_percent"]),
            })
    detail = pd.DataFrame(rows)
    main_ranks = a.loc[[f for f in common_features if f in a.index], "contribution_percent"].rank(ascending=False)
    sens_ranks = b.loc[[f for f in common_features if f in b.index], "contribution_percent"].rank(ascending=False)
    common_idx = main_ranks.index.intersection(sens_ranks.index)
    rho, p = stats.spearmanr(main_ranks.loc[common_idx], sens_ranks.loc[common_idx]) if len(common_idx) >= 3 else (np.nan, np.nan)
    # Top-4 overlap 只在 common_features 范围内计算；对温度重参数化尤其重要，
    # 避免把 TMX/TMN 与 Tmean/DTR 名称差异误判为非温度变量排序变化。
    common_available = [f for f in common_features if f in a.index and f in b.index]
    top_n = min(4, len(common_available))
    top4_main = set(a.loc[common_available].sort_values("contribution_percent", ascending=False).index[:top_n])
    top4_sens = set(b.loc[common_available].sort_values("contribution_percent", ascending=False).index[:top_n])
    summary = pd.DataFrame([{
        "comparison": label,
        "Spearman_rank_rho": float(rho) if np.isfinite(rho) else np.nan,
        "Spearman_p": float(p) if np.isfinite(p) else np.nan,
        "Top4_overlap_count": len(top4_main & top4_sens),
        "Top4_overlap_rate": len(top4_main & top4_sens) / float(top_n) if top_n > 0 else np.nan,
        "Main_Top4": ", ".join(top4_main),
        "Sensitivity_Top4": ", ".join(top4_sens),
    }])
    return detail, summary


def plot_performance_sensitivity_comparison(perf_table: pd.DataFrame, figure_folder: Path | str, title: str, file_stem: str) -> None:
    # 仅画 R² / RMSE / MAE，避免 MSE×10^-3 与其它量纲混在一张图。
    d = perf_table[perf_table["Metric"].isin(["Test_R2", "Test_RMSE", "Test_MAE"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    for i, (_, row) in enumerate(d.iterrows()):
        ax = axes[i]; vals = [row["Main_mean"], row["Sensitivity_mean"]]; errs = [row["Main_SD"], row["Sensitivity_SD"]]
        ax.bar([0,1], vals, yerr=errs, capsize=4, edgecolor="black", linewidth=0.7, width=0.62)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Main", "Sensitivity"], rotation=10); ax.set_title(str(row["Metric"]).replace("Test_", "")); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); add_panel_label(ax, panel_label(i)); force_times_new_roman(ax)
    fig.suptitle(title, y=1.02, fontsize=14, fontfamily="Times New Roman"); fig.tight_layout(); save_figure_formats(fig, Path(figure_folder) / file_stem)


def plot_shap_contribution_comparison(main_contrib: pd.DataFrame, sens_contrib: pd.DataFrame, common_features: Sequence[str], figure_folder: Path | str, title: str, file_stem: str) -> None:
    ma = main_contrib.set_index("feature"); se = sens_contrib.set_index("feature")
    feats = [f for f in common_features if f in ma.index and f in se.index]
    x = np.arange(len(feats)); width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 5.4)); ax.bar(x-width/2, [ma.loc[f,"contribution_percent"] for f in feats], width, label="Main", edgecolor="black", linewidth=0.5); ax.bar(x+width/2, [se.loc[f,"contribution_percent"] for f in feats], width, label="Sensitivity", edgecolor="black", linewidth=0.5, alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels([display_name(f) for f in feats]); ax.set_ylabel("Relative contribution (%)"); ax.set_title(title); ax.legend(frameon=False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); add_panel_label(ax, "(a)"); force_figure_times_new_roman(fig); fig.tight_layout(); save_figure_formats(fig, Path(figure_folder) / file_stem)


def plot_ale_main_vs_sensitivity(main_summary: pd.DataFrame, sens_summary: pd.DataFrame, features: Sequence[str], figure_folder: Path | str, title_prefix: str, file_stem: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.6)); axes = axes.reshape(-1)
    for i, feat in enumerate(features[:ALE_TOP_N]):
        ax = axes[i]; a = main_summary[main_summary["feature"] == feat].sort_values("x"); b = sens_summary[sens_summary["feature"] == feat].sort_values("x")
        if len(a): ax.plot(a["x"], a["mean_ALE"], linewidth=2.0, label="Main")
        if len(b): ax.plot(b["x"], b["mean_ALE"], linewidth=2.0, linestyle="--", label="Sensitivity")
        ax.axhline(0, color="black", linestyle="-.", linewidth=0.8); ax.set_title(display_name(feat)); set_feature_xaxis_label_with_unit(ax, feat, 12, 11, -0.105); ax.set_ylabel("ALE effect"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); add_panel_label(ax, panel_label(i)); force_times_new_roman(ax)
    for i in range(len(features[:ALE_TOP_N]), 6): fig.delaxes(axes[i])
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False); fig.suptitle(title_prefix, y=0.99, fontsize=14, fontfamily="Times New Roman"); fig.tight_layout(rect=[0,0,1,0.96]); save_figure_formats(fig, Path(figure_folder) / file_stem)


# =============================================================================
# 12. 输出清单
# =============================================================================
def save_output_manifest(run_dir: Path | str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    rows = []
    for p in run_dir.rglob("*"):
        if p.is_file():
            rows.append({"relative_path": str(p.relative_to(run_dir)), "file_name": p.name, "size_kb": p.stat().st_size / 1024.0})
    df = pd.DataFrame(rows).sort_values("relative_path") if rows else pd.DataFrame(columns=["relative_path","file_name","size_kb"])
    df.to_csv(run_dir / "output_manifest.csv", index=False, encoding="utf-8-sig")
    return df
