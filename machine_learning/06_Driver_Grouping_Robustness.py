# -*- coding: utf-8 -*-
"""
05_Reviewer2_Comment16_Grouped_Importance.py

Diversity Reviewer 2 / Comment 16 (original item 3.12)
川西北草地 RI：温度–能量组与水分组的重要性及分组稳健性检验

本脚本独立运行，不修改 01/02/03/04 的程序和任何原始输出。
使用已完成 01 主分析的相同输入 CSV、相同 25 套外层分组测试折、相同最终模型、
相同外层最优超参数和相同模型随机种子。不会重新执行五模型竞争或 Optuna。

执行内容：
 A. 现有 OOF-SHAP 7 个生态因子的分组汇总、patch_id1 cluster bootstrap 95% CI；
 B. 组级置换重要性（GPFI）：每次联合置换整个变量组，并按格网交换完整年度轨迹；
 C. 分组敏感性：VPD 单独分组、VPD 划入温度组、SR 单独分组；
 D. 真正的组级 Shapley 值：将整组变量视为一个博弈参与者，
    在分层抽样的外层测试数据上计算训练背景下的 interventional group Shapley；
 E. 读取已完成 No-Year 的 OOF-SHAP（如存在且确实对应同一主分析）作辅助比较；
 F. SR/PRE/VPD 等相关性核对、CSV/XLSX、600 dpi 图件、方法说明和复现配置。

请将本文件与正在使用的 ml_common.py 放在同一个目录内。
需要先运行完 01_Main_ML_NestedCV_OOFSHAP_ALE.py；02 可选，不需要重跑 01。

概念边界：
 1. 现有 SHAP 绝对值求和不等于严格的组 Shapley，分别报告，绝不混用名称。
 2. GPFI 的单位是外层测试 R² 下降量，不是 SHAP 百分比，不能直接相加成贡献率。
 3. 分组检验是模型预测依赖/归因，而不是生态因果效应。
 4. 按格网联合置换会保留组内相关性、同一年份和格网的完整时间序列；
    但不能消除不同组之间的相关性，仍可能构造不常见的跨组组合。
 5. Year 始终参与主模型。在 GPFI 中固定不变，在组 Shapley 中作为独立组参与。
 6. 真正组 Shapley 使用外测格网子样本和外训背景的 Monte Carlo 期望估计；
    样本量写入输出，不能将它冒充全量 24,084 样本的精确结果。
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_common import (
    TARGET, GROUP_COLUMN, YEAR_COLUMN, SAMPLE_ID_COLUMN,
    DRIVER_FEATURES, MAIN_MODEL_FEATURES,
    load_model_data, load_outer_tasks_from_assignment,
    validate_outer_tasks, fit_outer_model,
)

# =============================================================================
# 1. Portable paths and run options
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = Path(os.environ.get(
    "NW_C16_DATA_CSV", REPO_ROOT / "data" / "modeling_data.csv"
)).expanduser()
OUTPUT_ROOT = Path(os.environ.get(
    "NW_C16_OUTPUT_ROOT", REPO_ROOT / "output"
)).expanduser()

# 正常情况下自动读取 OUTPUT_ROOT/latest_main_run.txt。
# 如果没有这个指针，会自动搜索该目录下已完成的 Final_ML_Main_* 文件夹。
# 仅在确实需要手动指定其他主分析版本时填写完整目录；默认保持 None。
MAIN_RUN_OVERRIDE = os.environ.get("NW_C16_MAIN_RUN_DIR")

# 每个 25 外层测试折都做重复联合置换。数值可提高到 30 或 50，耗时也会增加。
GPFI_PERMUTATIONS_PER_FOLD = 20

# 真正组级 Shapley 以格网为单位取样，每个外测折最多抽取若干格网，各取 1 年；
# 背景来自相应外层训练折，确保不使用测试 RI 进行拟合。
RUN_TRUE_GROUP_SHAPLEY = True
GROUP_SHAP_TEST_GRIDS_PER_FOLD = 80
GROUP_SHAP_BACKGROUND_GRIDS_PER_FOLD = 32
GROUP_SHAP_SCENARIOS = (
    "Baseline", "VPD_Separate", "VPD_to_Temperature", "SR_Separate"
)

# 1000 次格网级重采样，只针对已经计算出的 OOF-SHAP / 组 Shapley；
# GPFI 汇总使用 25 套外测折的分布，不将其冒充相互独立的样本 CI。
CLUSTER_BOOTSTRAP_B = 1000

# 防止意外引用不同版本的输入数据或主分析。
VERIFY_MAIN_INPUT_HASH = True
MODEL_R2_MATCH_TOLERANCE = 0.003
MODEL_INTERNAL_N_JOBS = 4
RANDOM_SEED = 20261025
FIGURE_DPI = 600

# Optional overrides for local data/output locations and smaller diagnostic runs.
if os.environ.get("NW_C16_GPFI_PERM"):
    GPFI_PERMUTATIONS_PER_FOLD = int(os.environ["NW_C16_GPFI_PERM"])
if os.environ.get("NW_C16_SHAP_EVAL"):
    GROUP_SHAP_TEST_GRIDS_PER_FOLD = int(os.environ["NW_C16_SHAP_EVAL"])
if os.environ.get("NW_C16_SHAP_BG"):
    GROUP_SHAP_BACKGROUND_GRIDS_PER_FOLD = int(os.environ["NW_C16_SHAP_BG"])
if os.environ.get("NW_C16_FIG_DPI"):
    FIGURE_DPI = int(os.environ["NW_C16_FIG_DPI"])

# 原始理论分组 + 三种敏感性方案。
# SR 单独分组并非认定 SR 属于水分过程，而是为了显示其对类别解释的影响。
GROUP_SCENARIOS = OrderedDict({
    "Baseline": OrderedDict({
        "Temperature-energy": ("TMX", "TMN", "SR"),
        "Moisture": ("SM", "PRE", "VPD"),
        "Grazing": ("GI",),
    }),
    "VPD_Separate": OrderedDict({
        "Temperature-energy": ("TMX", "TMN", "SR"),
        "Moisture-supply": ("SM", "PRE"),
        "Atmospheric-dryness": ("VPD",),
        "Grazing": ("GI",),
    }),
    "VPD_to_Temperature": OrderedDict({
        "Temperature-energy-plus-VPD": ("TMX", "TMN", "SR", "VPD"),
        "Moisture-supply": ("SM", "PRE"),
        "Grazing": ("GI",),
    }),
    "SR_Separate": OrderedDict({
        "Temperature-only": ("TMX", "TMN"),
        "Solar-radiation": ("SR",),
        "Moisture": ("SM", "PRE", "VPD"),
        "Grazing": ("GI",),
    }),
})

# 用来比较两组的配对折内 ΔR²。不对单独的 SR 与 TMX/TMN 重要性作加法。
GPFI_PAIRS = OrderedDict({
    "Baseline": ("Temperature-energy", "Moisture"),
    "VPD_Separate": ("Temperature-energy", "Moisture-supply"),
    "VPD_to_Temperature": ("Temperature-energy-plus-VPD", "Moisture-supply"),
    "SR_Separate": ("Temperature-only", "Moisture"),
})

RUN_PREFIX = "Reviewer2_Comment16_GroupImportance"
ACTIVE_POINTER = "active_reviewer2_comment16.txt"
LATEST_POINTER = "latest_reviewer2_comment16.txt"


# =============================================================================
# 2. 基础、输出及来源校验
# =============================================================================
def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def serializer(obj: Any):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"不可 JSON 序列化的类型：{type(obj)}")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=serializer)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"缺少{name}：{path}")
    return path


def validate_scenarios() -> None:
    expected = set(DRIVER_FEATURES)
    if len(DRIVER_FEATURES) != 7 or len(MAIN_MODEL_FEATURES) != 8:
        raise RuntimeError("ml_common.py 变量体系不是预期的 7 生态因子 + Year，禁止静默运行。")
    for scenario, groups in GROUP_SCENARIOS.items():
        features = [f for group in groups.values() for f in group]
        if set(features) != expected or len(features) != len(set(features)):
            raise RuntimeError(f"分组方案 {scenario} 未恰好覆盖全部 7 个生态因子。")
    if not set(GROUP_SHAP_SCENARIOS).issubset(set(GROUP_SCENARIOS)):
        raise RuntimeError("GROUP_SHAP_SCENARIOS 有未定义的方案。")


def resolve_main_run() -> Path:
    if MAIN_RUN_OVERRIDE:
        path = Path(MAIN_RUN_OVERRIDE)
    else:
        pointer = OUTPUT_ROOT / "latest_main_run.txt"
        if pointer.is_file():
            path = Path(pointer.read_text(encoding="utf-8-sig").strip())
        else:
            candidates = [p for p in OUTPUT_ROOT.glob("Final_ML_Main_*")
                          if p.is_dir() and (p / "RUN_COMPLETE.flag").is_file()]
            if not candidates:
                raise FileNotFoundError(
                    "未找到已完成的 01 主分析。请确认 latest_main_run.txt 或 Final_ML_Main_* 目录存在。"
                )
            path = sorted(candidates, key=lambda p: p.name)[-1]
    if not path.is_dir() or not (path / "RUN_COMPLETE.flag").is_file():
        raise RuntimeError(f"01 主分析目录不存在或没有 RUN_COMPLETE.flag：{path}")
    require_file(path / "selected_model_for_sensitivity.json", "模型选择 JSON")
    require_file(path / "02_main_nested_cv" / "all_outer_scores.csv", "外层分组评分")
    require_file(path / "01_split_manifest" / "outer_split_assignment_by_observation.csv", "原外层划分")
    return path


def resolve_noyear_shap(main_run: Path) -> Path | None:
    pointer = OUTPUT_ROOT / "latest_sensitivity_run.txt"
    candidates = []
    if pointer.is_file():
        candidates.append(Path(pointer.read_text(encoding="utf-8-sig").strip()))
    candidates += sorted(OUTPUT_ROOT.glob("Final_ML_Sensitivity_*"), reverse=True)
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        link = candidate / "linked_main_run.txt"
        shap = candidate / "01_noYear" / "oof_shap" / "SHAP_contribution_bootstrap_CI_and_stability.csv"
        if not link.exists() or not shap.exists():
            continue
        if Path(link.read_text(encoding="utf-8-sig").strip()).resolve() == main_run.resolve():
            return shap
    return None


def create_or_resume_output(main_run: Path, data_hash: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pointer = OUTPUT_ROOT / ACTIVE_POINTER
    if pointer.exists():
        candidate = Path(pointer.read_text(encoding="utf-8-sig").strip())
        manifest = candidate / "00_config" / "run_identity.json"
        if candidate.is_dir() and manifest.is_file() and not (candidate / "RUN_COMPLETE.flag").exists():
            recorded = read_json(manifest)
            if (recorded.get("main_run") == str(main_run)
                    and recorded.get("input_sha256") == data_hash
                    and recorded.get("script_sha256") == sha256_file(Path(__file__))):
                log(f"断点续跑已有补充分析：{candidate}")
                return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = OUTPUT_ROOT / f"{RUN_PREFIX}_{stamp}"
    serial = 0
    while candidate.exists():
        serial += 1
        candidate = OUTPUT_ROOT / f"{RUN_PREFIX}_{stamp}_{serial:02d}"
    candidate.mkdir(parents=True)
    (candidate / "00_config").mkdir()
    write_json({
        "main_run": str(main_run),
        "input_sha256": data_hash,
        "script_sha256": sha256_file(Path(__file__)),
        "created": datetime.now().isoformat(),
    }, candidate / "00_config" / "run_identity.json")
    pointer.write_text(str(candidate), encoding="utf-8")
    return candidate


def verify_main_dataset_hash(main_run: Path, data_hash: str) -> str:
    path = main_run / "00_config_environment" / "file_hashes.csv"
    if not path.exists():
        if VERIFY_MAIN_INPUT_HASH:
            raise RuntimeError("01 未找到 file_hashes.csv，无法证实当前 CSV 与主分析完全一致。")
        return "NOT_AVAILABLE"
    records = pd.read_csv(path)
    matched = records[records["label"].astype(str) == "input_csv"]
    if matched.empty:
        raise RuntimeError("01 file_hashes.csv 未包含 input_csv 哈希。")
    original_hash = str(matched.iloc[0]["sha256"])
    if original_hash != data_hash and VERIFY_MAIN_INPUT_HASH:
        raise RuntimeError(
            "当前建模 CSV 与 01 主分析使用的文件 SHA256 不一致。"
            "请核对是否替换了数据文件，不要混用模型参数和新数据。\n"
            f"01 SHA256: {original_hash}\n当前 SHA256: {data_hash}"
        )
    return original_hash


def configure_figures() -> None:
    # Windows 通常自带 Times New Roman；其他环境采用 serif fallback，避免大量找字体警告。
    from matplotlib import font_manager
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in installed_fonts else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": font,
        "font.serif": [font],
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.85,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: Any, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base) + ".png", dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(str(base) + ".tif", dpi=FIGURE_DPI, format="tiff",
                    pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight", facecolor="white")
    except Exception:
        fig.savefig(str(base) + ".tif", dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(str(base) + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# 3. 旧 OOF-SHAP 的真实分组汇总及格网级 bootstrap
# =============================================================================
def calculate_relative_shap_groups(importance: dict, scenario: str) -> pd.DataFrame:
    groups = GROUP_SCENARIOS[scenario]
    denominator = sum(float(importance[f]) for f in DRIVER_FEATURES)
    if denominator <= 0:
        raise RuntimeError("已有 OOF-SHAP 生态因子重要性全部为 0。")
    rows = []
    for label, features in groups.items():
        value = sum(float(importance[f]) for f in features)
        rows.append({
            "scenario": scenario, "group": label, "features": "+".join(features),
            "n_features": len(features), "sum_individual_mean_abs_SHAP": value,
            "ecological_relative_percent": value / denominator * 100.0,
            "percent_per_feature_descriptive_only": value / denominator * 100.0 / len(features),
            "type": "Sum of individual mean absolute OOF-SHAP; NOT true group Shapley",
        })
    return pd.DataFrame(rows)


def bootstrap_original_shap(obs: pd.DataFrame, folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    abs_columns = [f"mean_abs_SHAP_{f}" for f in DRIVER_FEATURES]
    required = [GROUP_COLUMN, *abs_columns, f"mean_abs_SHAP_{YEAR_COLUMN}"]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(f"01 的 OOF-SHAP 观测级输出缺少字段：{missing}")

    importance = {f: float(obs[f"mean_abs_SHAP_{f}"].mean()) for f in DRIVER_FEATURES}
    all_importance = dict(importance)
    all_importance[YEAR_COLUMN] = float(obs[f"mean_abs_SHAP_{YEAR_COLUMN}"].mean())
    full_denom = sum(all_importance.values())
    year_share = all_importance[YEAR_COLUMN] / full_denom * 100.0

    original_rows = []
    for scenario in GROUP_SCENARIOS:
        original_rows.append(calculate_relative_shap_groups(importance, scenario))
    original = pd.concat(original_rows, ignore_index=True)
    write_csv(original, folder / "SHAP_Original_Grouped_AllScenarios.csv")

    gsums = obs.groupby(GROUP_COLUMN)[abs_columns].sum().to_numpy(dtype=float)
    gcounts = obs.groupby(GROUP_COLUMN).size().to_numpy(dtype=float)
    if len(gsums) < 5:
        raise ValueError("格网数过少，无法进行可靠的格网级 bootstrap。")
    rng = np.random.default_rng(RANDOM_SEED + 301)
    ci_records = []
    total_groups = len(gsums)
    for b in range(CLUSTER_BOOTSTRAP_B):
        group_draws = rng.integers(0, total_groups, size=total_groups)
        bmean = gsums[group_draws].sum(axis=0) / gcounts[group_draws].sum()
        bmap = dict(zip(DRIVER_FEATURES, bmean))
        for scenario in GROUP_SCENARIOS:
            gr = calculate_relative_shap_groups(bmap, scenario)
            for row in gr.to_dict("records"):
                ci_records.append((scenario, row["group"], b, row["ecological_relative_percent"]))
    boot = pd.DataFrame(ci_records, columns=["scenario", "group", "bootstrap_id", "percent"])
    ci = boot.groupby(["scenario", "group"], as_index=False).agg(
        CI_lower_2_5=("percent", lambda s: float(s.quantile(0.025))),
        CI_upper_97_5=("percent", lambda s: float(s.quantile(0.975))),
        bootstrap_mean=("percent", "mean"),
        bootstrap_SD=("percent", "std"),
    )
    original = original.merge(ci, on=["scenario", "group"], how="left")
    write_csv(original, folder / "SHAP_Original_Grouped_with_GridBootstrapCI.csv")
    write_csv(ci, folder / "SHAP_Grouped_GridBootstrapCI.csv")
    contrast_records = []
    pivot = boot.pivot_table(index=["scenario", "bootstrap_id"], columns="group", values="percent")
    point = original.set_index(["scenario", "group"])["ecological_relative_percent"]
    for scenario, (left, right) in GPFI_PAIRS.items():
        diff = pivot.loc[scenario, left] - pivot.loc[scenario, right]
        contrast_records.append({
            "scenario": scenario, "comparison": f"{left} minus {right}",
            "SHAP_difference_percentage_points": float(point.loc[scenario, left] - point.loc[scenario, right]),
            "bootstrap_CI_lower_2_5": float(diff.quantile(0.025)),
            "bootstrap_CI_upper_97_5": float(diff.quantile(0.975)),
            "bootstrap_B": CLUSTER_BOOTSTRAP_B,
            "note": "Arithmetic sum of individual OOF-SHAP; not true group Shapley",
        })
    write_csv(pd.DataFrame(contrast_records), folder / "SHAP_GroupDominance_GridBootstrap.csv")
    meta = pd.DataFrame([{
        "year_share_8_predictors_percent": year_share,
        "ecological_share_8_predictors_percent": 100.0 - year_share,
        "normalization_for_group_comparison": "7 ecological predictors only",
        "bootstrap_B": CLUSTER_BOOTSTRAP_B,
        "cluster_unit": GROUP_COLUMN,
        "n_unique_grids": int(obs[GROUP_COLUMN].nunique()),
        "n_unique_observations": len(obs),
    }])
    write_csv(meta, folder / "Main_Year_Share_and_Normalization.csv")
    return original, meta


def validate_shap_original(main_run: Path, obs: pd.DataFrame) -> None:
    contrib_path = main_run / "04_oof_shap" / "SHAP_contribution_bootstrap_CI_and_stability.csv"
    saved = pd.read_csv(require_file(contrib_path, "01 SHAP 原始贡献率"))
    recompute = {f: float(obs[f"mean_abs_SHAP_{f}"].mean()) for f in DRIVER_FEATURES}
    denom = sum(recompute.values())
    for _, row in saved.iterrows():
        f = str(row["feature"])
        if f in recompute:
            observed = 100.0 * recompute[f] / denom
            if abs(observed - float(row["contribution_percent"])) > 0.05:
                raise RuntimeError(f"重新汇总的 OOF-SHAP 与 01 文件不一致：{f}。")


def noyear_shap_comparison(main_run: Path, main_abs: pd.DataFrame, folder: Path) -> pd.DataFrame:
    path = resolve_noyear_shap(main_run)
    if path is None:
        log("未找到关联同一 01 主分析的 No-Year OOF-SHAP，跳过附加对比。")
        return pd.DataFrame()
    table = pd.read_csv(path)
    if not set(DRIVER_FEATURES).issubset(set(table["feature"].astype(str))):
        log("No-Year SHAP 文件变量字段不完整，跳过附加对比。")
        return pd.DataFrame()
    src = table.set_index("feature")
    importance = {f: float(src.loc[f, "contribution_percent"]) for f in DRIVER_FEATURES}
    result = pd.concat([calculate_relative_shap_groups(importance, s) for s in GROUP_SCENARIOS],
                       ignore_index=True)
    result = result.rename(columns={"ecological_relative_percent": "NoYear_relative_percent"})
    comparison = main_abs[["scenario", "group", "ecological_relative_percent"]].merge(
        result[["scenario", "group", "NoYear_relative_percent"]],
        on=["scenario", "group"], how="inner"
    )
    comparison["NoYear_minus_Main_percentage_points"] = (
        comparison["NoYear_relative_percent"] - comparison["ecological_relative_percent"]
    )
    write_csv(comparison, folder / "Main_vs_NoYear_SHAP_Grouped.csv")
    return comparison


# =============================================================================
# 4. 25 套外层模型重建、折内数据变换和按格网全时间序列联合置换
# =============================================================================
def prepare_core_model(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame
                       ) -> tuple[Any, np.ndarray, np.ndarray]:
    """一次折内处理，避免 20 次置换时反复 fit/transform imputer。"""
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        if "imputer" not in steps or "model" not in steps:
            raise RuntimeError("无法从 sklearn Pipeline 提取 imputer/model。")
        Ztr = steps["imputer"].transform(X_train)
        Zte = steps["imputer"].transform(X_test)
        if "scaler" in steps:
            Ztr = steps["scaler"].transform(Ztr)
            Zte = steps["scaler"].transform(Zte)
        return steps["model"], np.asarray(Ztr, dtype=float), np.asarray(Zte, dtype=float)
    if hasattr(model, "model") and hasattr(model, "transform"):
        return (model.model, np.asarray(model.transform(X_train), dtype=float),
                np.asarray(model.transform(X_test), dtype=float))
    raise RuntimeError("无法识别 ml_common.fit_outer_model 返回的模型结构。")


def make_grid_trajectory_permutation(grids: np.ndarray, years: np.ndarray,
                                     rng: np.random.Generator) -> np.ndarray:
    """
    完整格网 A 的 1985–2020 全部年份，由随机格网 B 同年份替换。
    donor grid 在一次 permutation 中对全部年份和全部组内变量均保持一致。
    与普通逐行打乱相比，保留组内共变关系和格网自身的年度轨迹。
    """
    gids = np.asarray(grids).astype(str)
    yrs = np.asarray(years)
    keys = pd.MultiIndex.from_arrays([gids, yrs], names=["grid", "year"])
    if keys.has_duplicates:
        raise RuntimeError("测试折存在重复的格网–年份记录。")
    unique = np.unique(gids)
    if len(unique) <= 1:
        raise RuntimeError("测试折只有一个格网，无法做格网联合置换。")
    # Avoid identity donors, which would weaken measured importance through unchanged grids.
    permutation = rng.permutation(len(unique))
    for _ in range(50):
        if not np.any(permutation == np.arange(len(unique))):
            break
        permutation = rng.permutation(len(unique))
    else:
        permutation = np.roll(np.arange(len(unique)), 1)
    shuffled = unique[permutation]
    donors = dict(zip(unique, shuffled))
    donor_grid = pd.Series(gids).map(donors).to_numpy()
    donor_keys = pd.MultiIndex.from_arrays([donor_grid, yrs], names=["grid", "year"])
    positions = keys.get_indexer(donor_keys)
    if np.any(positions < 0):
        raise RuntimeError(
            "各格网的年份覆盖不一致，无法安全交换完整年度轨迹。"
            "不采用可能改变方法含义的静默逐行回退。"
        )
    return positions.astype(int)


def unique_variable_groups() -> dict[tuple[str, ...], list[tuple[str, str]]]:
    result: dict[tuple[str, ...], list[tuple[str, str]]] = OrderedDict()
    for scenario, groups in GROUP_SCENARIOS.items():
        for name, features in groups.items():
            key = tuple(f for f in DRIVER_FEATURES if f in features)
            result.setdefault(key, []).append((scenario, name))
    return result


def build_model_for_fold(df: pd.DataFrame, task: dict, model_name: str,
                         score_row: pd.Series) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    r, f, seed = int(task["repeat"]), int(task["fold"]), int(task["seed"])
    tr, te = np.asarray(task["train_idx"]), np.asarray(task["test_idx"])
    Xtr = df.iloc[tr][MAIN_MODEL_FEATURES].copy()
    Xte = df.iloc[te][MAIN_MODEL_FEATURES].copy()
    ytr = df.iloc[tr][TARGET].copy()
    params = score_row["Best_Params_JSON"]
    params = json.loads(params) if isinstance(params, str) else dict(params or {})
    n_est = score_row.get("XGB_Final_n_estimators", np.nan)
    n_est = None if pd.isna(n_est) else int(n_est)
    model = fit_outer_model(
        model_name, params, n_est, Xtr, ytr, seed + r * 100 + f,
        MAIN_MODEL_FEATURES, model_n_jobs=MODEL_INTERNAL_N_JOBS,
    )
    core, Ztr, Zte = prepare_core_model(model, Xtr, Xte)
    # 必须确认折内变换并没有改变与原 Pipeline 等价的预测。
    test_a = np.asarray(model.predict(Xte), dtype=float)
    test_b = np.asarray(core.predict(Zte), dtype=float)
    if not np.allclose(test_a, test_b, atol=1e-7, rtol=1e-6):
        raise RuntimeError(f"R{r}F{f} 模型拆解前后预测不一致。")
    yte = df.iloc[te][TARGET].to_numpy(float)
    actual_r2 = float(r2_score(yte, test_b))
    stored_r2 = float(score_row["Test_R2"])
    if abs(actual_r2 - stored_r2) > MODEL_R2_MATCH_TOLERANCE:
        raise RuntimeError(
            f"R{r}F{f} 重建模型 R²={actual_r2:.6f}，而 01 记录 R²={stored_r2:.6f}。"
            "可能是 sklearn/xgboost 版本、参数或随机种子发生变化，请先核对。"
        )
    return model, core, Ztr, Zte


def compute_gpfi_one_fold(core: Any, Zte: np.ndarray, yte: np.ndarray,
                          grids: np.ndarray, years: np.ndarray,
                          task: dict, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    r, f = int(task["repeat"]), int(task["fold"])
    rng = np.random.default_rng(RANDOM_SEED + r * 10000 + f * 500)
    colmap = {name: i for i, name in enumerate(MAIN_MODEL_FEATURES)}
    group_indices = {group: np.array([colmap[x] for x in group], dtype=int)
                     for group in unique_variable_groups()}
    base_pred = core.predict(Zte)
    base_r2 = float(r2_score(yte, base_pred))
    base_rmse = float(np.sqrt(mean_squared_error(yte, base_pred)))
    raw = []

    # 同一次 donor 抽签应用于所有变量组，减少不同组比较的 Monte Carlo 噪声。
    for perm_no in range(1, GPFI_PERMUTATIONS_PER_FOLD + 1):
        donor_idx = make_grid_trajectory_permutation(grids, years, rng)
        Z_donor = Zte[donor_idx, :]
        for group_key, indices in group_indices.items():
            Zp = Zte.copy()
            Zp[:, indices] = Z_donor[:, indices]
            p = np.asarray(core.predict(Zp), dtype=float)
            r2p = float(r2_score(yte, p))
            rmsep = float(np.sqrt(mean_squared_error(yte, p)))
            raw.append({
                "Model": model_name, "repeat": r, "fold": f, "permutation": perm_no,
                "feature_group": "+".join(group_key),
                "features": "+".join(group_key), "n_features": len(group_key),
                "n_test_samples": len(yte), "n_test_grids": len(np.unique(grids)),
                "baseline_Test_R2": base_r2, "permuted_Test_R2": r2p,
                "GPFI_delta_R2": base_r2 - r2p,
                "baseline_Test_RMSE": base_rmse, "permuted_Test_RMSE": rmsep,
                "GPFI_delta_RMSE": rmsep - base_rmse,
                "permutation_unit": "joint-grid-trajectory_same-year",
                "Year_held_fixed": True,
            })

    raw_df = pd.DataFrame(raw)
    fold_df = raw_df.groupby(["Model", "repeat", "fold", "feature_group", "features", "n_features"],
                             as_index=False).agg(
        GPFI_delta_R2_mean=("GPFI_delta_R2", "mean"),
        GPFI_delta_R2_permSD=("GPFI_delta_R2", "std"),
        GPFI_delta_RMSE_mean=("GPFI_delta_RMSE", "mean"),
        n_permutations=("permutation", "nunique"),
        baseline_Test_R2=("baseline_Test_R2", "first"),
        n_test_samples=("n_test_samples", "first"),
        n_test_grids=("n_test_grids", "first"),
    )
    return raw_df, fold_df


def expand_gpfi_scenarios(folds: pd.DataFrame) -> pd.DataFrame:
    lookup_rows = []
    assignments = unique_variable_groups()
    for groupkey, uses in assignments.items():
        identifier = "+".join(groupkey)
        sub = folds[folds["feature_group"] == identifier].copy()
        for scenario, group_name in uses:
            dd = sub.copy()
            dd["scenario"] = scenario
            dd["group"] = group_name
            lookup_rows.append(dd)
    return pd.concat(lookup_rows, ignore_index=True)


def aggregate_gpfi(folds: pd.DataFrame, folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_folds = expand_gpfi_scenarios(folds)
    write_csv(scenario_folds, folder / "GPFI_25OuterFolds_ByScenario.csv")
    rows = []
    for (scenario, grp), d in scenario_folds.groupby(["scenario", "group"], sort=False):
        if len(d) != 25:
            raise RuntimeError(f"GPFI {scenario}/{grp} 缺少外层折，实际 {len(d)}/25。")
        vals = d["GPFI_delta_R2_mean"].to_numpy(float)
        rows.append({
            "scenario": scenario, "group": grp,
            "features": d["features"].iloc[0], "n_features": int(d["n_features"].iloc[0]),
            "n_outer_folds": len(vals), "permutations_per_fold": GPFI_PERMUTATIONS_PER_FOLD,
            "mean_delta_R2": float(vals.mean()), "SD_25_folds": float(vals.std(ddof=1)),
            "median_delta_R2": float(np.median(vals)),
            "fold_q2_5": float(np.quantile(vals, 0.025)),
            "fold_q97_5": float(np.quantile(vals, 0.975)),
            "mean_delta_RMSE": float(d["GPFI_delta_RMSE_mean"].mean()),
            "interval_warning": "Fold empirical percentiles, NOT 95% confidence interval",
        })
    summary = pd.DataFrame(rows)
    write_csv(summary, folder / "GPFI_GroupSummary_25OuterFolds.csv")

    comparisons = []
    pairwise_rows = []
    for scenario, (left, right) in GPFI_PAIRS.items():
        sub = scenario_folds[scenario_folds["scenario"] == scenario]
        a = sub[sub["group"] == left][["repeat", "fold", "GPFI_delta_R2_mean"]]
        b = sub[sub["group"] == right][["repeat", "fold", "GPFI_delta_R2_mean"]]
        paired = a.merge(b, on=["repeat", "fold"], validate="one_to_one",
                         suffixes=("_left", "_right"))
        if len(paired) != 25:
            raise RuntimeError(f"{scenario} 的两个变量组缺少配对折。")
        paired["scenario"] = scenario
        paired["left_group"] = left
        paired["right_group"] = right
        paired["difference_delta_R2_left_minus_right"] = (
            paired["GPFI_delta_R2_mean_left"] - paired["GPFI_delta_R2_mean_right"]
        )
        pairwise_rows.append(paired)
        d = paired["difference_delta_R2_left_minus_right"].to_numpy(float)
        by_repeat = paired.groupby("repeat")["difference_delta_R2_left_minus_right"].mean()
        comparisons.append({
            "scenario": scenario, "comparison": f"{left} minus {right}",
            "mean_paired_delta_R2_difference": float(d.mean()),
            "SD_25_folds": float(d.std(ddof=1)),
            "n_left_greater_than_right": int((d > 0).sum()),
            "n_outer_folds": len(d),
            "fraction_left_greater": float((d > 0).mean()),
            "repeat_level_mean_difference": float(by_repeat.mean()),
            "repeat_level_SD_5_repeats": float(by_repeat.std(ddof=1)),
            "n_repeats_with_positive_mean": int((by_repeat > 0).sum()),
            "inference_warning": "25 folds overlap in training data; do not treat as 25 independent tests",
        })
    contrasts = pd.DataFrame(comparisons)
    paired_full = pd.concat(pairwise_rows, ignore_index=True)
    write_csv(contrasts, folder / "GPFI_GroupDominance_Scenarios.csv")
    write_csv(paired_full, folder / "GPFI_PairedFoldDifferences.csv")
    return summary, contrasts, scenario_folds


# =============================================================================
# 5. 真正的 group Shapley：按变量组作为博弈参与者，训练背景蒙特卡洛积分
# =============================================================================
def one_observation_per_grid(grids: np.ndarray, limit: int,
                             rng: np.random.Generator) -> np.ndarray:
    groups = np.asarray(grids, dtype=str)
    selected_groups = rng.choice(np.unique(groups), size=min(limit, len(np.unique(groups))),
                                 replace=False)
    chosen = [int(rng.choice(np.flatnonzero(groups == g))) for g in selected_groups]
    return np.array(chosen, dtype=int)


def exact_group_shapley_interventional(core: Any, Z_eval: np.ndarray,
                                        Z_background: np.ndarray,
                                        group_definitions: OrderedDict
                                        ) -> np.ndarray:
    """
    对 k 个变量组枚举全部 2^k 个联盟，严格按 Shapley 组合权重计算；
    v(S) = mean_b model(x_S, background_b,-S)。
    'exact' 指枚举所有联盟，背景期望仍使用选定的训练集样本近似。
    """
    group_names = list(group_definitions)
    k = len(group_names)
    if not 2 <= k <= 8:
        raise RuntimeError("群体 Shapley 组数异常。")
    featpos = {f: i for i, f in enumerate(MAIN_MODEL_FEATURES)}
    group_cols = [np.array([featpos[f] for f in group_definitions[g]], dtype=int)
                  for g in group_names]
    ne = Z_eval.shape[0]
    nb = Z_background.shape[0]
    coalition_values = np.empty((1 << k, ne), dtype=float)
    # 一个联盟一次批量预测 n_eval * n_background 行；避免每个样本单独 predict。
    for mask in range(1 << k):
        hybrids = np.tile(Z_background, (ne, 1))
        for j, cols in enumerate(group_cols):
            if mask & (1 << j):
                hybrids[:, cols] = np.repeat(Z_eval[:, cols], nb, axis=0)
        preds = np.asarray(core.predict(hybrids), dtype=float)
        coalition_values[mask, :] = preds.reshape(ne, nb).mean(axis=1)

    output = np.zeros((ne, k), dtype=float)
    factorial = [math.factorial(i) for i in range(k + 1)]
    denom = float(factorial[k])
    for j in range(k):
        for mask in range(1 << k):
            if mask & (1 << j):
                continue
            size_s = int(mask.bit_count())
            weight = factorial[size_s] * factorial[k - size_s - 1] / denom
            output[:, j] += weight * (coalition_values[mask | (1 << j)] - coalition_values[mask])
    residual = (output.sum(axis=1) -
                (coalition_values[(1 << k) - 1] - coalition_values[0]))
    if not np.allclose(residual, 0.0, atol=2e-6, rtol=1e-5):
        raise RuntimeError("Group Shapley 的效率性（efficiency）检查不通过。")
    return output


def compute_true_group_shap_one_fold(core: Any, Ztr: np.ndarray, Zte: np.ndarray,
                                      df: pd.DataFrame, task: dict) -> pd.DataFrame:
    r, f = int(task["repeat"]), int(task["fold"])
    tr, te = np.asarray(task["train_idx"]), np.asarray(task["test_idx"])
    train_grids = df.iloc[tr][GROUP_COLUMN].to_numpy(str)
    test_grids = df.iloc[te][GROUP_COLUMN].to_numpy(str)
    rng = np.random.default_rng(RANDOM_SEED + r * 10000 + f * 700 + 100)
    test_sample = one_observation_per_grid(test_grids, GROUP_SHAP_TEST_GRIDS_PER_FOLD, rng)
    background_sample = one_observation_per_grid(train_grids, GROUP_SHAP_BACKGROUND_GRIDS_PER_FOLD, rng)
    Z_eval = Zte[test_sample, :]
    Z_bg = Ztr[background_sample, :]
    if len(test_sample) == 0 or len(background_sample) == 0:
        raise RuntimeError("真正的组 Shapley 没有可用的测试/训练背景样本。")
    sampled_df = df.iloc[te[test_sample]]

    rows = []
    for scenario in GROUP_SHAP_SCENARIOS:
        group_def = OrderedDict(GROUP_SCENARIOS[scenario])
        group_def["Year"] = (YEAR_COLUMN,)
        effects = exact_group_shapley_interventional(core, Z_eval, Z_bg, group_def)
        for i, (_, observation) in enumerate(sampled_df.iterrows()):
            for j, (group_name, variables) in enumerate(group_def.items()):
                rows.append({
                    "scenario": scenario, "repeat": r, "fold": f,
                    SAMPLE_ID_COLUMN: int(observation[SAMPLE_ID_COLUMN]),
                    GROUP_COLUMN: str(observation[GROUP_COLUMN]),
                    YEAR_COLUMN: int(observation[YEAR_COLUMN]),
                    "group": group_name, "features": "+".join(variables),
                    "signed_group_shapley": float(effects[i, j]),
                    "abs_group_shapley": float(abs(effects[i, j])),
                    "n_test_grids_sampled": len(test_sample),
                    "n_training_background_grids": len(background_sample),
                    "definition": "Interventional group Shapley; full coalition enumeration; MC train background",
                })
    return pd.DataFrame(rows)


def aggregate_true_group_shap(raw: pd.DataFrame, folder: Path) -> pd.DataFrame:
    summary_records = []
    rng = np.random.default_rng(RANDOM_SEED + 910)
    for scenario, d in raw.groupby("scenario"):
        per_grid = d.groupby([GROUP_COLUMN, "group"])["abs_group_shapley"].agg(["sum", "size"])
        names = list(GROUP_SCENARIOS[scenario]) + ["Year"]
        # 所有组必须有同一批抽样观测，因此 sum/size 均可通过 pivot 对齐。
        sums = per_grid["sum"].unstack("group").reindex(columns=names).fillna(0.0)
        ns = per_grid["size"].unstack("group").reindex(columns=names).fillna(0.0)
        if not np.all(ns.to_numpy() == ns.to_numpy()[:, [0]]):
            raise RuntimeError("组 Shapley 的不同变量组并非同一批观测。")
        total_count = ns[names[0]].sum()
        means = sums.sum(axis=0).to_numpy() / total_count
        ecological_total = float(means[:-1].sum())
        full_total = float(means.sum())
        n_grid = len(sums)
        draws = np.empty((CLUSTER_BOOTSTRAP_B, len(names)), float)
        arr = sums.to_numpy(float)
        countarr = ns[names[0]].to_numpy(float)
        for b in range(CLUSTER_BOOTSTRAP_B):
            ix = rng.integers(0, n_grid, size=n_grid)
            bootstrap_means = arr[ix].sum(axis=0) / countarr[ix].sum()
            draws[b, :] = 100 * bootstrap_means / bootstrap_means.sum()
        for j, group_name in enumerate(names):
            is_ecol = group_name != "Year"
            eco_pct = 100 * means[j] / ecological_total if is_ecol else np.nan
            # second bootstrap denominator only across ecological group effects
            boot_ecol = 100 * draws[:, j] / draws[:, :-1].sum(axis=1) if is_ecol else None
            summary_records.append({
                "scenario": scenario, "group": group_name,
                "features": "+".join(dict(GROUP_SCENARIOS[scenario], Year=("Year",))[group_name]),
                "n_unique_test_grids": n_grid,
                "n_sampled_observation_fold_occurrences": int(total_count),
                "mean_abs_true_group_shapley": float(means[j]),
                "share_all_groups_percent": float(100 * means[j] / full_total),
                "share_all_groups_CI_lower_2_5": float(np.quantile(draws[:, j], 0.025)),
                "share_all_groups_CI_upper_97_5": float(np.quantile(draws[:, j], 0.975)),
                "ecological_only_percent": float(eco_pct) if is_ecol else np.nan,
                "ecological_only_CI_lower_2_5": float(np.quantile(boot_ecol, 0.025)) if is_ecol else np.nan,
                "ecological_only_CI_upper_97_5": float(np.quantile(boot_ecol, 0.975)) if is_ecol else np.nan,
                "type": "True interventional group Shapley on subsampled out-of-fold observations",
                "bootstrap_unit": GROUP_COLUMN,
            })
    summary = pd.DataFrame(summary_records)
    write_csv(summary, folder / "True_GroupShapley_Summary_with_GridBootstrapCI.csv")
    return summary


# =============================================================================
# 6. 图件、相关性和方法说明
# =============================================================================
def correlations(df: pd.DataFrame, folder: Path) -> pd.DataFrame:
    variables = ["TMX", "TMN", "SR", "SM", "PRE", "VPD", "GI"]
    pairwise = []
    for i, a in enumerate(variables):
        for b in variables[i + 1:]:
            complete = df[[a, b]].dropna()
            pairwise.append({
                "feature_1": a, "feature_2": b,
                "Pearson_r": float(complete[a].corr(complete[b])),
                "n_nonmissing_pairs": len(complete),
            })
    corr = pd.DataFrame(pairwise)
    write_csv(corr, folder / "Predictor_Pearson_Correlations.csv")
    matrix = df[variables].corr()
    matrix.to_csv(folder / "Predictor_Pearson_Matrix.csv", encoding="utf-8-sig")
    # 仅去除每年均值再计算相关性，用于识别年际共同趋势影响；不称为偏相关。
    de = df[variables + [YEAR_COLUMN]].copy()
    for name in variables:
        de[name] = de[name] - de.groupby(YEAR_COLUMN)[name].transform("mean")
    de[variables].corr().to_csv(folder / "Predictor_WithinYear_Demeaned_Correlation.csv",
                                encoding="utf-8-sig")
    return corr


def plot_horizontal_bars(df: pd.DataFrame, value: str, low: str | None, high: str | None,
                         title: str, xlabel: str, base: Path) -> None:
    dat = df.copy().sort_values(value, ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(3.7, 0.53 * len(dat) + 1.6)))
    x = dat[value].to_numpy(float)
    y = np.arange(len(dat))
    if low and high:
        lower = dat[low].to_numpy(float)
        upper = dat[high].to_numpy(float)
        err = np.vstack([np.clip(x - lower, 0, None), np.clip(upper - x, 0, None)])
    else:
        err = None
    ax.barh(y, x, xerr=err, color="#4F7898", edgecolor="#28485E", height=0.64,
            error_kw={"capsize": 3, "elinewidth": 0.9})
    ax.set_yticks(y, dat["group"].tolist())
    ax.axvline(0, color="0.28", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, base)


def plot_sensitivity_comparison(original: pd.DataFrame, base: Path) -> None:
    # 一张表更适合展示多方案中不同变量组数目的变动；视觉图仅展示温度组/水分组对应行。
    mask = original["scenario"].isin(["Baseline", "VPD_Separate", "VPD_to_Temperature"])
    d = original[mask & original["group"].str.contains("Temperature|Moisture", regex=True)].copy()
    d["label"] = d["scenario"] + " | " + d["group"]
    d = d.sort_values("ecological_relative_percent")
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    colors = ["#B96D54" if "Temperature" in n else "#527EA0" for n in d["group"]]
    ax.barh(np.arange(len(d)), d["ecological_relative_percent"], color=colors,
            edgecolor="0.25", linewidth=0.5)
    ax.set_yticks(np.arange(len(d)), d["label"].tolist())
    ax.set_xlabel("Sum of individual OOF-SHAP importance (%)")
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, base)


def save_method_documentation(folder: Path, main_run: Path, selected_model: str,
                              year_share: float, n_rows: int, n_grid: int) -> None:
    paragraphs = f"""Comment 16 / Reviewer 2 item 3.12: Methods notes (for subsequent manuscript drafting)

1. Reused model: {selected_model}. The same 25 outer training/test partitions (5 folds repeated 5 times, grouped by patch_id1), fold-specific hyperparameters, and random seeds as the original nested group-CV analysis were used. This supplemental calculation DID NOT retune the model. Input observations: {n_rows}; spatial grid units: {n_grid}.

2. Ecological grouping: TMX and TMN characterize thermal conditions and SR represents available shortwave radiative energy. SM and PRE describe water availability, while VPD describes atmospheric evaporative demand. GI represents grazing pressure. VPD is temperature dependent, and cross-group correlation (including that of SR with PRE) should be acknowledged. These groups are interpretive categories, not independently identified causal processes.

3. Original OOF-SHAP aggregation: Sum the mean absolute SHAP contributions of component ecological predictors within each group; normalize by the sum across the seven ecological predictors only. This produces a grouped summary of INDIVIDUAL SHAP values, not group Shapley values. Year remained a model covariate, but was excluded only from this seven-predictor denominator. The full-model eight-predictor Year share calculated using these data is {year_share:.4f}%.

4. Grouped permutation feature importance (GPFI): Within each held-out outer fold, one shared random permutation of the grid identifiers was used to exchange complete same-year trajectories of all features belonging to the tested group; other predictor columns, including Year, were left unchanged. The change in held-out R2, R2_original - R2_permuted, was averaged over {GPFI_PERMUTATIONS_PER_FOLD} permutations per fold, then summarized across the 25 outer folds. Joint permutation preserves within-group dependence and the intra-grid temporal sequence, but it does not resolve between-group dependence and may generate unusual feature combinations. GPFI differences are NOT SHAP percentages and do NOT represent causal effect sizes.

5. Grouping sensitivity: Besides the baseline classes, alternative allocations treated VPD as a separate atmospheric-dryness category, included VPD in the temperature-related group, or treated SR as a separate radiation category. Sensitivity scenarios re-label or jointly perturb existing inputs without changing the prediction model. A change in grouped SHAP percentages from reallocation is arithmetic sensitivity, not fresh predictive evidence. For separate SR groups, group PFI values must NOT be summed because permutation importances are not additive.

6. True grouped Shapley (if run): groups (including Year as its own group) are players in the Shapley game. Each coalition's prediction expectation is approximated using background records from the matching outer TRAINING partition. All 2^k coalitions were enumerated, and standard Shapley weights applied. Evaluation uses at most {GROUP_SHAP_TEST_GRIDS_PER_FOLD} test grids per outer fold (one observation per grid) and at most {GROUP_SHAP_BACKGROUND_GRIDS_PER_FOLD} training-grid background samples; it is therefore a background-sampled, test-subsampled INTERVENTIONAL group Shapley estimate, not an exact conditional Shapley analysis over all 24,084 observations. Report true group-Shapley summaries separately from sums of individual SHAP absolute values.

7. Grouped SHAP uncertainty: patch_id1 cluster bootstrap with B={CLUSTER_BOOTSTRAP_B}. Group permutation empirical quantiles among 25 outer folds are descriptive and NOT independent-sample confidence intervals. Training/test splits from repeated group CV overlap, so a count of favorable folds is descriptive, not a binomial significance test.

8. Scientific conclusion: Explain original ecological classification and evaluate whether temperature/energy is consistently more influential under the specified sensitivity definitions. Do not assert the originally reported dominance is robust before inspecting the actual new results, and do not equate importance with causal control of grassland resilience.

以下为解释说明，不是已经证明的生态机制。审稿回复必须根据实际输出判断 H2 的稳健性，不能将SHAP/GPFI解释为独立因果贡献。
"""
    (folder / "Methods_and_Interpretation_Notes_EN_CN.txt").write_text(paragraphs, encoding="utf-8-sig")
    # 输出一份运行指引，用户可以直接据此检查哪些文件应进入补充材料。
    guide = """Diversity - Reviewer 2 Comment 16 / 运行结果目录说明

00_config: 输入/版本哈希、主模型目录和执行参数。
01_correlation: 原因子相关性、年均值去除后的相关性对照。
02_original_shap: 原 01 OOF-SHAP 分组汇总及格网 bootstrap、Year 占比、与 No-Year 对照（如存在）。
03_gpfi: 25 折的原始置换结果、各分组的 ΔR² / ΔRMSE、组间配对差值及描述性分位数。
04_true_group_shapley: 整组作为博弈参与者的组 Shapley 抽样结果及格网 bootstrap。
05_figures: 600 dpi PNG/TIF 和矢量 PDF/SVG 图。
06_methods: 方法与解释边界、全体结果摘要表。

对审稿人汇报时优先查阅：
 02_original_shap/SHAP_Original_Grouped_with_GridBootstrapCI.csv
 03_gpfi/GPFI_GroupSummary_25OuterFolds.csv
 03_gpfi/GPFI_GroupDominance_Scenarios.csv
 04_true_group_shapley/True_GroupShapley_Summary_with_GridBootstrapCI.csv
 06_methods/Comment16_Key_Tables.xlsx

注意：用户原路径内的 01、02、03、04 文件及运行结果均不会被覆盖。
"""
    (folder / "README_Results_CN.txt").write_text(guide, encoding="utf-8-sig")


# =============================================================================
# 7. 主执行
# =============================================================================
def main() -> None:
    started = datetime.now()
    validate_scenarios()
    if GPFI_PERMUTATIONS_PER_FOLD < 2:
        raise ValueError("GPFI_PERMUTATIONS_PER_FOLD 应至少为 2。")
    if GROUP_SHAP_BACKGROUND_GRIDS_PER_FOLD < 2:
        raise ValueError("组 Shapley 至少需要两个训练背景格网。")
    log("Reviewer 2 Comment 16：开始运行分组 SHAP 和 GPFI 补充检验。")
    require_file(DATA_CSV, "建模输入 CSV")
    main_run = resolve_main_run()
    log(f"01 主分析目录：{main_run}")
    data_hash = sha256_file(DATA_CSV)
    original_hash = verify_main_dataset_hash(main_run, data_hash)
    run_dir = create_or_resume_output(main_run, data_hash)
    folders = {name: run_dir / name for name in [
        "00_config", "01_correlation", "02_original_shap", "03_gpfi",
        "04_true_group_shapley", "05_figures", "06_methods",
    ]}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    log(f"新增结果文件夹：{run_dir}")

    selection = read_json(main_run / "selected_model_for_sensitivity.json")
    selected_model = str(selection["selected_model"])
    scores = pd.read_csv(main_run / "02_main_nested_cv" / "all_outer_scores.csv")
    scores = scores[scores["Model"].astype(str) == selected_model]
    if len(scores) != 25 or scores.duplicated(["Repeat", "Outer_Fold"]).any():
        raise RuntimeError(f"选定模型 {selected_model} 的 01 主分析结果不是完整的 25 套外层任务。")
    score_lookup = scores.set_index(["Repeat", "Outer_Fold"])
    df = load_model_data(DATA_CSV, MAIN_MODEL_FEATURES, require_block100=True)
    assignment = pd.read_csv(main_run / "01_split_manifest" /
                             "outer_split_assignment_by_observation.csv")
    tasks = load_outer_tasks_from_assignment(df, assignment, GROUP_COLUMN)
    if len(tasks) != 25:
        raise RuntimeError(f"主分析划分应该为 25 套，实际读到 {len(tasks)} 套。")
    checked = validate_outer_tasks(tasks, df, GROUP_COLUMN)
    if int(checked["overlap_groups"].sum()) != 0:
        raise RuntimeError("外层训练和测试格网出现泄漏。")
    write_csv(checked, folders["00_config"] / "Reused_Outer_Split_Validation.csv")
    log(f"数据量：{len(df)} 行，{df[GROUP_COLUMN].nunique()} 个格网；实际最优模型：{selected_model}。")

    write_json({
        "model": selected_model, "input_csv": str(DATA_CSV),
        "output_folder": str(run_dir), "main_run": str(main_run),
        "n_observations": len(df), "n_grids": int(df[GROUP_COLUMN].nunique()),
        "input_sha256": data_hash, "main_input_sha256": original_hash,
        "n_outer_models": 25, "outer_repeats": 5, "outer_folds": 5,
        "inner_CV_as_in_original_01": "5-fold Group CV used when determining the already-fitted hyperparameters",
        "permutations_per_outer_fold": GPFI_PERMUTATIONS_PER_FOLD,
        "permutation_scheme": "exchange complete grid trajectories between test grids, year aligned",
        "group_shapley_enabled": RUN_TRUE_GROUP_SHAPLEY,
        "group_shapley_scenarios": GROUP_SHAP_SCENARIOS,
        "group_shapley_test_grids_per_fold": GROUP_SHAP_TEST_GRIDS_PER_FOLD,
        "group_shapley_background_grids_per_fold": GROUP_SHAP_BACKGROUND_GRIDS_PER_FOLD,
        "cluster_bootstrap_B": CLUSTER_BOOTSTRAP_B,
        "group_scenarios": GROUP_SCENARIOS,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__,
        "start": started.isoformat(),
    }, folders["00_config"] / "Analysis_Parameters.json")

    # 1) 相关性 / 旧 SHAP
    log("步骤 1/4：核对相关性和已经完成的 OOF-SHAP 分组结果。")
    corr = correlations(df, folders["01_correlation"])
    obs_path = main_run / "04_oof_shap" / "OOF_SHAP_mean_across_5_repeats_per_observation.csv"
    obs = pd.read_csv(require_file(obs_path, "01 OOF-SHAP 观测级汇总"))
    if len(obs) != len(df) or obs[SAMPLE_ID_COLUMN].nunique() != len(df):
        raise RuntimeError("01 OOF-SHAP 并未恰好覆盖当前 CSV 的全部观测。")
    validate_shap_original(main_run, obs)
    original_group, meta = bootstrap_original_shap(obs, folders["02_original_shap"])
    noyear = noyear_shap_comparison(main_run, original_group, folders["02_original_shap"])
    log("已计算原始 SHAP 四套分组方案与格网级 bootstrap。")
    del obs
    gc.collect()

    # 2) 25 组外测重建与联合置换；同时做 group Shapley（模型只拟合一次/折）。
    log("步骤 2/4：复用 25 折、各折最优超参数，执行 GPFI 与真正的组 Shapley。")
    gpfi_ck = folders["03_gpfi"] / "fold_checkpoints"
    shap_ck = folders["04_true_group_shapley"] / "fold_checkpoints"
    gpfi_ck.mkdir(parents=True, exist_ok=True)
    shap_ck.mkdir(parents=True, exist_ok=True)
    for task_index, task in enumerate(tasks, start=1):
        r, f = int(task["repeat"]), int(task["fold"])
        base_name = f"repeat_{r:02d}_fold_{f:02d}"
        gpfi_file = gpfi_ck / f"{base_name}_GPFI_raw.csv"
        fold_file = gpfi_ck / f"{base_name}_GPFI_summary.csv"
        shap_file = shap_ck / f"{base_name}_TrueGroupShap.csv"
        have_gpfi = gpfi_file.exists() and fold_file.exists()
        have_shap = not RUN_TRUE_GROUP_SHAPLEY or shap_file.exists()
        if have_gpfi and have_shap:
            log(f"[{task_index}/25] R{r}F{f} 已完成，自动跳过。")
            continue
        log(f"[{task_index}/25] R{r}F{f} 重建 {selected_model}，计算剩余任务。")
        model, core, Ztr, Zte = build_model_for_fold(
            df, task, selected_model, score_lookup.loc[(r, f)]
        )
        if not have_gpfi:
            te = np.asarray(task["test_idx"])
            raw, fold_summary = compute_gpfi_one_fold(
                core, Zte, df.iloc[te][TARGET].to_numpy(float),
                df.iloc[te][GROUP_COLUMN].to_numpy(str),
                df.iloc[te][YEAR_COLUMN].to_numpy(), task, selected_model,
            )
            write_csv(raw, gpfi_file)
            write_csv(fold_summary, fold_file)
        if RUN_TRUE_GROUP_SHAPLEY and not have_shap:
            result = compute_true_group_shap_one_fold(core, Ztr, Zte, df, task)
            write_csv(result, shap_file)
        del model, core, Ztr, Zte
        gc.collect()

    # 3) 完整性检查与汇总
    log("步骤 3/4：整合 25 套外测结果与分组对照。")
    raw_files = sorted(gpfi_ck.glob("repeat_*_fold_*_GPFI_raw.csv"))
    fold_files = sorted(gpfi_ck.glob("repeat_*_fold_*_GPFI_summary.csv"))
    if len(raw_files) != 25 or len(fold_files) != 25:
        raise RuntimeError("GPFI 未生成完整的 25 套结果。")
    gpfi_all = pd.concat([pd.read_csv(p) for p in raw_files], ignore_index=True)
    gpfi_fold = pd.concat([pd.read_csv(p) for p in fold_files], ignore_index=True)
    write_csv(gpfi_all, folders["03_gpfi"] / "GPFI_AllPermutations_25OuterFolds.csv")
    write_csv(gpfi_fold, folders["03_gpfi"] / "GPFI_AllUniqueGroups_25OuterFolds.csv")
    gpfi_summary, gpfi_contrasts, gpfi_scenarios = aggregate_gpfi(gpfi_fold, folders["03_gpfi"])

    true_group_summary = pd.DataFrame()
    if RUN_TRUE_GROUP_SHAPLEY:
        shap_files = sorted(shap_ck.glob("repeat_*_fold_*_TrueGroupShap.csv"))
        if len(shap_files) != 25:
            raise RuntimeError("真正的分组 Shapley 没有完整覆盖 25 套外层折。")
        true_group_raw = pd.concat([pd.read_csv(p) for p in shap_files], ignore_index=True)
        write_csv(true_group_raw, folders["04_true_group_shapley"] / "True_GroupShapley_Sampled_25Folds.csv")
        true_group_summary = aggregate_true_group_shap(true_group_raw, folders["04_true_group_shapley"])

    # 4) 表、图和方法说明
    log("步骤 4/4：生成论文表格、出版用图件和方法说明。")
    configure_figures()
    figdir = folders["05_figures"]
    baseline_og = original_group[original_group["scenario"] == "Baseline"]
    plot_horizontal_bars(
        baseline_og, "ecological_relative_percent", "CI_lower_2_5", "CI_upper_97_5",
        "Original OOF-SHAP grouped summary", "Relative ecological SHAP importance (%)",
        figdir / "Fig_S_Comment16_Original_SHAP_Grouped",
    )
    baseline_gpfi = gpfi_summary[gpfi_summary["scenario"] == "Baseline"]
    plot_horizontal_bars(
        baseline_gpfi, "mean_delta_R2", "fold_q2_5", "fold_q97_5",
        "Grouped permutation importance (25 outer folds)",
        "Decrease in held-out R² (fold empirical interval)",
        figdir / "Fig_S_Comment16_GPFI_Baseline",
    )
    plot_sensitivity_comparison(original_group, figdir / "Fig_S_Comment16_SHAP_Grouping_Sensitivity")
    if not true_group_summary.empty:
        for scenario in GROUP_SHAP_SCENARIOS:
            subset = true_group_summary[true_group_summary["scenario"] == scenario]
            plot_horizontal_bars(
                subset, "share_all_groups_percent", "share_all_groups_CI_lower_2_5",
                "share_all_groups_CI_upper_97_5", f"True grouped Shapley: {scenario}",
                "Interventional group Shapley share, including Year (%)",
                figdir / f"Fig_S_Comment16_TrueGroupedShapley_{scenario}",
            )

    with pd.ExcelWriter(folders["06_methods"] / "Comment16_Key_Tables.xlsx", engine="openpyxl") as xls:
        original_group.to_excel(xls, sheet_name="Original_SHAP_groups", index=False)
        gpfi_summary.to_excel(xls, sheet_name="GPFI_25folds", index=False)
        gpfi_contrasts.to_excel(xls, sheet_name="GPFI_pairwise", index=False)
        if not true_group_summary.empty:
            true_group_summary.to_excel(xls, sheet_name="True_group_Shapley", index=False)
        if not noyear.empty:
            noyear.to_excel(xls, sheet_name="NoYear_groups", index=False)
        corr.to_excel(xls, sheet_name="Correlations", index=False)
        meta.to_excel(xls, sheet_name="Year_share", index=False)

    save_method_documentation(
        folders["06_methods"], main_run, selected_model,
        float(meta.iloc[0]["year_share_8_predictors_percent"]),
        len(df), int(df[GROUP_COLUMN].nunique()),
    )
    write_json({
        "run_dir": str(run_dir), "selected_model": selected_model,
        "n_folds": 25, "n_permutation": GPFI_PERMUTATIONS_PER_FOLD,
        "raw_gpfi_rows": len(gpfi_all),
        "n_group_shapley_rows": int(len(true_group_raw)) if RUN_TRUE_GROUP_SHAPLEY else 0,
        "main_shap_baseline_groups": baseline_og[["group", "ecological_relative_percent"]].to_dict("records"),
        "gpfi_baseline_groups": baseline_gpfi[["group", "mean_delta_R2"]].to_dict("records"),
        "key_group_comparisons": gpfi_contrasts.to_dict("records"),
        "end": datetime.now().isoformat(),
        "elapsed_minutes": (datetime.now() - started).total_seconds() / 60.0,
        "conclusion_note": "Numerical results are empirical; assess robustness before writing claims",
    }, folders["06_methods"] / "Summary_for_Response16.json")

    manifest = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            manifest.append({"relative_path": str(path.relative_to(run_dir)),
                             "size_MB": path.stat().st_size / 1024**2})
    write_csv(pd.DataFrame(manifest), folders["00_config"] / "Output_File_Manifest.csv")
    (run_dir / "RUN_COMPLETE.flag").write_text(datetime.now().isoformat(), encoding="utf-8")
    (OUTPUT_ROOT / LATEST_POINTER).write_text(str(run_dir), encoding="utf-8")
    pointer = OUTPUT_ROOT / ACTIVE_POINTER
    if pointer.exists() and pointer.read_text(encoding="utf-8-sig").strip() == str(run_dir):
        pointer.unlink()

    log("=" * 75)
    log("Comment 16 分组SHAP、GPFI与分组敏感性检验全部完成。")
    log(f"实际最优模型：{selected_model}；外层折：25；置换次数/折：{GPFI_PERMUTATIONS_PER_FOLD}")
    for _, item in gpfi_contrasts.iterrows():
        log(f"{item['scenario']}: 组间 ΔR²差 = {item['mean_paired_delta_R2_difference']:.5f}; "
            f"正向折数 {item['n_left_greater_than_right']}/25")
    log(f"结果已保存在：{run_dir}")
    log("请结合输出结果判断 H2 稳健性，不要提前认定某一组主导。")


if __name__ == "__main__":
    main()
