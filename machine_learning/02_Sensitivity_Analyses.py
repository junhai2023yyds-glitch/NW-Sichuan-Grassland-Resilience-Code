# -*- coding: utf-8 -*-
"""
02_Sensitivity_Analyses.py

川西北草地生态系统韧性驱动因素 —— 三个敏感性分析

运行前提：
必须先完整运行 01_Main_ML_NestedCV_OOFSHAP_ALE.py。
本脚本会自动读取 OUTPUT_ROOT_FOLDER 下 latest_main_run.txt，获得主分析真实最优模型。

三个敏感性只运行“主分析实际最优模型”，不会重新让五模型比赛：
A. No-Year：7 个生态驱动变量，不含 Year；沿用主分析相同 25 套 patch_id1 外层测试划分；每个外层任务重新内层调参。
B. 100 km：7 驱动 + Year，但外层/内层均改按 Block100 分组；重新生成 5×5 block split；每个外层任务重新调参。
C. Temperature reparameterization：Tmean=(TMX+TMN)/2、DTR=TMX-TMN 替换 TMX/TMN；沿用主分析相同 patch_id1 外层划分；重新调参。

No-Year：比较性能、SHAP 排名/贡献、Top-6 ALE 形态和转折稳定性。
100 km：主要比较空间严格化后的 Test R²/RMSE/MAE 及 ΔR²。
Temperature：比较性能、温度组总贡献（TMX+TMN vs Tmean+DTR）和非温度变量 SHAP 稳定性；同时输出该敏感性自身 OOF-SHAP/ALE 图件。

所有图件继续：Times New Roman、600 dpi；变量名横轴中央、单位横轴最右端。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_common import (
    TARGET, GROUP_COLUMN, BLOCK100_COLUMN, YEAR_COLUMN,
    DATA_FILE_PATH, OUTPUT_ROOT_PATH,
    DRIVER_FEATURES, MAIN_MODEL_FEATURES, REPEAT_SEEDS, OUTER_N_SPLITS, ALE_TOP_N, SHAP_OUTER_PARALLEL_JOBS,
    FEATURE_UNITS,
    ensure_dir, safe_json_dump, safe_json_load,
    create_or_resume_run_directory, mark_run_complete, setup_logger,
    export_software_versions, export_file_hashes, export_search_space,
    load_model_data, input_structure_summary,
    load_outer_tasks_from_assignment, build_repeated_group_splits, validate_outer_tasks,
    run_repeated_nested_cv, summarize_model_performance, export_best_params,
    run_oof_shap, plot_shap_outputs, run_ale_25models, plot_ale_outputs,
    compare_main_vs_sensitivity_performance, compare_shap_contributions,
    plot_performance_sensitivity_comparison, plot_shap_contribution_comparison,
    plot_ale_main_vs_sensitivity, save_output_manifest,
)


# =============================================================================
# 1. 与主分析一致的 GitHub 公共复现路径
# =============================================================================
CSV_FILE_PATH = DATA_FILE_PATH
OUTPUT_ROOT_FOLDER = OUTPUT_ROOT_PATH

LATEST_MAIN_POINTER = OUTPUT_ROOT_FOLDER / "latest_main_run.txt"
SENS_RUN_PREFIX = "Final_ML_Sensitivity"
ACTIVE_SENS_POINTER = "active_sensitivity_run.txt"
LATEST_SENS_POINTER = "latest_sensitivity_run.txt"


# =============================================================================
# 2. 小工具
# =============================================================================
def read_latest_main_run() -> Path:
    if not LATEST_MAIN_POINTER.exists():
        raise FileNotFoundError(
            f"没有找到 {LATEST_MAIN_POINTER}。\n"
            "请先完整运行 01_Main_ML_NestedCV_OOFSHAP_ALE.py。"
        )
    main_run = Path(LATEST_MAIN_POINTER.read_text(encoding="utf-8").strip())
    if not main_run.exists() or not (main_run / "RUN_COMPLETE.flag").exists():
        raise RuntimeError("latest_main_run 指向的主分析目录不存在或尚未完整结束。")
    return main_run


def selected_only_summary(results: pd.DataFrame) -> pd.DataFrame:
    return summarize_model_performance(results)


def save_module_summary(summary: pd.DataFrame, folder: Path, name: str) -> None:
    summary.to_csv(folder / f"{name}_model_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(folder / f"{name}_model_summary.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)


def main() -> None:
    run_start = datetime.now()
    main_run = read_latest_main_run()
    selection = safe_json_load(main_run / "selected_model_for_sensitivity.json")
    selected_model = str(selection["selected_model"])

    # 断点续跑。若 active sensitivity 属于另一个主分析版本，则创建新目录。
    sens_run, resumed = create_or_resume_run_directory(
        OUTPUT_ROOT_FOLDER,
        SENS_RUN_PREFIX,
        ACTIVE_SENS_POINTER,
    )
    linked_file = sens_run / "linked_main_run.txt"
    if resumed and linked_file.exists():
        old_link = linked_file.read_text(encoding="utf-8").strip()
        if old_link != str(main_run):
            # 不允许把不同主分析版本混进同一个敏感性目录。
            active = OUTPUT_ROOT_FOLDER / ACTIVE_SENS_POINTER
            if active.exists():
                active.unlink()
            sens_run, resumed = create_or_resume_run_directory(
                OUTPUT_ROOT_FOLDER, SENS_RUN_PREFIX, ACTIVE_SENS_POINTER
            )
    linked_file.write_text(str(main_run), encoding="utf-8")

    config_dir = ensure_dir(sens_run / "00_config_environment")
    noyear_dir = ensure_dir(sens_run / "01_noYear")
    block_dir = ensure_dir(sens_run / "02_100km_BlockCV")
    temp_dir = ensure_dir(sens_run / "03_temperature_Tmean_DTR")
    fig_dir = ensure_dir(sens_run / "04_figures")
    supp_fig_dir = ensure_dir(sens_run / "05_supplementary_figures")
    logs_dir = ensure_dir(sens_run / "06_logs_checkpoints")

    logger = setup_logger(logs_dir / "sensitivity_analysis.log", name="NW_Sichuan_Sensitivity")
    logger.info("=" * 90)
    logger.info("敏感性分析开始")
    logger.info(f"关联主分析：{main_run}")
    logger.info(f"主分析真实最优模型：{selected_model}")
    logger.info("仅对该模型运行 No-Year / 100 km / Tmean+DTR；每个设定均重新内层调参。")
    logger.info("=" * 90)

    # 环境与复现信息
    config = {
        "input_csv": str(CSV_FILE_PATH),
        "linked_main_run": str(main_run),
        "selected_model_from_main": selected_model,
        "noYear": "same 25 patch_id1 outer splits as main; retune inside each outer training set",
        "block100": "new repeated 5x5 splits by Block100 for outer and inner CV; retune",
        "temperature": "same 25 patch_id1 outer splits as main; Tmean+DTR replace TMX+TMN; retune",
        "shap_outer_parallel_jobs": SHAP_OUTER_PARALLEL_JOBS,
        "feature_units": FEATURE_UNITS,
        "run_start": run_start.isoformat(),
    }
    safe_json_dump(config, config_dir / "sensitivity_config.json")
    export_software_versions(config_dir)
    export_search_space(config_dir)
    export_file_hashes(
        {
            "input_csv": CSV_FILE_PATH,
            "sensitivity_script": Path(__file__),
            "common_module": Path(__file__).with_name("ml_common.py"),
            "main_selected_model_json": main_run / "selected_model_for_sensitivity.json",
            "main_split_assignment": main_run / "01_split_manifest" / "outer_split_assignment_by_observation.csv",
        },
        config_dir,
    )

    # 数据与主分析结果
    df = load_model_data(CSV_FILE_PATH, MAIN_MODEL_FEATURES, require_block100=True)
    safe_json_dump(input_structure_summary(df, MAIN_MODEL_FEATURES), config_dir / "input_structure_summary.json")
    main_results = pd.read_csv(main_run / "02_main_nested_cv" / "all_outer_scores.csv")
    main_selected_results = main_results[main_results["Model"] == selected_model].copy()
    main_assignment = pd.read_csv(main_run / "01_split_manifest" / "outer_split_assignment_by_observation.csv")
    main_tasks = load_outer_tasks_from_assignment(df, main_assignment, GROUP_COLUMN)
    validate_outer_tasks(main_tasks, df, GROUP_COLUMN).to_csv(config_dir / "reused_main_patch_split_check.csv", index=False, encoding="utf-8-sig")
    main_contrib = pd.read_csv(main_run / "04_oof_shap" / "SHAP_contribution_bootstrap_CI_and_stability.csv")

    # 主分析 ALE 统一使用最终 OOF-SHAP 排名前 6。
    # 兼容已经跑完的旧 Top-4 主分析：如果原 05_ALE 中缺少 PRE/GI 等 Top-6 变量，
    # 本敏感性脚本只重新拟合最终模型的 25 个外层模型并补算 Top-6 ALE，
    # 不会重新进行五模型 Nested CV / Optuna。
    main_top6 = (
        main_contrib.sort_values("contribution_percent", ascending=False)["feature"]
        .tolist()[:ALE_TOP_N]
    )
    pd.DataFrame({"rank": np.arange(1, len(main_top6) + 1), "feature": main_top6}).to_csv(
        main_run / "04_oof_shap" / "Top6_features_for_ALE.csv", index=False, encoding="utf-8-sig"
    )

    original_main_ale_summary = pd.read_csv(main_run / "05_ALE" / "ALE_summary_95_empirical_interval.csv")
    original_main_ale_stability = pd.read_csv(main_run / "05_ALE" / "ALE_threshold_stability.csv")
    if set(main_top6).issubset(set(original_main_ale_summary["feature"].dropna().astype(str))):
        main_ale_summary = original_main_ale_summary
        main_ale_stability = original_main_ale_stability
        logger.info(f"主分析 ALE 已包含 Top-{ALE_TOP_N}：{main_top6}，直接复用。")
    else:
        logger.info(f"检测到旧主分析仅有 Top-4 ALE；开始补算主分析 Top-{ALE_TOP_N}：{main_top6}")
        main_ale_top6_dir = ensure_dir(main_run / "05_ALE_Top6")
        main_ale_top6 = run_ale_25models(
            df=df,
            features=MAIN_MODEL_FEATURES,
            top_features=main_top6,
            selected_model=selected_model,
            outer_tasks=main_tasks,
            model_results=main_results,
            output_folder=main_ale_top6_dir,
            logger=logger,
            progress_label="02 Main补算 | Top-6 ALE",
        )
        plot_ale_outputs(
            main_ale_top6, main_top6, selected_model,
            main_run / "06_main_figures", main_run / "07_supplementary_figures",
        )
        main_ale_summary = main_ale_top6["summary"]
        main_ale_stability = main_ale_top6["threshold_stability"]
        logger.info("主分析 Top-6 ALE 已补算完成；未重新运行主分析 Nested CV / Optuna。")

    # =========================================================================
    # A. No-Year 敏感性
    # =========================================================================
    logger.info("开始敏感性 A：No-Year")
    noyear_features = DRIVER_FEATURES.copy()
    noyear_cv_dir = ensure_dir(noyear_dir / "nested_cv")
    noyear_shap_dir = ensure_dir(noyear_dir / "oof_shap")
    noyear_ale_dir = ensure_dir(noyear_dir / "ALE")
    noyear_fig = ensure_dir(fig_dir / "noYear")
    noyear_supp = ensure_dir(supp_fig_dir / "noYear")

    noyear_results = run_repeated_nested_cv(
        df=df,
        features=noyear_features,
        group_column=GROUP_COLUMN,
        outer_tasks=main_tasks,
        model_names=[selected_model],
        output_folder=noyear_cv_dir,
        checkpoint_folder=logs_dir / "noYear",
        sqlite_db=logs_dir / "optuna_noYear.sqlite3",
        study_prefix=f"NOYEAR_{selected_model}",
        logger=logger,
        progress_label="02 No-Year | Nested CV",
    )
    noyear_summary = selected_only_summary(noyear_results)
    save_module_summary(noyear_summary, noyear_dir, "NoYear")
    export_best_params(noyear_results, noyear_cv_dir, "NoYear_Best_Hyperparameters_25_outer_tasks")

    noyear_perf = compare_main_vs_sensitivity_performance(main_results, noyear_results, selected_model, "No-Year")
    noyear_perf.to_csv(noyear_dir / "NoYear_vs_Main_performance.csv", index=False, encoding="utf-8-sig")
    plot_performance_sensitivity_comparison(noyear_perf, noyear_fig, f"{selected_model}: Main vs No-Year", "NoYear_vs_Main_Performance")

    noyear_shap = run_oof_shap(
        df=df,
        features=noyear_features,
        driver_features=DRIVER_FEATURES,
        selected_model=selected_model,
        outer_tasks=main_tasks,
        model_results=noyear_results,
        output_folder=noyear_shap_dir,
        logger=logger,
        group_column_for_training=GROUP_COLUMN,
        progress_label="02 No-Year | OOF-SHAP",
    )
    noyear_top6 = plot_shap_outputs(
        noyear_shap,
        DRIVER_FEATURES,
        selected_model,
        noyear_fig,
        noyear_supp,
    )
    noyear_detail, noyear_rank = compare_shap_contributions(main_contrib, noyear_shap["contribution"], DRIVER_FEATURES, "Main vs No-Year")
    noyear_detail.to_csv(noyear_dir / "NoYear_vs_Main_SHAP_contribution_difference.csv", index=False, encoding="utf-8-sig")
    noyear_rank.to_csv(noyear_dir / "NoYear_vs_Main_SHAP_rank_stability.csv", index=False, encoding="utf-8-sig")
    plot_shap_contribution_comparison(main_contrib, noyear_shap["contribution"], DRIVER_FEATURES, noyear_fig, f"{selected_model}: Main vs No-Year SHAP contribution", "NoYear_vs_Main_SHAP_Contribution")

    # 为直接比较主模型结论，No-Year ALE 固定比较主分析 Top-6（而非只比较新的 Top-6）。
    noyear_ale = run_ale_25models(
        df=df,
        features=noyear_features,
        top_features=main_top6,
        selected_model=selected_model,
        outer_tasks=main_tasks,
        model_results=noyear_results,
        output_folder=noyear_ale_dir,
        logger=logger,
        progress_label="02 No-Year | ALE",
    )
    plot_ale_outputs(noyear_ale, main_top6, selected_model, noyear_fig, noyear_supp, prefix="NoYear_")
    plot_ale_main_vs_sensitivity(main_ale_summary, noyear_ale["summary"], main_top6, noyear_fig, f"{selected_model}: Main vs No-Year ALE", "NoYear_vs_Main_ALE_Combo")

    # 主/无 Year 转折稳定性并排输出。
    noyear_stab = noyear_ale["threshold_stability"].copy(); noyear_stab["setting"] = "No-Year"
    main_stab = main_ale_stability.copy(); main_stab["setting"] = "Main"
    pd.concat([main_stab, noyear_stab], ignore_index=True).to_csv(noyear_dir / "NoYear_vs_Main_ALE_threshold_stability.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # B. 100 km Block CV 敏感性
    # =========================================================================
    logger.info("开始敏感性 B：100 km Block CV")
    block_split_dir = ensure_dir(block_dir / "split_manifest")
    block_cv_dir = ensure_dir(block_dir / "nested_cv")
    block_fig = ensure_dir(fig_dir / "100km")

    block_tasks, block_assignment, block_group_manifest = build_repeated_group_splits(
        df,
        group_column=BLOCK100_COLUMN,
        seeds=REPEAT_SEEDS,
        n_splits=OUTER_N_SPLITS,
    )
    block_assignment.to_csv(block_split_dir / "Block100_outer_split_assignment_by_observation.csv", index=False, encoding="utf-8-sig")
    block_group_manifest.to_csv(block_split_dir / "Block100_outer_split_manifest_by_group.csv", index=False, encoding="utf-8-sig")
    validate_outer_tasks(block_tasks, df, BLOCK100_COLUMN).to_csv(block_split_dir / "Block100_split_counts_and_leakage_check.csv", index=False, encoding="utf-8-sig")

    block_results = run_repeated_nested_cv(
        df=df,
        features=MAIN_MODEL_FEATURES,
        group_column=BLOCK100_COLUMN,
        outer_tasks=block_tasks,
        model_names=[selected_model],
        output_folder=block_cv_dir,
        checkpoint_folder=logs_dir / "Block100",
        sqlite_db=logs_dir / "optuna_Block100.sqlite3",
        study_prefix=f"BLOCK100_{selected_model}",
        logger=logger,
        progress_label="02 100km BlockCV | Nested CV",
    )
    block_summary = selected_only_summary(block_results)
    save_module_summary(block_summary, block_dir, "Block100")
    export_best_params(block_results, block_cv_dir, "Block100_Best_Hyperparameters_25_outer_tasks")
    block_perf = compare_main_vs_sensitivity_performance(main_results, block_results, selected_model, "100 km Block CV")
    block_perf.to_csv(block_dir / "Block100_vs_Main_performance_and_delta.csv", index=False, encoding="utf-8-sig")
    plot_performance_sensitivity_comparison(block_perf, block_fig, f"{selected_model}: Main vs 100 km Block CV", "Block100_vs_Main_Performance")

    main_r2 = float(main_selected_results["Test_R2"].mean())
    block_r2 = float(block_results["Test_R2"].mean())
    pd.DataFrame([{
        "Model": selected_model,
        "Main_patch_id1_Test_R2_mean": main_r2,
        "Block100_Test_R2_mean": block_r2,
        "Delta_R2_Main_minus_100km": main_r2 - block_r2,
        "Main_group": GROUP_COLUMN,
        "Sensitivity_group": BLOCK100_COLUMN,
        "n_Block100": int(df[BLOCK100_COLUMN].nunique()),
    }]).to_csv(block_dir / "Block100_key_result_Delta_R2.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # C. Tmean + DTR 温度重参数化敏感性
    # =========================================================================
    logger.info("开始敏感性 C：Tmean + DTR")
    df_temp = df.copy()
    df_temp["Tmean"] = (df_temp["TMX"] + df_temp["TMN"]) / 2.0
    df_temp["DTR"] = df_temp["TMX"] - df_temp["TMN"]
    temp_drivers = ["SM", "PRE", "Tmean", "DTR", "SR", "VPD", "GI"]
    temp_features = temp_drivers + [YEAR_COLUMN]

    temp_cv_dir = ensure_dir(temp_dir / "nested_cv")
    temp_shap_dir = ensure_dir(temp_dir / "oof_shap")
    temp_ale_dir = ensure_dir(temp_dir / "ALE")
    temp_fig = ensure_dir(fig_dir / "temperature")
    temp_supp = ensure_dir(supp_fig_dir / "temperature")

    # 使用与主分析相同的 patch_id1 外层测试样本。
    temp_tasks = load_outer_tasks_from_assignment(df_temp, main_assignment, GROUP_COLUMN)
    temp_results = run_repeated_nested_cv(
        df=df_temp,
        features=temp_features,
        group_column=GROUP_COLUMN,
        outer_tasks=temp_tasks,
        model_names=[selected_model],
        output_folder=temp_cv_dir,
        checkpoint_folder=logs_dir / "temperature",
        sqlite_db=logs_dir / "optuna_temperature.sqlite3",
        study_prefix=f"TEMP_TMEAN_DTR_{selected_model}",
        logger=logger,
        progress_label="02 Tmean+DTR | Nested CV",
    )
    temp_summary = selected_only_summary(temp_results)
    save_module_summary(temp_summary, temp_dir, "Temperature_Tmean_DTR")
    export_best_params(temp_results, temp_cv_dir, "Temperature_Best_Hyperparameters_25_outer_tasks")
    temp_perf = compare_main_vs_sensitivity_performance(main_results, temp_results, selected_model, "Tmean + DTR")
    temp_perf.to_csv(temp_dir / "Temperature_vs_Main_performance.csv", index=False, encoding="utf-8-sig")
    plot_performance_sensitivity_comparison(temp_perf, temp_fig, f"{selected_model}: TMX+TMN vs Tmean+DTR", "Temperature_vs_Main_Performance")

    temp_shap = run_oof_shap(
        df=df_temp,
        features=temp_features,
        driver_features=temp_drivers,
        selected_model=selected_model,
        outer_tasks=temp_tasks,
        model_results=temp_results,
        output_folder=temp_shap_dir,
        logger=logger,
        group_column_for_training=GROUP_COLUMN,
        progress_label="02 Tmean+DTR | OOF-SHAP",
    )
    temp_top6 = plot_shap_outputs(temp_shap, temp_drivers, selected_model, temp_fig, temp_supp)

    # 温度组贡献：主模型 C_TMX + C_TMN；敏感性 C_Tmean + C_DTR。
    main_c = main_contrib.set_index("feature")["contribution_percent"]
    temp_c = temp_shap["contribution"].set_index("feature")["contribution_percent"]
    temp_group_table = pd.DataFrame([{
        "Model": selected_model,
        "Main_temperature_definition": "TMX + TMN",
        "Main_temperature_group_contribution_percent": float(main_c.get("TMX", 0.0) + main_c.get("TMN", 0.0)),
        "Sensitivity_temperature_definition": "Tmean + DTR",
        "Sensitivity_temperature_group_contribution_percent": float(temp_c.get("Tmean", 0.0) + temp_c.get("DTR", 0.0)),
        "Difference_sensitivity_minus_main": float((temp_c.get("Tmean", 0.0) + temp_c.get("DTR", 0.0)) - (main_c.get("TMX", 0.0) + main_c.get("TMN", 0.0))),
        "interpretation_note": "Group contribution comparison is a robustness diagnostic, not an independent causal decomposition.",
    }])
    temp_group_table.to_csv(temp_dir / "Temperature_group_contribution_comparison.csv", index=False, encoding="utf-8-sig")

    # 非温度变量比较，避免 TMX/TMN 与 Tmean/DTR 名称不一致影响 Spearman。
    non_temp = ["SM", "PRE", "SR", "VPD", "GI"]
    temp_detail, temp_rank = compare_shap_contributions(main_contrib, temp_shap["contribution"], non_temp, "Main TMX+TMN vs Tmean+DTR: non-temperature drivers")
    temp_detail.to_csv(temp_dir / "Temperature_nonTemperature_SHAP_contribution_difference.csv", index=False, encoding="utf-8-sig")
    temp_rank.to_csv(temp_dir / "Temperature_nonTemperature_SHAP_rank_stability.csv", index=False, encoding="utf-8-sig")

    # 图：非温度变量贡献对照 + 温度组总贡献。
    plot_shap_contribution_comparison(main_contrib, temp_shap["contribution"], non_temp, temp_fig, f"{selected_model}: non-temperature SHAP contribution", "Temperature_nonTemp_SHAP_Contribution")

    # 温度敏感性自身的 ALE（Top-6 随真实敏感性 SHAP 排名自动变化）。
    temp_ale = run_ale_25models(
        df=df_temp,
        features=temp_features,
        top_features=temp_top6,
        selected_model=selected_model,
        outer_tasks=temp_tasks,
        model_results=temp_results,
        output_folder=temp_ale_dir,
        logger=logger,
        progress_label="02 Tmean+DTR | ALE",
    )
    plot_ale_outputs(temp_ale, temp_top6, selected_model, temp_fig, temp_supp, prefix="TmeanDTR_")

    # =========================================================================
    # D. 敏感性总汇总
    # =========================================================================
    all_perf = pd.concat([noyear_perf, block_perf, temp_perf], ignore_index=True)
    all_perf.to_csv(sens_run / "Sensitivity_All_Performance_Comparisons.csv", index=False, encoding="utf-8-sig")

    noyear_rank2 = noyear_rank.copy(); noyear_rank2["module"] = "No-Year"
    temp_rank2 = temp_rank.copy(); temp_rank2["module"] = "Temperature"
    rank_summary = pd.concat([noyear_rank2, temp_rank2], ignore_index=True)

    with pd.ExcelWriter(sens_run / "Sensitivity_Summary.xlsx", engine="openpyxl") as writer:
        all_perf.to_excel(writer, sheet_name="Performance", index=False)
        noyear_detail.to_excel(writer, sheet_name="NoYear_SHAP_detail", index=False)
        noyear_rank.to_excel(writer, sheet_name="NoYear_rank", index=False)
        temp_group_table.to_excel(writer, sheet_name="Temperature_group", index=False)
        temp_detail.to_excel(writer, sheet_name="Temperature_nonTemp", index=False)
        temp_rank.to_excel(writer, sheet_name="Temperature_rank", index=False)
        block_perf.to_excel(writer, sheet_name="Block100", index=False)

    run_end = datetime.now()
    record = {
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "duration_hours": (run_end - run_start).total_seconds()/3600.0,
        "linked_main_run": str(main_run),
        "selected_model": selected_model,
        "noYear_complete": True,
        "block100_complete": True,
        "temperature_complete": True,
        "main_R2_mean": main_r2,
        "block100_R2_mean": block_r2,
        "Delta_R2_main_minus_100km": main_r2 - block_r2,
        "NoYear_Spearman_rank_rho": float(noyear_rank.iloc[0]["Spearman_rank_rho"]),
        "NoYear_Top4_overlap_rate": float(noyear_rank.iloc[0]["Top4_overlap_rate"]),
        "Temperature_nonTemp_Spearman_rank_rho": float(temp_rank.iloc[0]["Spearman_rank_rho"]),
        "Temperature_main_group_contribution": float(temp_group_table.iloc[0]["Main_temperature_group_contribution_percent"]),
        "Temperature_sensitivity_group_contribution": float(temp_group_table.iloc[0]["Sensitivity_temperature_group_contribution_percent"]),
    }
    safe_json_dump(record, config_dir / "sensitivity_run_record.json")
    pd.DataFrame([record]).to_csv(config_dir / "sensitivity_run_record.csv", index=False, encoding="utf-8-sig")
    save_output_manifest(sens_run)
    mark_run_complete(
        sens_run,
        OUTPUT_ROOT_FOLDER,
        latest_pointer_name=LATEST_SENS_POINTER,
        active_pointer_name=ACTIVE_SENS_POINTER,
    )

    logger.info("=" * 90)
    logger.info("三个敏感性分析全部完成。")
    logger.info(f"最终模型：{selected_model}")
    logger.info(f"100 km ΔR² (Main - 100km) = {main_r2 - block_r2:.4f}")
    logger.info(f"输出目录：{sens_run}")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
