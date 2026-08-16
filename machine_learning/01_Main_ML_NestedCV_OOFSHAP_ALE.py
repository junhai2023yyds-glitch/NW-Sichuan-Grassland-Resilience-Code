# -*- coding: utf-8 -*-
"""
01_Main_ML_NestedCV_OOFSHAP_ALE.py

川西北草地生态系统韧性驱动因素 —— 正式主分析

运行顺序：先运行本脚本；完全结束后，再运行 02_Sensitivity_Analyses.py。
本脚本调用同目录 ml_common.py，不要移动其中任一文件。

正式主分析：
1. 7 个生态驱动变量 SM、PRE、TMN、TMX、SR、VPD、GI + Year 时间协助变量；
2. OLS / 精确 RBF-SVR / RF / LightGBM / XGBoost；
3. 5 次重复 × 5 折 Nested Group CV，patch_id1 分组，种子 42–46；
4. Optuna-TPE 直接最大化内层平均 R²；
5. XGBoost Early Stopping；
6. 自动按 25 个外层任务平均 Test R² 选择最终解释模型（不预设 RF）；
7. Top-2 repeated k-fold corrected t-test (r=5,k=5) + Wilcoxon；
8. 最终模型 25 个外层模型重复 OOF-SHAP；
9. mean|SHAP|±SD、Kendall's W、格网级 1000 次 bootstrap 95% CI；
10. Year 不进入 7 个生态驱动贡献率，剔除后重新归一化到 100%；
11. Top-6 ALE，25 条外层曲线 + 2.5%–97.5%经验不确定性区间 + 转折稳定性；
12. 单图、组合图、CSV/XLSX/JSON、环境版本、文件哈希、断点续跑、Optuna SQLite。
注意：VIF/Pearson 前期筛选已由用户完成，本脚本不重复变量筛选。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_common import (
    TARGET, GROUP_COLUMN, BLOCK100_COLUMN, YEAR_COLUMN, SAMPLE_ID_COLUMN,
    DATA_FILE_PATH, OUTPUT_ROOT_PATH,
    DRIVER_FEATURES, MAIN_MODEL_FEATURES, RUN_MODELS, REPEAT_SEEDS,
    N_REPEATS, OUTER_N_SPLITS, INNER_N_SPLITS, SHAP_CLUSTER_BOOTSTRAP_N, SHAP_OUTER_PARALLEL_JOBS,
    OPTUNA_TRIALS, SVR_C_RANGE, SVR_EPSILON_RANGE, SVR_GAMMA_RANGE,
    SVR_CACHE_SIZE_MB, SVR_INNER_PARALLEL_JOBS, SVR_STUDY_TAG,
    ALE_TOP_N, ALE_N_BINS, ALE_Q_RANGE, FEATURE_UNITS,
    ensure_dir, safe_json_dump, load_model_data, input_structure_summary,
    create_or_resume_run_directory, mark_run_complete, setup_logger,
    export_software_versions, export_file_hashes, export_search_space,
    build_repeated_group_splits, validate_outer_tasks,
    run_repeated_nested_cv, summarize_model_performance, export_best_params,
    compare_top2_models, plot_model_performance, plot_selected_oof_observed_predicted,
    run_oof_shap, plot_shap_outputs, run_ale_25models, plot_ale_outputs,
    save_output_manifest,
)
# =============================================================================
# 1. GitHub 公共复现路径
# =============================================================================
# 默认目录：
#   <repo>/data/modeling_data.csv
#   <repo>/output/machine_learning/
#
# 如需使用本机其他路径，不必修改代码，可设置环境变量：
#   NW_SICHUAN_DATA_FILE
#   NW_SICHUAN_OUTPUT_ROOT
CSV_FILE_PATH = DATA_FILE_PATH
OUTPUT_ROOT_FOLDER = OUTPUT_ROOT_PATH

# 自动断点续跑：
# - 若上次主分析未完成，再次运行本脚本会续跑同一个目录；
# - 若上次已完成，会自动创建新的时间戳目录，不覆盖旧结果。
MAIN_RUN_PREFIX = "Final_ML_Main"
ACTIVE_POINTER = "active_main_run.txt"
LATEST_POINTER = "latest_main_run.txt"


# =============================================================================
# 2. 主程序
# =============================================================================
def main() -> None:
    run_start = datetime.now()
    run_dir, resumed = create_or_resume_run_directory(
        OUTPUT_ROOT_FOLDER,
        MAIN_RUN_PREFIX,
        ACTIVE_POINTER,
    )

    # 目录尽量直接对应论文正文、补充材料和复现记录。
    config_dir = ensure_dir(run_dir / "00_config_environment")
    split_dir = ensure_dir(run_dir / "01_split_manifest")
    cv_dir = ensure_dir(run_dir / "02_main_nested_cv")
    comparison_dir = ensure_dir(run_dir / "03_model_comparison")
    shap_dir = ensure_dir(run_dir / "04_oof_shap")
    ale_dir = ensure_dir(run_dir / "05_ALE")
    main_fig_dir = ensure_dir(run_dir / "06_main_figures")
    supp_fig_dir = ensure_dir(run_dir / "07_supplementary_figures")
    logs_dir = ensure_dir(run_dir / "08_logs_checkpoints")

    logger = setup_logger(logs_dir / "main_analysis.log", name="NW_Sichuan_Main_ML")
    logger.info("=" * 90)
    logger.info("川西北草地生态系统韧性机器学习正式主分析开始")
    logger.info(f"运行目录：{run_dir}")
    logger.info(f"本次是否为断点续跑：{resumed}")
    logger.info("正式模型：OLS / exact RBF-SVR / RF / LightGBM / XGBoost")
    logger.info(
        f"SVR：exact RBF；trials={OPTUNA_TRIALS['SVR']}；"
        f"C={SVR_C_RANGE[0]}–{SVR_C_RANGE[1]}；"
        f"epsilon={SVR_EPSILON_RANGE[0]}–{SVR_EPSILON_RANGE[1]}；"
        f"gamma={SVR_GAMMA_RANGE[0]}–{SVR_GAMMA_RANGE[1]}；"
        f"inner-fold parallel={SVR_INNER_PARALLEL_JOBS}。"
    )
    logger.info("最终解释模型由 25 个外层 Test R² 平均值自动决定；不预设 RF。")
    logger.info("=" * 90)

    # -------------------------------------------------------------------------
    # A. 环境、配置、搜索空间、文件哈希
    # -------------------------------------------------------------------------
    run_config = {
        "input_csv": str(CSV_FILE_PATH),
        "output_root": str(OUTPUT_ROOT_FOLDER),
        "run_dir": str(run_dir),
        "target": TARGET,
        "group_column": GROUP_COLUMN,
        "block100_column": BLOCK100_COLUMN,
        "driver_features": DRIVER_FEATURES,
        "control_feature": YEAR_COLUMN,
        "main_model_features": MAIN_MODEL_FEATURES,
        "models": RUN_MODELS,
        "repeat_seeds": REPEAT_SEEDS,
        "n_repeats": N_REPEATS,
        "outer_n_splits": OUTER_N_SPLITS,
        "inner_n_splits": INNER_N_SPLITS,
        "optuna_objective": "maximize mean inner validation R2; no SD penalty",
        "svr_method": "exact RBF-SVR (sklearn.svm.SVR, kernel=rbf); no Nystroem approximation",
        "svr_optuna_trials_per_outer_task": OPTUNA_TRIALS["SVR"],
        "svr_C_range": SVR_C_RANGE,
        "svr_epsilon_range": SVR_EPSILON_RANGE,
        "svr_gamma_range": SVR_GAMMA_RANGE,
        "svr_kernel_cache_MB_per_process": SVR_CACHE_SIZE_MB,
        "svr_inner_fold_parallel_jobs": SVR_INNER_PARALLEL_JOBS,
        "svr_study_tag": SVR_STUDY_TAG,
        "top2_primary_test": "repeated k-fold corrected t-test",
        "top2_r": N_REPEATS,
        "top2_k": OUTER_N_SPLITS,
        "top2_supplementary_test": "Wilcoxon signed-rank",
        "shap_outer_models": N_REPEATS * OUTER_N_SPLITS,
        "shap_outer_parallel_jobs": SHAP_OUTER_PARALLEL_JOBS,
        "shap_cluster_bootstrap_B": SHAP_CLUSTER_BOOTSTRAP_N,
        "shap_driver_normalization": "exclude Year, renormalize 7 ecological drivers to 100%",
        "ale_top_n": ALE_TOP_N,
        "ale_n_bins": ALE_N_BINS,
        "ale_q_range": ALE_Q_RANGE,
        "feature_units": FEATURE_UNITS,
        "vif_pearson_screening_in_this_script": False,
        "reason_no_vif_pearson": "The final 7 ecological drivers were already determined in the preceding screening stage.",
        "run_start": run_start.isoformat(),
    }
    safe_json_dump(run_config, config_dir / "run_config.json")
    export_software_versions(config_dir)
    export_search_space(cv_dir)
    export_file_hashes(
        {
            "input_csv": CSV_FILE_PATH,
            "main_script": Path(__file__),
            "common_module": Path(__file__).with_name("ml_common.py"),
        },
        config_dir,
    )

    # -------------------------------------------------------------------------
    # B. 读取最终输入表。这里只做必要运行安全检查，不重新筛变量。
    # -------------------------------------------------------------------------
    df = load_model_data(CSV_FILE_PATH, MAIN_MODEL_FEATURES, require_block100=True)
    structure = input_structure_summary(df, MAIN_MODEL_FEATURES)
    safe_json_dump(structure, config_dir / "input_structure_summary.json")
    pd.DataFrame([structure]).to_csv(config_dir / "input_structure_summary.csv", index=False, encoding="utf-8-sig")

    logger.info(
        f"输入数据：{len(df):,} 行；{df[GROUP_COLUMN].nunique()} 个 patch_id1；"
        f"{df[BLOCK100_COLUMN].nunique()} 个 Block100；"
        f"年份 {int(df[YEAR_COLUMN].min())}–{int(df[YEAR_COLUMN].max())}。"
    )
    if df[GROUP_COLUMN].nunique() != 669:
        logger.warning(f"当前 patch_id1 数为 {df[GROUP_COLUMN].nunique()}，不是预期 669；程序继续，但论文请以真实输出为准。")
    if len(df) != 24084:
        logger.warning(f"当前观测数为 {len(df)}，不是预期 24,084；程序继续，但论文请以真实输出为准。")

    # -------------------------------------------------------------------------
    # C. 5 次重复 × 5 折 patch_id1 外层划分；保存 manifest。
    # -------------------------------------------------------------------------
    assignment_file = split_dir / "outer_split_assignment_by_observation.csv"
    group_manifest_file = split_dir / "outer_split_manifest_by_group.csv"

    if assignment_file.exists() and group_manifest_file.exists():
        logger.info("发现已有 split manifest；为保证断点续跑一致性，重新按固定种子生成并核对。")

    outer_tasks, assignment_df, group_manifest_df = build_repeated_group_splits(
        df,
        group_column=GROUP_COLUMN,
        seeds=REPEAT_SEEDS,
        n_splits=OUTER_N_SPLITS,
    )
    assignment_df.to_csv(assignment_file, index=False, encoding="utf-8-sig")
    group_manifest_df.to_csv(group_manifest_file, index=False, encoding="utf-8-sig")
    split_check = validate_outer_tasks(outer_tasks, df, GROUP_COLUMN)
    split_check.to_csv(split_dir / "outer_split_counts_and_leakage_check.csv", index=False, encoding="utf-8-sig")

    if int(split_check["overlap_groups"].sum()) != 0:
        raise RuntimeError("检测到外层格网泄漏，主分析停止。")

    # -------------------------------------------------------------------------
    # D. 五模型 × 25 外层任务真正 Nested CV。
    # -------------------------------------------------------------------------
    sqlite_db = logs_dir / "optuna_main.sqlite3"
    results = run_repeated_nested_cv(
        df=df,
        features=MAIN_MODEL_FEATURES,
        group_column=GROUP_COLUMN,
        outer_tasks=outer_tasks,
        model_names=RUN_MODELS,
        output_folder=cv_dir,
        checkpoint_folder=logs_dir,
        sqlite_db=sqlite_db,
        study_prefix="MAIN_PATCH20",
        logger=logger,
        progress_label="01 Main | Nested CV",
    )

    # 完整性检查：每个模型必须 25 条。
    counts = results.groupby("Model").size().to_dict()
    for model in RUN_MODELS:
        n = int(counts.get(model, 0))
        if n != N_REPEATS * OUTER_N_SPLITS:
            raise RuntimeError(f"模型 {model} 外层结果不完整：{n}/25。")

    summary = summarize_model_performance(results)
    summary.to_csv(comparison_dir / "model_summary_mean_SD.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(comparison_dir / "model_summary_mean_SD.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="ModelSummary", index=False)
        results.to_excel(writer, sheet_name="All25Tasks", index=False)

    export_best_params(results, cv_dir, "Best_Hyperparameters_25_outer_tasks")
    plot_model_performance(summary, main_fig_dir)

    # -------------------------------------------------------------------------
    # E. 自动选择真实最优模型 + Top-2 corrected t-test（r=5,k=5）。
    # -------------------------------------------------------------------------
    selected_model = str(summary.iloc[0]["Model"])
    second_model = str(summary.iloc[1]["Model"])
    top2_test = compare_top2_models(results, summary, comparison_dir)

    selection = {
        "selected_model": selected_model,
        "selection_rule": "highest mean outer-fold Test R2 across 5 repeats × 5 folds",
        "tie_breakers": ["lower mean Test RMSE", "lower mean Test MAE", "lower mean Train-Test R2 gap"],
        "selected_model_Test_R2_mean": float(summary.iloc[0]["Test_R2_mean"]),
        "selected_model_Test_R2_SD": float(summary.iloc[0]["Test_R2_SD"]),
        "second_model": second_model,
        "second_model_Test_R2_mean": float(summary.iloc[1]["Test_R2_mean"]),
        "top2_primary_test": "repeated k-fold corrected t-test",
        "r": N_REPEATS,
        "k": OUTER_N_SPLITS,
        "top2_corrected_t_p": float(top2_test.iloc[0]["Corrected_t_two_sided_p"]),
        "top2_wilcoxon_p": float(top2_test.iloc[0]["Wilcoxon_two_sided_p"]) if pd.notna(top2_test.iloc[0]["Wilcoxon_two_sided_p"]) else None,
        "important_note": "The selected interpretation model is determined by predictive ranking, not by choosing whichever paired test is significant.",
    }
    safe_json_dump(selection, comparison_dir / "selected_model.json")
    # 第二套敏感性分析优先读取这个稳定位置。
    safe_json_dump(selection, run_dir / "selected_model_for_sensitivity.json")
    logger.info(f"主分析真实最优模型：{selected_model}；第二名：{second_model}")
    logger.info(f"Top-2 corrected t-test p={selection['top2_corrected_t_p']:.6g}")

    # 重复 OOF 预测图与平均预测表。
    plot_selected_oof_observed_predicted(cv_dir, selected_model, main_fig_dir)

    # -------------------------------------------------------------------------
    # F. 最终模型 25 个外层模型重复 OOF-SHAP。
    # -------------------------------------------------------------------------
    if selected_model == "SVR":
        logger.warning(
            "最终模型为精确 RBF-SVR。Permutation SHAP 将明显慢于 TreeSHAP；"
            "当前默认仍解释完整外层测试折，以保持重复 OOF-SHAP 设计。"
        )

    shap_result = run_oof_shap(
        df=df,
        features=MAIN_MODEL_FEATURES,
        driver_features=DRIVER_FEATURES,
        selected_model=selected_model,
        outer_tasks=outer_tasks,
        model_results=results,
        output_folder=shap_dir,
        logger=logger,
        progress_label="01 Main | OOF-SHAP",
        group_column_for_training=GROUP_COLUMN,
    )

    # Year 单独报告：参与模型，但不进入正式 7 驱动贡献率。
    fold_imp = shap_result["fold_importance"]
    year_fold = fold_imp[fold_imp["feature"] == YEAR_COLUMN].copy()
    year_summary = pd.DataFrame([{
        "Model": selected_model,
        "feature": YEAR_COLUMN,
        "role": "Temporal control / assistance variable",
        "mean_abs_SHAP_25models_mean": float(year_fold["mean_abs_SHAP"].mean()) if len(year_fold) else np.nan,
        "mean_abs_SHAP_25models_SD": float(year_fold["mean_abs_SHAP"].std(ddof=1)) if len(year_fold) > 1 else np.nan,
        "included_in_formal_7_driver_contribution": "No",
    }])
    year_summary.to_csv(shap_dir / "Year_control_SHAP_summary.csv", index=False, encoding="utf-8-sig")

    top_features = plot_shap_outputs(
        shap_result=shap_result,
        driver_features=DRIVER_FEATURES,
        model_name=selected_model,
        main_figure_folder=main_fig_dir,
        supplementary_figure_folder=supp_fig_dir,
    )
    pd.DataFrame({"rank": np.arange(1, len(top_features)+1), "feature": top_features}).to_csv(
        shap_dir / "Top6_features_for_ALE.csv", index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # G. Top-6 ALE：25 条外层模型曲线 + 95%经验不确定性 + 转折稳定性。
    # -------------------------------------------------------------------------
    ale_result = run_ale_25models(
        df=df,
        features=MAIN_MODEL_FEATURES,
        top_features=top_features,
        selected_model=selected_model,
        outer_tasks=outer_tasks,
        model_results=results,
        output_folder=ale_dir,
        logger=logger,
        progress_label="01 Main | ALE",
    )
    plot_ale_outputs(
        ale_result=ale_result,
        top_features=top_features,
        model_name=selected_model,
        main_figure_folder=main_fig_dir,
        supplementary_figure_folder=supp_fig_dir,
    )

    # -------------------------------------------------------------------------
    # H. 最终运行记录与完成标记。
    # -------------------------------------------------------------------------
    run_end = datetime.now()
    run_record = {
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "duration_hours": (run_end - run_start).total_seconds() / 3600.0,
        "input_csv": str(CSV_FILE_PATH),
        "run_dir": str(run_dir),
        "n_rows": int(len(df)),
        "n_patch_id1": int(df[GROUP_COLUMN].nunique()),
        "n_block100": int(df[BLOCK100_COLUMN].nunique()),
        "selected_model": selected_model,
        "second_model": second_model,
        "top2_corrected_t_p": selection["top2_corrected_t_p"],
        "shap_method": shap_result["shap_method"],
        "Kendalls_W_all_25": float(shap_result["kendall"].iloc[0]["Kendalls_W"]),
        "ALE_top6": top_features,
        "main_analysis_complete": True,
        "next_step": "Run 02_Sensitivity_Analyses.py",
    }
    safe_json_dump(run_record, config_dir / "run_record.json")
    pd.DataFrame([run_record]).to_csv(config_dir / "run_record.csv", index=False, encoding="utf-8-sig")
    save_output_manifest(run_dir)
    mark_run_complete(
        run_dir,
        OUTPUT_ROOT_FOLDER,
        latest_pointer_name=LATEST_POINTER,
        active_pointer_name=ACTIVE_POINTER,
    )

    logger.info("=" * 90)
    logger.info("主分析全部完成。")
    logger.info(f"真实最优模型：{selected_model}")
    logger.info(f"Kendall's W（25 套 SHAP 排名）：{run_record['Kendalls_W_all_25']:.4f}")
    logger.info(f"ALE Top-6：{top_features}")
    logger.info(f"输出目录：{run_dir}")
    logger.info("下一步：运行 02_Sensitivity_Analyses.py。")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
