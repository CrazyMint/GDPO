import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate math evaluation results from JSON and Parquet files"
    )
    parser.add_argument(
        "-f",
        type=str,
        required=True,
        help="Path to the results directory containing results_*.json and details_*.parquet files",
    )
    parser.add_argument(
        "-o",
        type=str,
        default=None,
        help="Output CSV file path (default: <result_dir>/summary.csv)",
    )
    args = parser.parse_args()

    result_dir = Path(args.f)
    output_path = Path(args.o) if args.o else result_dir / "summary.csv"

    if not result_dir.exists():
        print(f"Error: Directory not found: {result_dir}")
        return

    # 定义固定的列名
    benchmarks = ["aime24", "amc23", "math_500", "minerva", "olympiadbench"]
    accuracy_cols = [f"{b}_accuracy" for b in benchmarks]
    length_cols = [f"{b}_avg_length" for b in benchmarks]
    exceed_cols = [f"{b}_exceed_rate" for b in benchmarks]

    columns = (
        ["model"]
        + accuracy_cols
        + length_cols
        + exceed_cols
        + ["avg_accuracy", "avg_length", "avg_exceed_rate"]
    )

    # --- 核心改进：加载现有进度 ---
    if output_path.exists():
        print(f"Loading existing summary from: {output_path}")
        df = pd.read_csv(output_path)
        # 确保列名一致，防止 CSV 格式过旧
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
    else:
        df = pd.DataFrame(columns=columns)

    # 处理 JSON 文件
    json_list = sorted(result_dir.rglob("results_*.json"))
    print(f"Found {len(json_list)} JSON files")

    for json_path in json_list:
        with open(json_path, "r") as f:
            result = json.load(f)
            model_name = Path(
                result["config_general"]["model_config"]["model_name"]
            ).name

            # 如果模型已存在且准确率列都满了，则跳过
            if model_name in df["model"].values:
                existing_row = df[df["model"] == model_name].iloc[0]
                if not existing_row[accuracy_cols].isnull().any():
                    continue

            if model_name not in df["model"].values:
                new_row = {"model": model_name}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            model_idx = df[df["model"] == model_name].index[0]
            results_dict = result.get("results", {})

            # 映射逻辑
            mapping = {
                "aime24_|0": ("aime24_accuracy", "avg@n:n=10"),
                "amc23|0": ("amc23_accuracy", "avg@n:n=10"),
                "math_500_|0": ("math_500_accuracy", "avg@n:n=3"),
                "minerva|0": ("minerva_accuracy", "avg@n:n=3"),
                "olympiadbench|0": ("olympiadbench_accuracy", "avg@n:n=3"),
            }

            for key, (col_name, metric) in mapping.items():
                if key in results_dict:
                    val = results_dict[key].get(metric)
                    # 只有当原始值为 NaN 时才写入，避免覆盖已转换百分比的数据
                    if pd.isnull(df.loc[model_idx, col_name]):
                        df.loc[model_idx, col_name] = round(val * 100, 2)

    # 处理 Parquet 文件
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True
    )
    MAX_LENGTH = 4096

    parquet_list = sorted(result_dir.rglob("details_*.parquet"))
    print(f"Processing {len(parquet_list)} parquet files...")

    for parquet_path in parquet_list:
        model_name = parquet_path.parent.parent.name

        benchmark = next((b for b in benchmarks if b in parquet_path.name), None)
        if not benchmark or model_name not in df["model"].values:
            continue

        model_idx = df[df["model"] == model_name].index[0]
        col_length = f"{benchmark}_avg_length"
        col_exceed = f"{benchmark}_exceed_rate"

        # --- 核心改进：跳过已存在的数据 ---
        if pd.notnull(df.loc[model_idx, col_length]) and pd.notnull(
            df.loc[model_idx, col_exceed]
        ):
            continue

        print(f"Calculating tokens for {model_name} - {benchmark}...")
        detail_df = pd.read_parquet(parquet_path)
        all_lengths = []
        exceed_count = 0
        total_count = 0

        for i in range(len(detail_df)):
            response_list = detail_df.iloc[i]["model_response"]["text"]
            for response in response_list:
                if response is None or (
                    isinstance(response, float) and np.isnan(response)
                ):
                    continue
                tokens = tokenizer.encode(str(response), add_special_tokens=False)
                length = len(tokens)
                all_lengths.append(length)
                total_count += 1
                if length > MAX_LENGTH:
                    exceed_count += 1

        if total_count > 0:
            df.loc[model_idx, col_length] = np.mean(all_lengths)
            df.loc[model_idx, col_exceed] = round(exceed_count / total_count * 100, 2)

    # --- 计算统计数据 ---
    # 确保数值类型，以便计算均值
    for col in accuracy_cols + length_cols + exceed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["avg_accuracy"] = df[accuracy_cols].mean(axis=1, skipna=True)
    df["avg_length"] = df[length_cols].mean(axis=1, skipna=True)
    df["avg_exceed_rate"] = df[exceed_cols].mean(axis=1, skipna=True)

    # 转换为百分比（仅针对尚未转换过的原始小数）
    # # 逻辑：如果最大值 <= 1.0，则认为是原始小数，需要转换
    # target_pct_cols = accuracy_cols + exceed_cols + ["avg_accuracy", "avg_exceed_rate"]
    # for col in target_pct_cols:
    #     if df[col] <= 1.0:
    #         df[col] = (df[col] * 100).round(2)
    #     else:
    #         df[col] = df[col].round(2)

    df["avg_length"] = df["avg_length"].round(2)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print(df)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
