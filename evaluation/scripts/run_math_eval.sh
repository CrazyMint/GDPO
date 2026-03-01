#!/bin/bash
# ===== Math Benchmark Evaluation (Sober Reasoning settings) =====
# Runs math benchmarks (AIME24, AIME25, AMC23, MATH-500, Minerva, OlympiadBench)
# using lighteval with vLLM backend.
#
# Usage: bash scripts/run_math_eval.sh <MODEL_PATH> [OUTPUT_DIR]
#
# Examples:
#   bash scripts/run_math_eval.sh /data/sxw240003/GDPO/results/toolrl_grpo_qwen2.5-1.5b/actor/global_step_100
#   bash scripts/run_math_eval.sh /path/to/model /path/to/output

# ========== Configuration ==========
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sober}"
TASKS="${TASKS:-aime24_,aime25_,amc23,math_500_,minerva,olympiadbench}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
# ====================================

MODEL_PATH=${1:?"Usage: bash scripts/run_math_eval.sh <MODEL_PATH> [OUTPUT_DIR]"}
OUTPUT_DIR=${2:-}

CMD="python \"$EVAL_ROOT/math/math_eval.py\" \
    --model_path \"$MODEL_PATH\" \
    --tasks \"$TASKS\" \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_model_length $MAX_MODEL_LENGTH \
    --conda_env \"$CONDA_ENV\""

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

echo "=========================================================="
echo "Math Evaluation"
echo "=========================================================="
echo "  Model: $MODEL_PATH"
echo "  Tasks: $TASKS"
echo "  Output: ${OUTPUT_DIR:-<auto>}"
echo "=========================================================="

eval $CMD
