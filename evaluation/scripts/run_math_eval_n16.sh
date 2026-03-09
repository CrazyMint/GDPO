#!/bin/bash
# ===== Math Benchmark Evaluation (temp=0.6, top_p=0.95) =====
# Uses updated sampling settings while keeping original sample counts:
#   - temperature=0.6, top_p=0.95
#   - AIME/AMC: 10 samples, MATH-500/Minerva/OlympiadBench: 3 samples
#   - max response length 32k
#
# Usage: bash scripts/run_math_eval_n16.sh <MODEL_PATH> [OUTPUT_DIR]

# ========== Configuration ==========
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sober}"
TASKS="${TASKS:-aime24_,aime25_,amc23,math_500_,minerva,olympiadbench}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
# Use original tasks definition (n=10 for AIME/AMC, n=3 for others)
CUSTOM_TASKS="${EVAL_ROOT}/math_eval_module/lighteval_tasks.py"
# Use vLLM v0 engine to avoid flashinfer JIT compilation issues
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
# ====================================

MODEL_PATH=${1:?"Usage: bash scripts/run_math_eval_n16.sh <MODEL_PATH> [OUTPUT_DIR]"}
OUTPUT_DIR=${2:-}

# Generate output dir if not provided
if [ -z "$OUTPUT_DIR" ]; then
    # Extract model name from path
    if [[ "$MODEL_PATH" == *"/actor/"* ]]; then
        # e.g. .../math_grpo_classic_deepseek-r1-1.5b/actor/global_step_450
        PARENT=$(echo "$MODEL_PATH" | sed 's|.*/\([^/]*/actor/.*\)|\1|' | sed 's|/actor/|_|')
    else
        PARENT=$(basename "$MODEL_PATH")
    fi
    OUTPUT_DIR="${EVAL_ROOT}/eval_results/math_n16/${PARENT}"
fi

echo "=========================================================="
echo "Math Evaluation (temp=0.6, top_p=0.95)"
echo "=========================================================="
echo "  Model: $MODEL_PATH"
echo "  Tasks: $TASKS"
echo "  Output: $OUTPUT_DIR"
echo "  Custom Tasks: $CUSTOM_TASKS"
echo "=========================================================="

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python "$EVAL_ROOT/math_eval_module/math_main.py" \
    --model "$MODEL_PATH" \
    --task "$TASKS" \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_model_length $MAX_MODEL_LENGTH \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --data_parallel_size $DATA_PARALLEL_SIZE \
    --custom_tasks_directory "$CUSTOM_TASKS" \
    --use_chat_template
