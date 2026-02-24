#!/bin/bash
# ===== BFCL v4 AST Tool Calling Evaluation =====
# Evaluates models on BFCL v4 with multiple runs each.
# Split across 4 GPUs for parallelism.
# Models are PRE-REGISTERED in model_config.py and supported_models.py.
# DO NOT use --force_reregister: parallel workers corrupt shared config files.
#
# Usage: bash scripts/run_tool_eval_v4.sh <GPU_ID>
#   GPU_ID: 0, 1, 2, or 3 (each runs a subset of models)
#
# Run all 4 in parallel:
#   for gpu in 0 1 2 3; do bash scripts/run_tool_eval_v4.sh $gpu & done; wait

# ========== Configuration ==========
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_BASE="${CKPT_BASE:-/data/sxw240003/GDPO/results}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/sxw240003/GDPO/eval_results/toolrl_dgdo2}"
NUM_RUNS="${NUM_RUNS:-5}"
# ====================================

GPU_ID=${1:?"Usage: bash scripts/run_tool_eval_v4.sh <GPU_ID: 0|1|2|3>"}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export LOCAL_SERVER_PORT=$((1054 + GPU_ID))

CKPT_ROOT="$CKPT_BASE"

if [ "$GPU_ID" -eq 0 ]; then
    MODEL_LIST=($CKPT_ROOT/toolrl_grpo_static82_qwen2.5-1.5b/actor/global_step_100)
elif [ "$GPU_ID" -eq 1 ]; then
    MODEL_LIST=($CKPT_ROOT/toolrl_grpo_static64_qwen2.5-1.5b/actor/global_step_100)
elif [ "$GPU_ID" -eq 2 ]; then
    MODEL_LIST=($CKPT_ROOT/toolrl_gdpo_static64_qwen2.5-1.5b/actor/global_step_100)
elif [ "$GPU_ID" -eq 3 ]; then
    MODEL_LIST=($CKPT_ROOT/toolrl_gdpo_static37_qwen2.5-1.5b/actor/global_step_100)
else
    echo "Invalid GPU_ID: $GPU_ID (must be 0, 1, 2, or 3)"
    exit 1
fi

for path in "${MODEL_LIST[@]}"; do
    # Build a name that contains "tool" so the tool eval branch triggers
    step_num=$(basename "$path" | sed 's/global_step_//')
    variant=$(echo "$path" | grep -oP 'toolrl_[^/]+' | sed 's/_qwen2.5-1.5b//')
    name="${variant}_step_${step_num}"
    echo "=========================================================="
    echo "[GPU $GPU_ID] Processing Model: $name"
    echo "=========================================================="

    RUN_IDS=(1 2 3 4 5)
    if [[ "$name" == *"tool"* ]]; then
        for run_id in "${RUN_IDS[@]}"; do
            echo "  >>> [Tool Run $run_id]"

            output_path="$OUTPUT_DIR/tool/run_$run_id/$name"

            python "$EVAL_ROOT/bfcl_v4/tool_eval.py" \
                --model_path "$path" \
                --model_id "$name" \
                --output_dir "$output_path"

            echo "  Finished Run $run_id"
        done
    fi

    if [[ "$name" == *"math"* ]]; then
        echo "  >>> [Math Evaluation]"

        output_path="$OUTPUT_DIR/math/$name"

        python "$EVAL_ROOT/math/math_eval.py" \
            --model_path "$path" \
            --output_dir "$output_path"

        echo "  Finished Math Eval"
    fi

    echo -e "Done with $name\n"
done

echo "[GPU $GPU_ID] All models finished!"
