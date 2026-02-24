#!/bin/bash
# Format Correctness Evaluation (BFCL v4) for all 23 models (base + 22 checkpoints)
# Uses offline vLLM batch inference - no server needed
#
# Usage: bash scripts/run_format_eval_v4.sh <GPU_ID>
#   GPU_ID: 0, 1, 2, or 3 (each runs a subset of models)
#
# Run all 4 in parallel:
#   for gpu in 0 1 2 3; do bash scripts/run_format_eval_v4.sh $gpu & done; wait

# ========== Configuration ==========
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_BASE="${CKPT_BASE:-/data/sxw240003/GDPO/results}"
BASE_MODEL="${BASE_MODEL:-/data/sxw240003/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/sxw240003/GDPO/eval_results/toolrl_dgdo2/format}"
# ====================================

GPU_ID=${1:?Usage: bash scripts/run_format_eval_v4.sh <GPU_ID (0-3)>}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Define model variants per GPU (base model on GPU 0)
case $GPU_ID in
    0) VARIANTS=(base grpo gdpo dgdo dgdo2 grpo_static gdpo_static) ;;
    1) VARIANTS=(grpo_static91 grpo_static82 grpo_static73 grpo_static64 grpo_static46 grpo_static37) ;;
    2) VARIANTS=(grpo_static28 grpo_static19 gdpo_static91 gdpo_static82 gdpo_static73) ;;
    3) VARIANTS=(gdpo_static64 gdpo_static46 gdpo_static37 gdpo_static28 gdpo_static19) ;;
    *) echo "ERROR: GPU_ID must be 0-3, got $GPU_ID"; exit 1 ;;
esac

echo "GPU $GPU_ID: ${VARIANTS[*]}"

for variant in "${VARIANTS[@]}"; do
    if [ "$variant" == "base" ]; then
        model_path="$BASE_MODEL"
        model_id="qwen2.5-1.5b-instruct_base"
    else
        model_path="$CKPT_BASE/toolrl_${variant}_qwen2.5-1.5b/actor/global_step_100"
        model_id="toolrl_${variant}_step_100"
    fi

    if [ ! -d "$model_path" ]; then
        echo "SKIP: $model_path does not exist"
        continue
    fi

    if [ -f "$OUTPUT_DIR/${model_id}_format_summary.json" ]; then
        echo "SKIP: $model_id already evaluated"
        continue
    fi

    echo "=========================================================="
    echo "Format Eval: $model_id (GPU $GPU_ID)"
    echo "=========================================================="

    python "$EVAL_ROOT/bfcl_v4/format_eval.py" \
        --model_path "$model_path" \
        --model_id "$model_id" \
        --output_dir "$OUTPUT_DIR"

    echo "Done with $model_id"
    echo ""
done

echo "GPU $GPU_ID: All format evaluations finished!"
