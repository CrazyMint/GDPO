#!/bin/bash
# ===== Batch Math Evaluation for all trained math models =====
# Evaluates all math training checkpoints using Sober Reasoning settings.
# Uses the last checkpoint (global_step_450) for each model.
#
# Runs up to N_GPUS models in parallel (one model per GPU).
#
# Usage: bash scripts/run_math_eval_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/data/sxw240003/GDPO/results"
LOG_DIR="${SCRIPT_DIR}/../eval_logs"
mkdir -p "$LOG_DIR"

# Available GPUs (comma-separated -> array)
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
if [ ${#GPUS[@]} -eq 0 ]; then
    GPUS=(0 1 2 3)
fi
N_GPUS=${#GPUS[@]}

# Trained models and their last checkpoints
MODELS=(
    # "${RESULTS_BASE}/math_grpo_classic_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_gdpo_classic_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_mw25_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_condlen_a20_deepseek-r1-1.5b/actor/global_step_450"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # --- Soft length ablations (uncomment after training) ---
    # "${RESULTS_BASE}/math_dgdo2_soft_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_soft_mw25_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_soft_a05_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_soft_a15_deepseek-r1-1.5b/actor/global_step_450"
    # --- 8k classic ablations (uncomment after training) ---
    "${RESULTS_BASE}/math_dgdo2_classic_8k_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_8k_a05_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_8k_a20_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_8k_mw30_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_gdpo_classic_8k_deepseek-r1-1.5b/actor/global_step_450"
    "${RESULTS_BASE}/math_grpo_classic_8k_deepseek-r1-1.5b/actor/global_step_450"
    # --- 8k soft ablations (uncomment after training) ---
    # "${RESULTS_BASE}/math_dgdo2_soft_8k_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_soft_8k_mw25_deepseek-r1-1.5b/actor/global_step_450"
    # --- Format reward models (uncomment after training) ---
    # "${RESULTS_BASE}/math_dgdo2_fmt_8k_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_grpo_fmt_8k_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_fmtlen_8k_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_grpo_fmtlen_8k_deepseek-r1-1.5b/actor/global_step_450"
)

echo "=========================================================="
echo "Batch Math Evaluation (${#MODELS[@]} models, ${N_GPUS} GPUs)"
echo "=========================================================="

PIDS=()
GPU_MODEL_MAP=()
FAILED=()

# Launch models in batches of N_GPUS
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU_IDX=$((i % N_GPUS))
    GPU_ID="${GPUS[$GPU_IDX]}"
    # Use experiment name (parent dirs) for unique naming
    # e.g. ".../math_grpo_classic_deepseek-r1-1.5b/actor/global_step_450" -> "math_grpo_classic_deepseek-r1-1.5b"
    # For HF IDs like "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" -> "DeepSeek-R1-Distill-Qwen-1.5B"
    if [[ "$MODEL" == *"/actor/"* ]]; then
        MODEL_SHORT="$(echo "$MODEL" | sed 's|.*/\([^/]*/actor/.*\)|\1|' | sed 's|/actor/|_|')"
    else
        MODEL_SHORT="$(basename "$MODEL")"
    fi
    LOG_FILE="${LOG_DIR}/${MODEL_SHORT}.log"

    # If all GPU slots are full, wait for current batch to finish
    if [ ${#PIDS[@]} -ge $N_GPUS ]; then
        echo ""
        echo "--- Waiting for batch to finish (${#PIDS[@]} running) ---"
        for j in "${!PIDS[@]}"; do
            wait "${PIDS[$j]}"
            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                echo "[WARN] Failed: ${GPU_MODEL_MAP[$j]} (exit=$EXIT_CODE)"
                FAILED+=("${GPU_MODEL_MAP[$j]}")
            else
                echo "[OK]   Done: ${GPU_MODEL_MAP[$j]}"
            fi
        done
        PIDS=()
        GPU_MODEL_MAP=()
    fi

    echo ""
    echo ">>> [$((i+1))/${#MODELS[@]}] GPU $GPU_ID: $MODEL_SHORT"
    echo "    Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID DATA_PARALLEL_SIZE=1 \
        bash "$SCRIPT_DIR/run_math_eval.sh" "$MODEL" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    GPU_MODEL_MAP+=("$MODEL_SHORT")
done

# Wait for last batch
if [ ${#PIDS[@]} -gt 0 ]; then
    echo ""
    echo "--- Waiting for final batch (${#PIDS[@]} running) ---"
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "[WARN] Failed: ${GPU_MODEL_MAP[$j]} (exit=$EXIT_CODE)"
            FAILED+=("${GPU_MODEL_MAP[$j]}")
        else
            echo "[OK]   Done: ${GPU_MODEL_MAP[$j]}"
        fi
    done
fi

echo ""
echo "=========================================================="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Completed with ${#FAILED[@]} failure(s):"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
else
    echo "All evaluations complete!"
fi
echo "Logs: $LOG_DIR/"
echo "Run: python math_eval_module/math_res.py -f eval_results/math/ to aggregate results."
echo "=========================================================="
