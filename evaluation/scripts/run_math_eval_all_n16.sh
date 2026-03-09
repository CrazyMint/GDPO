#!/bin/bash
# ===== Batch Math Evaluation (temp=0.6, top_p=0.95) =====
# Re-evaluates trained models with updated sampling settings.
# Results are saved to eval_results/math_n16/ (separate from original results).
#
# Runs up to N_GPUS models in parallel (one model per GPU).
#
# Usage: bash scripts/run_math_eval_all_n16.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/data/sxw240003/GDPO/results"
LOG_DIR="${SCRIPT_DIR}/../eval_logs_n16"
mkdir -p "$LOG_DIR"

# Available GPUs (comma-separated -> array)
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
if [ ${#GPUS[@]} -eq 0 ]; then
    GPUS=(0 1 2 3)
fi
N_GPUS=${#GPUS[@]}

# All trained models with finished checkpoints
MODELS=(
    # --- 4k models ---
    # "${RESULTS_BASE}/math_grpo_classic_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_gdpo_classic_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_mw25_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_condlen_a20_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_soft_deepseek-r1-1.5b/actor/global_step_450"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # --- 8k models ---
    "${RESULTS_BASE}/math_dgdo2_classic_8k_deepseek-r1-1.5b/actor/global_step_450"
    "${RESULTS_BASE}/math_grpo_classic_8k_deepseek-r1-1.5b/actor/global_step_450"
    "${RESULTS_BASE}/math_dgdo2_soft_8k_deepseek-r1-1.5b/actor/global_step_450"
    # --- 8k models (uncomment after training) ---
    # "${RESULTS_BASE}/math_dgdo2_classic_8k_a05_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_8k_a20_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_classic_8k_mw30_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_gdpo_classic_8k_deepseek-r1-1.5b/actor/global_step_450"
    # "${RESULTS_BASE}/math_dgdo2_soft_8k_mw25_deepseek-r1-1.5b/actor/global_step_450"
)

echo "=========================================================="
echo "Batch Math Evaluation (${#MODELS[@]} models, ${N_GPUS} GPUs)"
echo "temp=0.6, top_p=0.95"
echo "=========================================================="

PIDS=()
GPU_MODEL_MAP=()
FAILED=()

# Launch models in batches of N_GPUS
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU_IDX=$((i % N_GPUS))
    GPU_ID="${GPUS[$GPU_IDX]}"
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
        bash "$SCRIPT_DIR/run_math_eval_n16.sh" "$MODEL" \
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
echo "Results: eval_results/math_n16/"
echo "=========================================================="
