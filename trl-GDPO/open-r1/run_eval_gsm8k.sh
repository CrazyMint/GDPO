#!/bin/bash
# Evaluate trained DGDO model on GSM8K test set using vLLM

source env_local.sh

# Default model path - change to your checkpoint
MODEL_PATH="${1:-/data/sxw240003/GDPO/outputs/Qwen2.5-1.5B-gsm8k-GDPO-3-rewards}"

echo "Evaluating model: $MODEL_PATH"
echo "Using vLLM for fast batched inference"

python eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --max_new_tokens 1024 \
    --batch_size 64 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.9
