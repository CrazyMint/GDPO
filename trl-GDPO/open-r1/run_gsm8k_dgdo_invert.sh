#!/bin/bash
# DGDO Training Script - Invert Mode
# Gives MORE weight to stable (low instability) rewards
# This prioritizes correctness if it's more stable than format/int_reward

source env_local.sh

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/dgdo_invert_gsm8k/config.yaml \
    --vllm_mode colocate
