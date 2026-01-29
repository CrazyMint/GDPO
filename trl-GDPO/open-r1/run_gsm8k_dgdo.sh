#!/bin/bash
# DGDO Training Script for GSM8K
# Dynamic Gradient-Decoupled Optimization with instability-based weighting

source env_local.sh

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/dgdo_gsm8k/config.yaml \
    --vllm_mode colocate
