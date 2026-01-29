#!/bin/bash
# DGDO Training Script - Warmup Variant
# Features: 30-step warmup, beta schedule (0.5→0.9), min_weight=0.15

source env_local.sh

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/dgdo_warmup_gsm8k/config.yaml \
    --vllm_mode colocate
