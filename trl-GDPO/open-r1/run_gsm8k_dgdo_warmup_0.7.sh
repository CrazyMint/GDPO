#!/bin/bash
# DGDO Training Script - Beta 0.7 (more responsive)
# Features: 30-step warmup, beta=0.7 (no beta schedule), min_weight=0.15

source env_local.sh

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/dgdo_warmup_0.7_gsm8k/config.yaml \
    --vllm_mode colocate
