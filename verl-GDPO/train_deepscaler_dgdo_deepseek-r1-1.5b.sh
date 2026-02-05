#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DeepscaleR DGDO Training Script - DeepSeek-R1-Distill-Qwen-1.5B

# Load CUDA for Flash Attention and B200 support
module load cuda/12.4.1

source "$(dirname $0)/.env"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export ROLLOUT_TP_SIZE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Paths - can be overridden via environment variables
export DATA_DIR="${DATA_DIR:-$(dirname $0)/data/deepscaler}"
export BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek-r1-1.5B-deepscaler-DGDO}"
export CKPT_DIR="${CKPT_DIR:-./results/deepscaler_dgdo_deepseek-r1-1.5b}"

# Ray configuration
export RAY_USAGE_STATS_ENABLED=0
export RAY_TMPDIR="${RAY_TMPDIR:-$HOME/ray_tmp}"
mkdir -p "$RAY_TMPDIR"

# Clean up any existing Ray instances
ray stop 2>/dev/null || true
sleep 1

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=dgdo \
    algorithm.dgdo_beta=0.9 \
    algorithm.dgdo_epsilon=1e-6 \
    algorithm.dgdo_warmup_steps=10 \
    algorithm.dgdo_beta_start=0.5 \
    algorithm.dgdo_beta_warmup_steps=200 \
    algorithm.dgdo_min_weight=0.15 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/aime.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4000 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=4000 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=seq_reward \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=DeepscaleR_DGDO \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=auto \
    trainer.wandb_kwargs.resume=allow \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.total_epochs=7
