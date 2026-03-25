#!/bin/bash
# MATH DGDO-Eff (Symmetric) Training Script - DeepSeek-R1-Distill-Qwen-1.5B
# Online per-problem optimal length L* from correct rollouts
# R_eff = 1 - |l - L*| / L* (symmetric: penalizes both shorter AND longer than L*)
# Standard Potential for both D_corr and D_eff

source "$(dirname $0)/.env"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export ROLLOUT_TP_SIZE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Paths
export DATA_DIR="${DATA_DIR:-$(dirname $0)/data/math}"
export BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek-r1-1.5B-math-DGDO-Eff-Sym}"
export CKPT_DIR="${CKPT_DIR:-/data/sxw240003/GDPO/results/math_dgdo_eff_sym_deepseek-r1-1.5b}"

export RAY_USAGE_STATS_ENABLED=0
export WANDB_RUN_ID=z8xa57yv
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# Correctness reward only (R_eff computed online in advantage estimator)
export DEEPSCALE_CORRECT_REWARD=1.0
export DEEPSCALE_LENGTH_REWARD=0.0
export DEEPSCALE_LENGTH_MODE=classic

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=dgdo_eff \
    algorithm.dgdo_eff_alpha=1.0 \
    algorithm.dgdo_eff_beta=0.95 \
    algorithm.dgdo_eff_epsilon=1e-8 \
    algorithm.dgdo_eff_bmax=8000 \
    algorithm.dgdo_eff_w_min=0.3 \
    algorithm.dgdo_eff_lstar_mode=max_sym \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=8000 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=8000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=MATH_GDPO \
    trainer.experiment_name=$EXPERIMENT_NAME \
    +trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.wandb_kwargs.resume=allow \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.total_epochs=7
