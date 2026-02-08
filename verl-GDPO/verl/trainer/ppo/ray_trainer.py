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
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean, masked_whiten



def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, dgdo_state=None):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

    elif adv_estimator == 'grpo_no_std':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_no_std_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns


    elif adv_estimator == 'gdpo':
        ## handle correctness, length, and optionally format rewards
        token_level_scores_correctness = data.batch['token_level_scores_correctness']
        token_level_scores_format = data.batch['token_level_scores_format']
        token_level_scores_length = data.batch.get('token_level_scores_length', None)

        # shared variables
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        ## handle correctness
        correctness_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_scores_correctness,
                                                                        eos_mask=response_mask,
                                                                        index=index)

        new_advantage = correctness_normalized_score

        ## handle format (only if enabled, i.e. has non-zero values)
        if token_level_scores_format.abs().sum() > 0:
            format_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_scores_format,
                                                                            eos_mask=response_mask,
                                                                            index=index)
            new_advantage = new_advantage + format_normalized_score

        ## handle length reward (only if enabled, i.e. has non-zero values)
        if token_level_scores_length is not None and token_level_scores_length.abs().sum() > 0:
            length_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_scores_length,
                                                                            eos_mask=response_mask,
                                                                            index=index)
            new_advantage = new_advantage + length_normalized_score

        advantages = masked_whiten(new_advantage, response_mask) * response_mask

        data.batch['advantages'] = advantages
        data.batch['returns'] = advantages

    elif adv_estimator == 'dgdo':
        # DGDO: Dynamic Gradient-Decoupled Optimization
        # Extends GDPO with instability-based dynamic weighting

        if dgdo_state is None:
            raise ValueError("dgdo_state must be provided for DGDO advantage estimator")

        # Get reward tensors
        token_level_scores_correctness = data.batch['token_level_scores_correctness']
        token_level_scores_format = data.batch['token_level_scores_format']
        token_level_scores_length = data.batch.get('token_level_scores_length', None)

        # Shared variables
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        device = response_mask.device

        # Normalize each reward independently
        correctness_adv, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_scores_correctness,
            eos_mask=response_mask,
            index=index)

        reward_advantages = [correctness_adv]
        reward_names = ['correctness']

        # Only include format if enabled (has non-zero values)
        if token_level_scores_format.abs().sum() > 0:
            format_adv, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_scores_format,
                eos_mask=response_mask,
                index=index)
            reward_advantages.append(format_adv)
            reward_names.append('format')

        # Only include length if enabled (has non-zero values)
        if token_level_scores_length is not None and token_level_scores_length.abs().sum() > 0:
            length_adv, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_scores_length,
                eos_mask=response_mask,
                index=index)
            reward_advantages.append(length_adv)
            reward_names.append('length')

        num_rewards = len(reward_advantages)

        # Increment step counter
        dgdo_state['step_count'] = dgdo_state.get('step_count', 0) + 1
        current_step = dgdo_state['step_count']

        dgdo_metrics = {}
        dgdo_metrics['dgdo/step'] = float(current_step)

        # WARMUP: Return uniform weights during warmup period
        dgdo_warmup_steps = dgdo_state.get('warmup_steps', 0)
        if dgdo_warmup_steps > 0 and current_step <= dgdo_warmup_steps:
            uniform_weights = torch.ones(num_rewards, device=device) / num_rewards
            dgdo_metrics['dgdo/in_warmup'] = 1.0
            dgdo_metrics['dgdo/warmup_progress'] = current_step / dgdo_warmup_steps
            for k in range(num_rewards):
                dgdo_metrics[f'dgdo/reward_{reward_names[k]}/weight'] = uniform_weights[k].item()
            entropy = -(uniform_weights * torch.log(uniform_weights + 1e-8)).sum()
            dgdo_metrics['dgdo/weight_entropy'] = entropy.item()
            dgdo_state['metrics'] = dgdo_metrics

            # Weighted combination with uniform weights
            new_advantage = sum(w * adv for w, adv in zip(uniform_weights.tolist(), reward_advantages))
            advantages = masked_whiten(new_advantage, response_mask) * response_mask
            data.batch['advantages'] = advantages
            data.batch['returns'] = advantages
            return data  # Early return during warmup

        dgdo_metrics['dgdo/in_warmup'] = 0.0

        # Compute DGDO dynamic weights
        dgdo_beta = dgdo_state.get('beta', 0.9)
        dgdo_epsilon = dgdo_state.get('epsilon', 1e-6)

        instabilities = torch.zeros(num_rewards, device=device)

        for k, adv in enumerate(reward_advantages):
            # Extract scalar scores from token-level advantages
            scores = (adv * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)

            # Group by prompt index
            unique_indices = list(set(index))
            group_means = []
            group_vars = []

            for idx in unique_indices:
                mask_indices = [i for i, x in enumerate(index) if x == idx]
                if len(mask_indices) > 0:
                    group_scores = scores[mask_indices]
                    group_means.append(group_scores.mean())
                    if len(mask_indices) > 1:
                        group_vars.append(group_scores.var(unbiased=False))
                    else:
                        group_vars.append(torch.tensor(0.0, device=device))

            group_means = torch.stack(group_means)
            group_vars = torch.stack(group_vars)

            # Law of Total Variance
            mu_batch = group_means.mean()
            mean_within_var = group_vars.mean()
            between_group_var = group_means.var(unbiased=False)
            sigma_batch = torch.sqrt(mean_within_var + between_group_var + 1e-8)

            # Coefficient of variation (instability)
            instabilities[k] = sigma_batch / (torch.abs(mu_batch) + dgdo_epsilon)

            dgdo_metrics[f'dgdo/reward_{reward_names[k]}/mu_batch'] = mu_batch.item()
            dgdo_metrics[f'dgdo/reward_{reward_names[k]}/sigma_batch'] = sigma_batch.item()
            dgdo_metrics[f'dgdo/reward_{reward_names[k]}/instability'] = instabilities[k].item()

        # Get variant options
        dgdo_invert = dgdo_state.get('invert', False)
        dgdo_min_weight = dgdo_state.get('min_weight', None)
        dgdo_importance_priors = dgdo_state.get('importance_priors', None)

        # Compute target weights (with optional invert)
        if dgdo_invert:
            # Give MORE weight to stable (low instability) rewards
            target_weights = torch.softmax(-instabilities, dim=0)
        else:
            # Original: give more weight to unstable rewards
            target_weights = torch.softmax(instabilities, dim=0)

        # Apply importance priors if provided
        if dgdo_importance_priors is not None:
            priors = torch.tensor(dgdo_importance_priors, device=device, dtype=target_weights.dtype)
            target_weights = target_weights * priors
            target_weights = target_weights / target_weights.sum()

        # Note: min_weight is applied AFTER EMA smoothing (see below)

        # Compute effective beta (with optional schedule)
        dgdo_beta_start = dgdo_state.get('beta_start', None)
        dgdo_beta_warmup_steps = dgdo_state.get('beta_warmup_steps', 0)
        if dgdo_beta_start is not None and dgdo_beta_warmup_steps > 0:
            steps_since_warmup = current_step - dgdo_warmup_steps
            progress = min(max(steps_since_warmup, 0) / dgdo_beta_warmup_steps, 1.0)
            effective_beta = dgdo_beta_start + progress * (dgdo_beta - dgdo_beta_start)
        else:
            effective_beta = dgdo_beta
        dgdo_metrics['dgdo/effective_beta'] = effective_beta

        # EMA smoothing (always blend with previous weights)
        if dgdo_state.get('weights') is None or not dgdo_state.get('initialized', False):
            # Initialize with uniform weights, then blend
            uniform_weights = torch.ones(num_rewards, device=device) / num_rewards
            dgdo_state['weights'] = (effective_beta * uniform_weights + (1 - effective_beta) * target_weights).detach().cpu()
            dgdo_state['initialized'] = True
        else:
            prev_weights = dgdo_state['weights'].to(device)
            dgdo_state['weights'] = (effective_beta * prev_weights + (1 - effective_beta) * target_weights).detach().cpu()

        current_weights = dgdo_state['weights'].to(device)

        # Apply minimum weight constraint AFTER EMA smoothing
        if dgdo_min_weight is not None and dgdo_min_weight > 0:
            current_weights = torch.clamp(current_weights, min=dgdo_min_weight)
            current_weights = current_weights / current_weights.sum()

        # Log final weights
        for k in range(num_rewards):
            dgdo_metrics[f'dgdo/reward_{reward_names[k]}/weight'] = current_weights[k].item()

        # Weight entropy (uniformity measure)
        entropy = -(current_weights * torch.log(current_weights + 1e-8)).sum()
        dgdo_metrics['dgdo/weight_entropy'] = entropy.item()

        # Store metrics for logging
        dgdo_state['metrics'] = dgdo_metrics

        # Weighted combination
        new_advantage = sum(w * adv for w, adv in zip(current_weights.tolist(), reward_advantages))
        advantages = masked_whiten(new_advantage, response_mask) * response_mask

        data.batch['advantages'] = advantages
        data.batch['returns'] = advantages

    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    
    sequence_score_format = batch.batch['token_level_scores_format'].sum(-1)
    sequence_score_correctness = batch.batch['token_level_scores_correctness'].sum(-1)
    sequence_score_length = batch.batch['token_level_scores_length'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # format score
        'critic/format_score/mean':
            torch.mean(sequence_score_format).detach().item(),
        'critic/format_score/max':
            torch.max(sequence_score_format).detach().item(),
        'critic/format_score/min':
            torch.min(sequence_score_format).detach().item(),
        # correctness score
        'critic/correctness_score/mean':
            torch.mean(sequence_score_correctness).detach().item(),
        'critic/correctness_score/max':
            torch.max(sequence_score_correctness).detach().item(),
        'critic/correctness_score/min':
            torch.min(sequence_score_correctness).detach().item(),
        # length score
        'critic/length_score/mean':
            torch.mean(sequence_score_length).detach().item(),
        'critic/length_score/max':
            torch.max(sequence_score_length).detach().item(),
        'critic/length_score/min':
            torch.min(sequence_score_length).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/var':
            torch.var(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        # DGDO state initialization
        self.dgdo_weights = None
        self.dgdo_initialized = False
        self.dgdo_beta = config.algorithm.get('dgdo_beta', 0.9)
        self.dgdo_epsilon = config.algorithm.get('dgdo_epsilon', 1e-6)
        # DGDO variant options
        self.dgdo_min_weight = config.algorithm.get('dgdo_min_weight', None)
        self.dgdo_invert = config.algorithm.get('dgdo_invert', False)
        self.dgdo_importance_priors = config.algorithm.get('dgdo_importance_priors', None)
        # DGDO warmup and beta schedule
        self.dgdo_warmup_steps = config.algorithm.get('dgdo_warmup_steps', 0)
        self.dgdo_beta_start = config.algorithm.get('dgdo_beta_start', None)
        self.dgdo_beta_warmup_steps = config.algorithm.get('dgdo_beta_warmup_steps', 0)
        self._dgdo_step_count = 0

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         # use_chat_template=self.config.data.use_chat_template,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='left')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       # use_chat_template=self.config.data.use_chat_template,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='left')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        format_tensor_lst = []
        correctness_tensor_lst = []
        length_tensor_lst = []

        data_source_lst = []
        uid_lst = []

        # Get validation sampling config from val_kwargs (following ME's approach)
        # Defaults: n=1 (single sample), do_sample=False (greedy)
        val_kwargs = self.config.actor_rollout_ref.rollout.get('val_kwargs', {})
        val_n = val_kwargs.get('n', 1)
        val_do_sample = val_kwargs.get('do_sample', False)
        val_temperature = val_kwargs.get('temperature', self.config.actor_rollout_ref.rollout.temperature)
        val_max_tokens = val_kwargs.get('max_tokens', None)  # None means use default response_length

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Generate unique uid for each prompt BEFORE repeating
            batch_size = len(test_batch)
            if 'uid' not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch['uid'] = np.array(
                    [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
                )

            # Repeat test_batch BEFORE popping for generation (following ME's approach)
            # This ensures batch sizes align after generation
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': val_do_sample,
                'validate': True,
                'temperature': val_temperature if val_do_sample else 1.0,
            }
            # Add max_tokens override for validation if specified
            if val_max_tokens is not None:
                test_gen_batch.meta_info['max_tokens'] = val_max_tokens

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Union test_batch with generated outputs (sizes now match)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor, format_tensor, correctness_tensor, length_tensor = self.val_reward_fn(test_batch, self.global_steps)

            reward_tensor_lst.append(reward_tensor)
            format_tensor_lst.append(format_tensor)
            correctness_tensor_lst.append(correctness_tensor)
            length_tensor_lst.append(length_tensor)
            # Get data_source and uid (already repeated in test_batch)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(test_batch)))
            uid_lst.append(test_batch.non_tensor_batch['uid'])

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        format_tensor = torch.cat(format_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        correctness_tensor = torch.cat(correctness_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        length_tensor = torch.cat(length_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        # Flatten lists of data sources and uids
        data_sources = [ds for batch_ds in data_source_lst for ds in batch_ds]
        sample_uids = [uid for batch_uids in uid_lst for uid in batch_uids]

        # Build infos_dict for process_validation_metrics
        infos_dict = {
            'reward': reward_tensor.tolist(),
            'correctness': correctness_tensor.tolist(),
            'format': format_tensor.tolist(),
            'length': length_tensor.tolist(),
        }

        # Compute @N metrics using process_validation_metrics
        data_src2var2metric2val = process_validation_metrics(
            data_sources=data_sources,
            sample_uids=sample_uids,
            infos_dict=infos_dict
        )

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            # Determine core variable (correctness if available, else reward)
            core_var = "correctness" if "correctness" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if not metric2val:
                    continue
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    # val-core: core metrics with mean/best/maj at max N
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Also keep the simple mean metrics for backward compatibility
        for data_source in set(data_sources):
            mask = data_sources == data_source
            metric_dict[f'val/test_score/{data_source}'] = reward_tensor[mask].mean().item()
            metric_dict[f'val/test_format/{data_source}'] = format_tensor[mask].mean().item()
            metric_dict[f'val/test_correctness/{data_source}'] = correctness_tensor[mask].mean().item()
            metric_dict[f'val/test_length/{data_source}'] = length_tensor[mask].mean().item()

        return metric_dict

    
    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.adv_estimator == 'grpo_no_std':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'gdpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'dgdo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None # if self.config.trainer.default_hdfs_dir is None else os.path.join(
            # self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None # if self.config.trainer.default_hdfs_dir is None else os.path.join(
                # self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Create progress bar for training
        pbar = tqdm(total=self.total_training_steps, initial=0, desc="Training", unit="step")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                pbar.set_description(f"Epoch {epoch}")
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, format_tensor, correctness_tensor, length_tensor = self.reward_fn(batch, self.global_steps)
                        ## reward_tensor = format_tensor + correctness_tensor + length_tensor

                        batch.batch['token_level_scores'] = reward_tensor
                        batch.batch['token_level_scores_format'] = format_tensor
                        batch.batch['token_level_scores_correctness'] = correctness_tensor
                        batch.batch['token_level_scores_length'] = length_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        # Prepare DGDO state if using DGDO
                        dgdo_state = None
                        if self.config.algorithm.adv_estimator == 'dgdo':
                            dgdo_state = {
                                'weights': self.dgdo_weights,
                                'initialized': self.dgdo_initialized,
                                'beta': self.dgdo_beta,
                                'epsilon': self.dgdo_epsilon,
                                'min_weight': self.dgdo_min_weight,
                                'invert': self.dgdo_invert,
                                'importance_priors': self.dgdo_importance_priors,
                                # Warmup and beta schedule
                                'warmup_steps': self.dgdo_warmup_steps,
                                'beta_start': self.dgdo_beta_start,
                                'beta_warmup_steps': self.dgdo_beta_warmup_steps,
                                'step_count': self._dgdo_step_count,
                            }

                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  dgdo_state=dgdo_state)

                        # Update DGDO state and log metrics
                        if self.config.algorithm.adv_estimator == 'dgdo' and dgdo_state is not None:
                            self.dgdo_weights = dgdo_state.get('weights')
                            self.dgdo_initialized = dgdo_state.get('initialized', False)
                            self._dgdo_step_count = dgdo_state.get('step_count', self._dgdo_step_count)
                            if 'metrics' in dgdo_state:
                                metrics.update(dgdo_state['metrics'])

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                pbar.update(1)

                if self.global_steps >= self.total_training_steps:
                    pbar.close()

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

        pbar.close()
