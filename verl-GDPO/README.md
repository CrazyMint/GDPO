# GDPO/DGDO - Group Reward-Decoupled Policy Optimization

Implementation of GDPO and DGDO (Dynamic Gradient-Decoupled Optimization) for multi-reward reinforcement learning, built on the [veRL](https://github.com/volcengine/verl) framework.

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd verl-GDPO

# 2. Install dependencies
pip install -e .
pip install flash-attn --no-build-isolation

# 3. Configure environment
cp .env.example .env
# Edit .env with your WANDB_API_KEY, HF_TOKEN, HF_HOME

# 4. Download dataset
bash scripts/data/download_deepscaler.sh

# 5. Run training
bash train_deepscaler_gdpo_deepseek-r1-1.5b.sh
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12.1+ (12.4+ for B200 GPUs)
- 4+ GPUs with 40GB+ VRAM each

### Install PyTorch and vLLM

```bash
# PyTorch (adjust CUDA version as needed)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# vLLM for fast generation
pip install vllm==0.6.3

# Ray for distributed training
pip install ray
```

### Install GDPO Package

```bash
cd verl-GDPO
pip install -e .

# Flash Attention 2 (required)
pip install flash-attn --no-build-isolation
```

## Dataset Setup

### Option 1: Automatic Download

```bash
bash scripts/data/download_deepscaler.sh
```

### Option 2: Manual Setup

```bash
# Clone DeepScaleR repository
git clone https://github.com/agentica-project/deepscaler.git
pip install -e deepscaler/

# Generate datasets
python scripts/data/prepare_deepscaler.py --local_dir ./data/deepscaler
```

## Environment Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"
export HF_HOME="/path/to/huggingface/cache"
```

## Training

### GDPO Training

```bash
# DeepSeek-R1-1.5B
bash train_deepscaler_gdpo_deepseek-r1-1.5b.sh

# DeepSeek-R1-7B
bash train_deepscaler_gdpo_deepseek-r1-7b.sh

# Qwen3-4B
bash train_deepscaler_gdpo_qwen3-4b.sh
```

### DGDO Training (with Dynamic Weighting)

```bash
# DeepSeek-R1-1.5B
bash train_deepscaler_dgdo_deepseek-r1-1.5b.sh

# DeepSeek-R1-7B
bash train_deepscaler_dgdo_deepseek-r1-7b.sh
```

### Custom Paths

Override default paths via environment variables:

```bash
export DATA_DIR="/custom/path/to/data"
export CKPT_DIR="/custom/path/to/checkpoints"
bash train_deepscaler_gdpo_deepseek-r1-1.5b.sh
```

## Hardware Configurations

### 4x A100 (80GB)

Default configurations in training scripts are tuned for 4x A100:

| Model | train_batch_size | ppo_mini_batch_size | gpu_memory_utilization |
|-------|------------------|---------------------|------------------------|
| 1.5B  | 512              | 64                  | 0.65                   |
| 4B    | 384              | 48                  | 0.55                   |
| 7B    | 256              | 32                  | 0.50                   |

### 4x B200 (192GB)

For B200 GPUs, use larger batch sizes. See [docs/B200_SETUP.md](docs/B200_SETUP.md) for detailed instructions.

```bash
bash train_deepscaler_gdpo_b200.sh
```

| Model | train_batch_size | ppo_mini_batch_size | gpu_memory_utilization |
|-------|------------------|---------------------|------------------------|
| 1.5B  | 1024             | 128                 | 0.85                   |
| 4B    | 768              | 96                  | 0.80                   |
| 7B    | 512              | 64                  | 0.75                   |

## Algorithm Details

### GDPO (Group Reward-Decoupled Policy Optimization)

GDPO handles multiple reward signals by decoupling their normalization, preserving relative differences for accurate multi-reward optimization.

See implementation in [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L175).

### DGDO (Dynamic Gradient-Decoupled Optimization)

DGDO extends GDPO with instability-based dynamic weighting:

- EMA smoothing with configurable beta schedule
- Warmup period for stable initialization
- Min weight constraints and importance priors

Key parameters:
- `algorithm.dgdo_beta=0.9` - EMA smoothing factor
- `algorithm.dgdo_warmup_steps=30` - Steps before dynamic weighting
- `algorithm.dgdo_beta_start=0.5` - Initial beta during warmup
- `algorithm.dgdo_min_weight=0.15` - Minimum weight constraint

## Monitoring

Training metrics are logged to Weights & Biases:

- `train/reward` - Average reward
- `train/advantage` - Advantage estimates
- `dgdo/effective_beta` - Current beta value (DGDO only)
- `timing/generation_time` - vLLM generation time per step

## Troubleshooting

### vLLM Out of Memory

Reduce `gpu_memory_utilization` or `train_batch_size`:

```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.55
data.train_batch_size=256
```

### Slow Training

Generation time dominates. To speed up:

1. Reduce `max_response_length` (e.g., 4000 → 3000)
2. Reduce `train_batch_size` for faster iterations
3. Increase `gpu_memory_utilization` for larger KV cache

### Config Errors

Ensure all required configs exist in `verl/trainer/config/ppo_trainer.yaml`:
- `clip_ratio_low`, `clip_ratio_high`
- `filter_groups.enable`, `filter_groups.metric`
- `val_kwargs.n`, `val_kwargs.do_sample`, `val_kwargs.max_tokens`

## Citation

```bibtex
@article{gdpo2024,
  title={GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

Apache 2.0
