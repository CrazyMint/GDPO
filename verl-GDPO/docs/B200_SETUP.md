# GDPO/DGDO Training on NVIDIA B200 GPUs

This guide covers setup and optimization for training on NVIDIA B200 (Blackwell) GPUs.

## Hardware Requirements

- **GPUs**: 4x NVIDIA B200 (192GB HBM3e each)
- **System RAM**: 256GB+ recommended
- **Storage**: NVMe SSD for fast data loading
- **CUDA**: 12.4+ required for Blackwell architecture

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n gdpo python=3.10 -y
conda activate gdpo
```

### 2. Install PyTorch (B200/Blackwell Support)

B200 requires PyTorch 2.4+ with CUDA 12.4:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install GDPO Dependencies

```bash
cd verl-GDPO
pip install -e .

# Flash Attention 2 (required)
pip install flash-attn --no-build-isolation

# vLLM for fast generation
pip install vllm>=0.6.0
```

### 4. Configure Environment Variables

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"
export HF_HOME="/path/to/huggingface/cache"
```

### 5. Download Dataset

```bash
bash scripts/data/download_deepscaler.sh
```

### 6. Download Models

```bash
# DeepSeek models
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Qwen models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen3-4B
```

## B200 Memory Optimization

With 192GB HBM3e per GPU (vs 80GB on A100), B200 allows significantly larger batch sizes:

| Model Size | train_batch_size | ppo_mini_batch_size | gpu_memory_utilization |
|------------|------------------|---------------------|------------------------|
| 1.5B       | 1024             | 128                 | 0.85                   |
| 4B         | 768              | 96                  | 0.80                   |
| 7B         | 512              | 64                  | 0.75                   |

### Key Parameters to Adjust

1. **gpu_memory_utilization**: Increase from default 0.65 to 0.80-0.85
2. **train_batch_size**: Can be doubled or more compared to A100
3. **FSDP offloading**: Likely not needed for 1.5B/4B models on B200

## Running Training

### Option 1: Use Existing Scripts with Overrides

```bash
source .env

# Override batch sizes for B200
export CKPT_DIR="./results/b200_gdpo_1.5b"
bash train_deepscaler_gdpo_deepseek-r1-1.5b.sh
```

### Option 2: Create Custom B200 Script

See `train_deepscaler_gdpo_b200.sh` for a B200-optimized template.

## Expected Performance

B200 provides approximately:
- **2x faster generation** compared to A100 (higher memory bandwidth)
- **2-4x larger batch sizes** due to increased VRAM
- **Overall 3-4x faster training** per step

### Estimated Training Time (1.5B model, 500 steps)

| Hardware      | Batch Size | Time per Step | Total Time |
|---------------|------------|---------------|------------|
| 4x A100 80GB  | 512        | ~45 min       | ~375 hours |
| 4x B200 192GB | 1024       | ~15 min       | ~125 hours |

## Troubleshooting

### vLLM Out of Memory

If vLLM runs out of memory during generation:

1. Reduce `gpu_memory_utilization` (try 0.70, 0.65)
2. Reduce `train_batch_size`
3. Reduce `max_response_length`

### Slow Generation

If generation is slower than expected:

1. Increase `gpu_memory_utilization` for larger KV cache
2. Check CUDA version is 12.4+
3. Verify vLLM version supports Blackwell

### CUDA Version Issues

B200 requires CUDA 12.4+. Check with:
```bash
nvidia-smi
nvcc --version
```

## Monitoring

### GPU Utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Detailed stats
nvidia-smi dmon -s u
```

### Training Metrics

Check WandB dashboard for:
- `timing/generation_time` - Should be faster on B200
- `timing/update_time` - Training step time
- GPU memory utilization

## Multi-Node Training

For multi-node B200 clusters:

```bash
# Edit training script
export N_GPUS=4
trainer.nnodes=2  # For 2 nodes
trainer.n_gpus_per_node=4
```

Ensure proper network configuration for NCCL communication.
