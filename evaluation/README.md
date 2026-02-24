# GDPO Evaluation Module

Self-contained evaluation module for GDPO models. Supports BFCL tool calling (v3/v4) and math benchmarks.

## Directory Structure

```
GDPO/evaluation/
├── README.md
├── .gitignore
├── __init__.py
│
├── bfcl_v4/                     # BFCL v4 tool eval
│   ├── __init__.py
│   ├── bfcl_register.py         # Dynamic model registration
│   ├── tool_eval.py             # AST accuracy evaluation
│   └── format_eval.py           # Format correctness evaluation
│
├── bfcl_v3/                     # BFCL v3 tool eval
│   ├── __init__.py
│   ├── bfcl_register.py
│   ├── tool_eval.py
│   └── format_eval.py
│
├── math/                        # Math eval (lighteval-based)
│   ├── __init__.py
│   ├── math_eval.py             # Main math evaluation driver
│   ├── math_main.py             # vLLM + lighteval runner
│   ├── lighteval_tasks.py       # Custom task definitions
│   └── math_res.py              # Results aggregation
│
├── scripts/                     # Parameterized runner scripts
│   ├── run_all_eval_v3.sh       # Combined format + AST (v3)
│   ├── run_tool_eval_v3.sh      # AST accuracy only (v3)
│   ├── run_tool_eval_v4.sh      # AST accuracy only (v4)
│   ├── run_format_eval_v3.sh    # Format correctness (v3)
│   ├── run_format_eval_v4.sh    # Format correctness (v4)
│   └── run_math_eval.sh         # Math benchmarks
│
├── notebooks/
│   └── tool_res.ipynb           # Results analysis notebook
│
├── gorilla/                     # (cloned at setup, gitignored)
├── gorilla-v3/                  # (cloned at setup, gitignored)
└── eval_results/                # (output, gitignored)
```

## Setup

### 1. Clone BFCL Frameworks

```bash
cd GDPO/evaluation

# BFCL v4 (latest)
git clone https://github.com/ShishirPatil/gorilla.git gorilla

# BFCL v3 (specific commit)
git clone https://github.com/ShishirPatil/gorilla.git gorilla-v3
cd gorilla-v3 && git checkout cd9429c -- berkeley-function-call-leaderboard/ && cd ..
```

### 2. Create Conda Environments

**BFCL environment** (for tool calling evaluation):

```bash
conda create -n BFCL python=3.10 -y
conda activate BFCL
pip install -e gorilla/berkeley-function-call-leaderboard/
pip install vllm torch
```

**sober environment** (for math evaluation):

```bash
conda create -n sober python=3.10 -y
conda activate sober
pip install lighteval vllm math-verify
```

### 3. Configure Paths

All scripts use a config block at the top with environment variable overrides:

```bash
CKPT_BASE="${CKPT_BASE:-/data/sxw240003/GDPO/results}"
BASE_MODEL="${BASE_MODEL:-/data/sxw240003/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/...}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/sxw240003/GDPO/eval_results/toolrl_dgdo2_v3}"
```

Override via environment variables or edit the scripts directly.

## Evaluation Types

### BFCL AST Accuracy

Measures whether the model generates correct function calls (name + parameters) using Abstract Syntax Tree matching.

```bash
# Single model
python bfcl_v3/tool_eval.py \
    --model_path /path/to/checkpoint \
    --model_id my_model \
    --output_dir eval_results/test \
    --force_reregister

# Multi-GPU batch (v3)
for gpu in 0 1 2 3; do bash scripts/run_tool_eval_v3.sh $gpu & done; wait
```

Test categories: `live`, `non_live`, `multi_turn`

### Format Correctness

Checks whether outputs follow the RLLA `<think>/<tool_call>/<response>` structure. Uses offline vLLM batch inference (no server needed).

```bash
# Single model
python bfcl_v3/format_eval.py \
    --model_path /path/to/checkpoint \
    --model_id my_model \
    --output_dir eval_results/format

# Multi-GPU batch (v3)
for gpu in 0 1 2 3; do bash scripts/run_format_eval_v3.sh $gpu & done; wait
```

### Math Benchmarks

Evaluates on AIME24, AMC23, MATH-500, Minerva, and OlympiadBench using lighteval.

```bash
# Single model
python math/math_eval.py \
    --model_path /path/to/checkpoint \
    --output_dir eval_results/math/my_model

# Via script
bash scripts/run_math_eval.sh /path/to/checkpoint
```

Tasks: `aime24_`, `amc23`, `math_500_`, `minerva`, `olympiadbench`

## Quick Start

Run combined v3 evaluation (format + AST accuracy) across 4 GPUs:

```bash
cd GDPO/evaluation
for gpu in 0 1 2 3; do bash scripts/run_all_eval_v3.sh $gpu & done; wait
```

## Adding New Models

Edit the `VARIANTS` array in the shell scripts:

```bash
case $GPU_ID in
    0) VARIANTS=(base grpo gdpo dgdo dgdo2 grpo_static gdpo_static) ;;
    1) VARIANTS=(grpo_static91 grpo_static82 ...) ;;
    ...
esac
```

Model paths are constructed as: `$CKPT_BASE/toolrl_${variant}_qwen2.5-1.5b/actor/global_step_100`

## v3 vs v4

| Feature | BFCL v3 | BFCL v4 |
|---------|---------|---------|
| Simple categories | `simple`, `java`, `javascript` | `simple_python`, `simple_java`, `simple_javascript` |
| Version prefix | `BFCL_v3` | `BFCL_v4` |
| Gorilla directory | `gorilla-v3/` | `gorilla/` |
| Default result dir | `eval_results/tool_v3` | `eval_results/tool` |

Use v3 for consistency with the paper's reported numbers. Use v4 for the latest BFCL categories.

## Conda Environments

| Environment | Purpose | Key Packages |
|-------------|---------|-------------|
| `BFCL` | Tool calling evaluation | bfcl, vllm, torch |
| `sober` | Math evaluation | lighteval, vllm, math-verify |

## Troubleshooting

### Model registration race condition
When running multiple GPUs in parallel, `--force_reregister` can corrupt the shared BFCL config files. Pre-register all models first, or run one GPU at a time for registration.

### `config.json` not found
Ensure the model path points to a directory containing `config.json`. For RL checkpoints, this is typically `results/<run>/actor/global_step_<N>/`.

### CUDA OOM
Reduce `--gpu_memory_utilization` (default: 0.9 for tool eval, 0.6 for format eval) or use fewer concurrent GPU workers.

### BFCL not found
Ensure gorilla repos are cloned in the `evaluation/` directory and the `BFCL` conda environment is set up with `pip install -e gorilla/berkeley-function-call-leaderboard/`.
