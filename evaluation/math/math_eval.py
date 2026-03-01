"""
Math Evaluation using lighteval.

This module provides functions to evaluate merged models on math tasks
using lighteval with custom task definitions.
"""
import os
import subprocess
from typing import List, Optional
from dataclasses import dataclass


# Math evaluation configuration
MATH_CONDA_ENV = "sober"

# Path resolution
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(_THIS_DIR)
# Path to lighteval_tasks.py (local copy in math/)
LIGHTEVAL_TASKS_PATH = os.path.join(_THIS_DIR, "lighteval_tasks.py")
# Path to math_main.py (lighteval runner script)
MAIN_PY_PATH = os.path.join(_THIS_DIR, "math_main.py")
# Default result directory (absolute path)
DEFAULT_RESULT_DIR = os.path.join(EVAL_ROOT, "eval_results", "math")

# Available math tasks
MATH_TASKS = ["aime24_", "aime25_", "amc23", "math_500_", "minerva", "olympiadbench"]


def _resolve_path(path: str) -> str:
    """
    Resolve a path to absolute, handling both relative and absolute paths.

    Handles:
    - Absolute paths: /path/to/model -> /path/to/model
    - Relative paths: ./model or ../model -> resolved from EVAL_ROOT
    - HuggingFace IDs: org/model -> kept as-is (not treated as path)
    """
    if os.path.isabs(path):
        return path

    # Check if it's a HuggingFace ID (contains "/" but not starting with "./" or "../")
    if "/" in path and not path.startswith("./") and not path.startswith("../"):
        # Likely a HuggingFace ID like "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        return path

    # Otherwise, treat as relative path and resolve from EVAL_ROOT
    return os.path.abspath(os.path.join(EVAL_ROOT, path))


@dataclass
class MathEvalConfig:
    """Configuration for math evaluation."""
    model_path: str             # HuggingFace ID or local path
    tasks: List[str]            # Math tasks to run
    output_dir: str             # Output directory for results
    temperature: float = 0.8    # Sampling temperature (Sober Reasoning default)
    top_p: float = 0.9          # Top-p sampling (Sober Reasoning default)
    max_new_tokens: int = 32768 # Max new tokens
    max_model_length: int = 32768  # Max model length


def run_math_eval(
    model_path: str,
    tasks: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 32768,
    max_model_length: int = 32768,
    conda_env: str = MATH_CONDA_ENV,
    verbose: bool = True,
) -> bool:
    """
    Run math evaluation on a model using lighteval.

    Args:
        model_path: HuggingFace model ID or local path
        tasks: List of math tasks (default: all)
        output_dir: Output directory for results (default: eval_results/math/<model_name>)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum new tokens to generate
        max_model_length: Maximum model context length
        conda_env: Conda environment name
        verbose: Print detailed output

    Returns:
        True if evaluation completed successfully, False otherwise

    Example:
        >>> run_math_eval(
        ...     model_path="./merged_models/math_merge_sign_base_merge_1.5B",
        ...     tasks=["aime24_", "math_500_"]
        ... )
    """
    if tasks is None:
        tasks = MATH_TASKS

    # Resolve model path
    model_path_resolved = _resolve_path(model_path)

    # Generate model name for output directory
    model_name = os.path.basename(model_path.rstrip('/'))

    # Default output directory: eval_results/math/<model_name>
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_RESULT_DIR, model_name)
    else:
        output_dir = _resolve_path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Math Evaluation: {model_name}")
        print(f"{'='*60}")
        print(f"  Model Path: {model_path_resolved}")
        print(f"  Tasks: {', '.join(tasks)}")
        print(f"  Output Dir: {output_dir}")
        print(f"  Temperature: {temperature}")
        print(f"  Top-p: {top_p}")
        print(f"  Custom Tasks: {LIGHTEVAL_TASKS_PATH}")
        print(f"{'='*60}")

    tasks_str = ",".join(tasks)

    # Run from EVAL_ROOT and use local math_main.py (lighteval runner)
    eval_cmd = f"""
cd {EVAL_ROOT}
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate {conda_env}
python {MAIN_PY_PATH} \\
    --model "{model_path_resolved}" \\
    --task "{tasks_str}" \\
    --temperature {temperature} \\
    --top_p {top_p} \\
    --output_dir "{output_dir}" \\
    --max_new_tokens {max_new_tokens} \\
    --max_model_length {max_model_length} \\
    --custom_tasks_directory "{LIGHTEVAL_TASKS_PATH}" \\
    --use_chat_template
"""

    if verbose:
        print("\nRunning math evaluation...")
        print(f"Command:\n{eval_cmd}")

    try:
        result = subprocess.run(
            eval_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=not verbose,
            text=True
        )
        if result.returncode != 0:
            print(f"[ERROR] Math evaluation failed with return code {result.returncode}")
            if not verbose and result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run math evaluation: {e}")
        return False

    if verbose:
        print(f"\n[DONE] Evaluation complete!")
        print(f"Results saved to: {output_dir}")

    return True


def run_math_eval_batch(
    configs: List[MathEvalConfig],
    conda_env: str = MATH_CONDA_ENV,
    verbose: bool = True,
) -> List[bool]:
    """
    Run math evaluation on multiple models.

    Args:
        configs: List of MathEvalConfig objects
        conda_env: Conda environment name
        verbose: Print detailed output

    Returns:
        List of success/failure status for each model
    """
    results = []

    for i, config in enumerate(configs, 1):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# Batch Progress: {i}/{len(configs)}")
            print(f"{'#'*60}")

        success = run_math_eval(
            model_path=config.model_path,
            tasks=config.tasks,
            output_dir=config.output_dir,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            max_model_length=config.max_model_length,
            conda_env=conda_env,
            verbose=verbose,
        )
        results.append(success)

    return results


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Math Evaluation using lighteval",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path"
    )

    # Evaluation options
    parser.add_argument(
        "--tasks",
        type=str,
        default="aime24_,aime25_,amc23,math_500_,minerva,olympiadbench",
        help="Comma-separated list of math tasks (default: all tasks)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: eval_results/math/<model_name>)"
    )

    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8, Sober Reasoning)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9, Sober Reasoning)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum new tokens to generate (default: 32768)"
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=32768,
        help="Maximum model context length (default: 32768)"
    )

    # Other options
    parser.add_argument(
        "--conda_env",
        type=str,
        default=MATH_CONDA_ENV,
        help=f"Conda environment name (default: {MATH_CONDA_ENV})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output"
    )

    args = parser.parse_args()

    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Validate tasks
    for task in tasks:
        if task not in MATH_TASKS:
            print(f"Warning: Unknown math task '{task}'. Available: {MATH_TASKS}")

    verbose = args.verbose and not args.quiet

    # Run evaluation
    success = run_math_eval(
        model_path=args.model_path,
        tasks=tasks,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_model_length=args.max_model_length,
        conda_env=args.conda_env,
        verbose=verbose,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)
