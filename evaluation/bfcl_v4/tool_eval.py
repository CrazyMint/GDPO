"""
Tool Calling Evaluation using BFCL.

This module provides functions to evaluate merged models on tool calling tasks
using the Berkeley Function-Calling Leaderboard (BFCL) framework.
"""
import os
import subprocess
import sys
from typing import List, Optional
from dataclasses import dataclass

# Handle both relative import (when used as module) and direct execution
try:
    from .bfcl_register import (
        register_model_to_bfcl,
        unregister_model_from_bfcl,
        is_model_registered,
        convert_model_id,
    )
except ImportError:
    from bfcl_register import (
        register_model_to_bfcl,
        unregister_model_from_bfcl,
        is_model_registered,
        convert_model_id,
    )


# Path resolution
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(_THIS_DIR)

# BFCL configuration
BFCL_CONDA_ENV = "BFCL"
BFCL_ROOT = os.path.join(EVAL_ROOT, "gorilla", "berkeley-function-call-leaderboard")

# Default result directory (absolute path)
DEFAULT_RESULT_DIR = os.path.join(EVAL_ROOT, "eval_results", "tool")

# Available test categories
TOOL_TEST_CATEGORIES = ["live", "non_live", "multi_turn"]


def _resolve_path(path: str) -> str:
    """Resolve a path to absolute, handling both relative and absolute paths."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(EVAL_ROOT, path))


@dataclass
class ToolEvalConfig:
    """Configuration for tool evaluation."""
    model_id: str               # Model identifier for BFCL
    model_path: str             # Local path to model
    test_categories: List[str]  # Test categories to run
    output_dir: str             # Output directory for results
    num_gpus: int = 1           # Number of GPUs to use
    gpu_memory_utilization: float = 0.9  # GPU memory utilization
    backend: str = "vllm"       # Inference backend
    force_reregister: bool = False  # Force re-registration even if already registered
    run_evaluate: bool = True   # Whether to run bfcl evaluate step


def run_tool_eval(
    model_id: str,
    model_path: str,
    test_categories: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    num_gpus: int = 1,
    gpu_memory_utilization: float = 0.9,
    conda_env: str = BFCL_CONDA_ENV,
    verbose: bool = True,
    force_reregister: bool = False,
    run_evaluate: bool = True,
) -> bool:
    """
    Run BFCL tool calling evaluation on a model.

    Args:
        model_id: Unique identifier for the model (underscores will be converted to hyphens for BFCL)
        model_path: Local path to the model checkpoint
        test_categories: List of test categories (default: all)
        output_dir: Output directory for results (default: eval_results/tool/<model_id>)
        num_gpus: Number of GPUs to use
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        conda_env: Conda environment name for BFCL
        verbose: Print detailed output
        force_reregister: Force re-registration even if model is already registered (default: False)
        run_evaluate: Whether to run bfcl evaluate step (default: True)

    Returns:
        True if evaluation completed successfully, False otherwise

    Example:
        >>> run_tool_eval(
        ...     model_id="tool_merge_sign_base_1.5B",
        ...     model_path="./merged_models/tool_merge_sign_base_merge_1.5B",
        ...     test_categories=["live", "non_live"],
        ...     force_reregister=True
        ... )
    """
    if test_categories is None:
        test_categories = TOOL_TEST_CATEGORIES

    # Convert model_id for BFCL (underscore -> hyphen)
    bfcl_model_id = convert_model_id(model_id)

    # Resolve model path
    model_path_resolved = _resolve_path(model_path)

    # Default output directory: eval_results/tool/<model_id>
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_RESULT_DIR, model_id)
    else:
        output_dir = _resolve_path(output_dir)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Determine total steps
    total_steps = 3 if run_evaluate else 2

    # Step 1: Register model to BFCL
    if verbose:
        print(f"\n{'='*60}")
        print(f"Tool Evaluation: {model_id}")
        print(f"{'='*60}")
        print(f"  Model Path: {model_path_resolved}")
        print(f"  BFCL Model ID: {bfcl_model_id}")
        print(f"  Categories: {', '.join(test_categories)}")
        print(f"  Output Dir: {output_dir}")
        print(f"  Force Re-register: {force_reregister}")
        print(f"  Run Evaluate: {run_evaluate}")
        print(f"{'='*60}")

    # Handle force re-registration
    model_already_registered = is_model_registered(model_id)

    if force_reregister and model_already_registered:
        if verbose:
            print(f"\n[1/{total_steps}] Force re-registration: Unregistering existing model...")
        success = unregister_model_from_bfcl(model_id, verbose=verbose)
        if not success:
            print(f"[WARNING] Failed to unregister model {model_id}, continuing with registration...")
        model_already_registered = False  # Treat as not registered

    if not model_already_registered:
        if verbose:
            print(f"\n[1/{total_steps}] Registering model to BFCL...")
        success = register_model_to_bfcl(
            model_id=model_id,
            model_path=model_path_resolved,
            display_name=model_id,
            handler="Qwen25Handler",
            is_fc_model=False,
            verbose=verbose
        )
        if not success:
            print(f"[ERROR] Failed to register model {model_id}")
            return False
    else:
        if verbose:
            print(f"\n[1/{total_steps}] Model {bfcl_model_id} already registered, skipping...")

    # Step 2: Run bfcl generate
    if verbose:
        print(f"\n[2/{total_steps}] Running BFCL generate...")

    categories_str = ",".join(test_categories)

    generate_cmd = f"""
cd {EVAL_ROOT}
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate {conda_env}
export BFCL_PROJECT_ROOT={output_dir}
bfcl generate \\
    --model {bfcl_model_id} \\
    --test-category {categories_str} \\
    --backend vllm \\
    --num-gpus {num_gpus} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    --local-model-path {model_path_resolved}
"""
    if verbose:
        print(f"Command:\n{generate_cmd}")

    try:
        result = subprocess.run(
            generate_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=not verbose,
            text=True
        )
        if result.returncode != 0:
            print(f"[ERROR] BFCL generate failed with return code {result.returncode}")
            if not verbose and result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run BFCL generate: {e}")
        return False

    # Step 3: Run bfcl evaluate (optional)
    if run_evaluate:
        if verbose:
            print(f"\n[3/{total_steps}] Running BFCL evaluate...")

        evaluate_cmd = f"""
cd {EVAL_ROOT}
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate {conda_env}
export BFCL_PROJECT_ROOT={output_dir}
bfcl evaluate \\
    --model {bfcl_model_id} \\
    --test-category {categories_str}
"""
        if verbose:
            print(f"Command:\n{evaluate_cmd}")

        try:
            result = subprocess.run(
                evaluate_cmd,
                shell=True,
                executable="/bin/bash",
                capture_output=not verbose,
                text=True
            )
            if result.returncode != 0:
                print(f"[ERROR] BFCL evaluate failed with return code {result.returncode}")
                if not verbose and result.stderr:
                    print(result.stderr)
                return False
        except Exception as e:
            print(f"[ERROR] Failed to run BFCL evaluate: {e}")
            return False

        if verbose:
            print(f"\n[DONE] Evaluation complete!")
            print(f"Results saved to: {output_dir}")
    else:
        if verbose:
            print(f"\n[DONE] Generation complete! (Skipped evaluate step)")
            print(f"Results saved to: {output_dir}")

    return True


def run_tool_eval_batch(
    configs: List[ToolEvalConfig],
    conda_env: str = BFCL_CONDA_ENV,
    verbose: bool = True,
) -> List[bool]:
    """
    Run BFCL evaluation on multiple models.

    Args:
        configs: List of ToolEvalConfig objects
        conda_env: Conda environment name for BFCL
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

        success = run_tool_eval(
            model_id=config.model_id,
            model_path=config.model_path,
            test_categories=config.test_categories,
            output_dir=config.output_dir,
            num_gpus=config.num_gpus,
            gpu_memory_utilization=config.gpu_memory_utilization,
            conda_env=conda_env,
            verbose=verbose,
            force_reregister=config.force_reregister,
            run_evaluate=config.run_evaluate,
        )
        results.append(success)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tool Calling Evaluation using BFCL",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (local path or HuggingFace ID)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model ID for BFCL registration (default: derived from model_path)"
    )

    # Evaluation options
    parser.add_argument(
        "--test_categories",
        type=str,
        default="live,non_live,multi_turn",
        help="Comma-separated list of test categories (default: live,non_live,multi_turn)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: eval_results/tool/<model_id>)"
    )

    # GPU options
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)"
    )

    # Other options
    parser.add_argument(
        "--conda_env",
        type=str,
        default=BFCL_CONDA_ENV,
        help=f"Conda environment name (default: {BFCL_CONDA_ENV})"
    )
    parser.add_argument(
        "--force_reregister",
        action="store_true",
        help="Force re-registration even if model is already registered"
    )
    parser.add_argument(
        "--skip_evaluate",
        action="store_true",
        help="Skip the bfcl evaluate step (only run generate)"
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

    # Derive model_id from model_path if not specified
    if args.model_id is None:
        args.model_id = os.path.basename(args.model_path.rstrip("/"))

    # Parse test categories
    test_categories = [c.strip() for c in args.test_categories.split(",")]

    # Validate categories
    for cat in test_categories:
        if cat not in TOOL_TEST_CATEGORIES:
            print(f"Warning: Unknown test category '{cat}'. Available: {TOOL_TEST_CATEGORIES}")

    verbose = args.verbose and not args.quiet

    # Run evaluation
    success = run_tool_eval(
        model_id=args.model_id,
        model_path=args.model_path,
        test_categories=test_categories,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        conda_env=args.conda_env,
        verbose=verbose,
        force_reregister=args.force_reregister,
        run_evaluate=not args.skip_evaluate,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)
