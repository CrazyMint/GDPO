"""GDPO Evaluation Module.

Provides evaluation tools for BFCL tool calling (v3/v4) and math benchmarks.
"""
from .bfcl_v4.bfcl_register import (
    register_model_to_bfcl,
    unregister_model_from_bfcl,
    is_model_registered,
    convert_model_id,
    get_model_ids,
)
from .bfcl_v4.tool_eval import run_tool_eval, run_tool_eval_batch, ToolEvalConfig, TOOL_TEST_CATEGORIES
from .math.math_eval import run_math_eval, run_math_eval_batch, MathEvalConfig, MATH_TASKS

__all__ = [
    # BFCL registration
    "register_model_to_bfcl",
    "unregister_model_from_bfcl",
    "is_model_registered",
    "convert_model_id",
    "get_model_ids",
    # Tool evaluation
    "run_tool_eval",
    "run_tool_eval_batch",
    "ToolEvalConfig",
    "TOOL_TEST_CATEGORIES",
    # Math evaluation
    "run_math_eval",
    "run_math_eval_batch",
    "MathEvalConfig",
    "MATH_TASKS",
]
