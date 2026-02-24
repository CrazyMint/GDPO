"""
Dynamic model registration for BFCL v3 evaluation.

This module provides functions to dynamically register models to BFCL's
configuration files without modifying the BFCL codebase itself.

Points to gorilla-v3/ (BFCL v3) instead of gorilla/ (BFCL v4).
"""
import os
import re
from typing import Optional, Tuple
from dataclasses import dataclass


# Path resolution
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(_THIS_DIR)

# BFCL v3 configuration file paths (absolute)
BFCL_ROOT = os.path.join(EVAL_ROOT, "gorilla-v3", "berkeley-function-call-leaderboard")
MODEL_CONFIG_PATH = os.path.join(BFCL_ROOT, "bfcl_eval/constants/model_config.py")
SUPPORTED_MODELS_PATH = os.path.join(BFCL_ROOT, "bfcl_eval/constants/supported_models.py")


def convert_model_id(model_id: str) -> str:
    """
    Convert model_id for BFCL registration.

    BFCL doesn't support underscores in model IDs, so we convert them to hyphens.
    The saved model files still use underscores.

    Args:
        model_id: Original model ID (may contain underscores)

    Returns:
        BFCL-compatible model ID (underscores replaced with hyphens)
    """
    return model_id.replace("_", "-")


def get_model_ids(model_id: str) -> Tuple[str, str]:
    """
    Get both the original model ID and BFCL-compatible model ID.

    Args:
        model_id: Original model ID

    Returns:
        Tuple of (original_id, bfcl_id)
    """
    bfcl_id = convert_model_id(model_id)
    return model_id, bfcl_id


@dataclass
class BFCLModelConfig:
    """Configuration for a BFCL model registration."""
    model_id: str           # Unique model identifier (key in config)
    model_path: str         # Local path or HuggingFace ID
    display_name: str       # Display name on leaderboard
    handler: str = "Qwen25Handler"  # Handler class name
    url: str = ""           # Reference URL
    org: str = "Custom"     # Organization
    license: str = "MIT"    # License
    is_fc_model: bool = False  # Function-calling mode
    underscore_to_dot: bool = False


def _generate_model_config_entry(config: BFCLModelConfig) -> str:
    """Generate a ModelConfig entry string for model_config.py."""
    return f'''    "{config.model_id}": ModelConfig(
        model_name="{config.model_path}",
        display_name="{config.display_name}",
        url="{config.url}",
        org="{config.org}",
        license="{config.license}",
        model_handler={config.handler},
        input_price=None,
        output_price=None,
        is_fc_model={config.is_fc_model},
        underscore_to_dot={config.underscore_to_dot},
    ),'''


def _resolve_path(path: str) -> str:
    """Resolve a path to absolute, handling both relative and absolute paths."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(EVAL_ROOT, path))


def register_model_to_bfcl(
    model_id: str,
    model_path: str,
    display_name: Optional[str] = None,
    handler: str = "Qwen25Handler",
    is_fc_model: bool = False,
    verbose: bool = True
) -> bool:
    """
    Register a model to BFCL v3's configuration files.

    This function modifies two files:
    1. model_config.py: Adds ModelConfig entry to local_inference_model_map
    2. supported_models.py: Adds model_id to SUPPORTED_MODELS list

    Note: model_id will be converted to BFCL-compatible format (underscores -> hyphens)

    Args:
        model_id: Unique identifier for the model (underscores will be converted to hyphens)
        model_path: Local path to the model or HuggingFace ID
        display_name: Display name for leaderboard (defaults to model_id)
        handler: Handler class name (default: Qwen25Handler)
        is_fc_model: Whether this is a function-calling model
        verbose: Print status messages

    Returns:
        True if registration successful, False otherwise
    """
    # Convert model_id for BFCL compatibility
    bfcl_model_id = convert_model_id(model_id)

    # Resolve model path
    model_path_resolved = _resolve_path(model_path)

    if display_name is None:
        display_name = model_id

    if verbose and model_id != bfcl_model_id:
        print(f"[BFCL v3 Register] Converting model_id: {model_id} -> {bfcl_model_id}")

    config = BFCLModelConfig(
        model_id=bfcl_model_id,
        model_path=model_path_resolved,
        display_name=display_name,
        handler=handler,
        is_fc_model=is_fc_model,
    )

    try:
        # Step 1: Update model_config.py
        if verbose:
            print(f"[BFCL v3 Register] Adding model to model_config.py: {bfcl_model_id}")

        with open(MODEL_CONFIG_PATH, 'r') as f:
            model_config_content = f.read()

        # Check if model already registered
        if f'"{bfcl_model_id}"' in model_config_content:
            if verbose:
                print(f"[BFCL v3 Register] Model {bfcl_model_id} already exists in model_config.py, skipping...")
        else:
            # Find the position to insert (before the closing brace of local_inference_model_map)
            local_map_match = re.search(r'local_inference_model_map\s*=\s*\{', model_config_content)
            if local_map_match:
                start_pos = local_map_match.end()
                brace_count = 1
                pos = start_pos
                while brace_count > 0 and pos < len(model_config_content):
                    if model_config_content[pos] == '{':
                        brace_count += 1
                    elif model_config_content[pos] == '}':
                        brace_count -= 1
                    pos += 1

                insert_pos = pos - 1

                new_entry = _generate_model_config_entry(config)

                model_config_content = (
                    model_config_content[:insert_pos] +
                    "\n" + new_entry + "\n" +
                    model_config_content[insert_pos:]
                )

                with open(MODEL_CONFIG_PATH, 'w') as f:
                    f.write(model_config_content)

                if verbose:
                    print(f"[BFCL v3 Register] Successfully added to model_config.py")

        # Step 2: Update supported_models.py
        if verbose:
            print(f"[BFCL v3 Register] Adding model to supported_models.py: {bfcl_model_id}")

        with open(SUPPORTED_MODELS_PATH, 'r') as f:
            supported_models_content = f.read()

        if f'"{bfcl_model_id}"' in supported_models_content:
            if verbose:
                print(f"[BFCL v3 Register] Model {bfcl_model_id} already exists in supported_models.py, skipping...")
        else:
            match = re.search(r'SUPPORTED_MODELS\s*=\s*\[', supported_models_content)
            if match:
                start_pos = match.end()
                bracket_count = 1
                pos = start_pos
                while bracket_count > 0 and pos < len(supported_models_content):
                    if supported_models_content[pos] == '[':
                        bracket_count += 1
                    elif supported_models_content[pos] == ']':
                        bracket_count -= 1
                    pos += 1

                insert_pos = pos - 1

                new_entry = f'    "{bfcl_model_id}",'

                supported_models_content = (
                    supported_models_content[:insert_pos] +
                    "\n" + new_entry + "\n" +
                    supported_models_content[insert_pos:]
                )

                with open(SUPPORTED_MODELS_PATH, 'w') as f:
                    f.write(supported_models_content)

                if verbose:
                    print(f"[BFCL v3 Register] Successfully added to supported_models.py")

        if verbose:
            print(f"[BFCL v3 Register] Model {bfcl_model_id} registered successfully!")

        return True

    except Exception as e:
        print(f"[BFCL v3 Register] Error registering model: {e}")
        import traceback
        traceback.print_exc()
        return False


def unregister_model_from_bfcl(model_id: str, verbose: bool = True) -> bool:
    """
    Remove a model from BFCL v3's configuration files.

    Args:
        model_id: Unique identifier for the model to remove (will be converted to BFCL format)
        verbose: Print status messages

    Returns:
        True if unregistration successful, False otherwise
    """
    bfcl_model_id = convert_model_id(model_id)

    try:
        # Step 1: Remove from model_config.py
        if verbose:
            print(f"[BFCL v3 Unregister] Removing model from model_config.py: {bfcl_model_id}")

        with open(MODEL_CONFIG_PATH, 'r') as f:
            content = f.read()

        pattern = rf'    "{re.escape(bfcl_model_id)}": ModelConfig\(.*?\),\n'
        new_content = re.sub(pattern, '', content, flags=re.DOTALL)

        if new_content != content:
            with open(MODEL_CONFIG_PATH, 'w') as f:
                f.write(new_content)
            if verbose:
                print(f"[BFCL v3 Unregister] Removed from model_config.py")
        else:
            if verbose:
                print(f"[BFCL v3 Unregister] Model not found in model_config.py")

        # Step 2: Remove from supported_models.py
        if verbose:
            print(f"[BFCL v3 Unregister] Removing model from supported_models.py: {bfcl_model_id}")

        with open(SUPPORTED_MODELS_PATH, 'r') as f:
            content = f.read()

        pattern = rf'\s*"{re.escape(bfcl_model_id)}",?\n?'
        new_content = re.sub(pattern, '', content)

        if new_content != content:
            with open(SUPPORTED_MODELS_PATH, 'w') as f:
                f.write(new_content)
            if verbose:
                print(f"[BFCL v3 Unregister] Removed from supported_models.py")
        else:
            if verbose:
                print(f"[BFCL v3 Unregister] Model not found in supported_models.py")

        if verbose:
            print(f"[BFCL v3 Unregister] Model {bfcl_model_id} unregistered successfully!")

        return True

    except Exception as e:
        print(f"[BFCL v3 Unregister] Error unregistering model: {e}")
        import traceback
        traceback.print_exc()
        return False


def is_model_registered(model_id: str) -> bool:
    """
    Check if a model is already registered in BFCL v3.

    Args:
        model_id: Model ID to check (will be converted to BFCL format)

    Returns:
        True if registered, False otherwise
    """
    bfcl_model_id = convert_model_id(model_id)
    try:
        with open(MODEL_CONFIG_PATH, 'r') as f:
            content = f.read()
        return f'"{bfcl_model_id}"' in content
    except Exception:
        return False
