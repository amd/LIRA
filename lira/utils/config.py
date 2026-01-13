# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.json"

# DATASETS_DIR = PROJECT_ROOT / "datasets"
# EXPORTED_MODELS_DIR = PROJECT_ROOT / "exported_models"


def get_cache_dir():
    """Returns the path to the project's cache directory."""
    cache_dir = Path.home() / ".cache" / "lira"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_npu_cache_dir():
    """Returns the path to the NPU cache directory."""
    npu_cache_dir = get_cache_dir() / "npu"
    npu_cache_dir.mkdir(parents=True, exist_ok=True)
    return npu_cache_dir


def get_exported_cache_dir():
    """Returns the path to the exported cache directory."""
    exported_cache_dir = get_cache_dir() / "exported_models"
    exported_cache_dir.mkdir(parents=True, exist_ok=True)
    return exported_cache_dir


def get_provider(
    device, model, component, cache_dir=None, config_path=MODEL_CONFIG_PATH
):
    """
    Selects the appropriate ONNX Runtime provider and options based on the device, model, and component.

    Args:
        device (str): The target device ('cpu' or 'npu').
        model (str): The model type (e.g., 'zipformer', 'whisper-base').
        component (str): The model component (e.g., 'encoder', 'decoder', 'joiner').
        cache_dir (str, optional): Custom cache directory path. Defaults to None.
        config_path (str): Path to the JSON configuration file.

    Returns:
        tuple: A tuple containing the list of providers and provider options.
    """
    # Normalize model name for variations like whisper-base, whisper-small, etc.

    if model.startswith("whisper"):
        cache_key_startswith = model.replace("-", "_")
        model = "whisper"

    if device == "npu":
        with open(config_path, "r") as f:
            config = json.load(f)

        if model not in config:
            raise ValueError(f"Configuration for model '{model}' not found.")

        device_config = config[model].get("npu", {}).get(component, {})

        # Extract the config file path
        config_file = device_config.get("config_file")
        if not config_file:
            print(
                f"Config file for component '{component}' in model '{model}' not found. Falling back to CPUExecutionProvider."
            )
            return ["CPUExecutionProvider"]

        config_file_path = Path(config_file)
        if not config_file_path.exists():
            raise FileNotFoundError(
                f"Config file '{config_file}' for component '{component}' does not exist."
            )

        # Set cache directory
        if cache_dir is None:
            cache_dir = get_npu_cache_dir()

        # Generate provider options
        options = {
            "cache_dir": str(cache_dir),
            "cache_key": f"{cache_key_startswith}_{component}",
            "config_file": str(config_file_path),
            # "enable_cache_file_io_in_mem": 0,
        }

        return [("VitisAIExecutionProvider", options)]

    if device == "gpu":
        return ["DmlExecutionProvider"]

    # Simplify CPU case to return only the list of providers
    return ["CPUExecutionProvider"]
