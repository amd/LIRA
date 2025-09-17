# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import subprocess
import onnx
import onnxruntime as ort
import json
import os
import argparse
import shutil

from onnx import shape_inference
from lira.utils.cache import get_cache_dir

# Global variable for static shape parameters
STATIC_PARAMS = {
    "batch_size": "1",
    "encoder_sequence_length / 2": "1500",
    "decoder_sequence_length": "448",
}
STATIC_PARAMS_KV = {
    "batch_size": "1",
    "encoder_sequence_length / 2": "1500",
    "decoder_sequence_length": "1",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export and fix ONNX models for Whisper"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model name (e.g., openai/whisper-base.en, openai/whisper-medium, etc.)",
    )
    parser.add_argument(
        "--output_dir", help="Directory to save the exported ONNX models"
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--static", action="store_true", help="Use static shape parameters"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force export even if cache already exists",
    )
    args = parser.parse_args()

    if not args.static:
        with open("config/params.json", "r") as f:
            args.params_to_fix = json.load(f)
    else:
        args.params_to_fix = STATIC_PARAMS

    return args


def export_with_optimum_cli(model_name, output_dir, opset):
    """Run the optimum-cli export command to generate ONNX models."""
    command = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        f"openai/{model_name}",
        "--opset",
        str(opset),
        output_dir,
    ]
    print(f"Running optimum-cli export for model: {model_name}")
    subprocess.run(command, check=True)
    print(f"Exported ONNX model to: {output_dir}")


def force_set_static(model_path, output_path, mapping):
    """
    mapping: dict of {dim_param_name: value}
    Example:
        {
            "batch_size": 1,
            "encoder_sequence_length / 2": 1500,
            "decoder_sequence_length": 1,
        }
    """
    model = onnx.load(model_path)

    def patch_value_info(value_infos):
        for vi in value_infos:
            if not vi.type.HasField("tensor_type"):
                continue
            shape = vi.type.tensor_type.shape
            for dim in shape.dim:
                if dim.dim_param in mapping:
                    dim.dim_value = int(mapping[dim.dim_param])

    patch_value_info(model.graph.input)
    patch_value_info(model.graph.output)
    patch_value_info(model.graph.value_info)

    onnx.save(model, output_path)
    print(f"Forced static shapes written to {output_path}")


def export_whisper_model(
    model_name, output_dir=None, opset=17, static=False, force=False
):
    """Exports the Whisper model and fixes static shapes if required."""
    # Set default output directory to project cache if not specified
    if output_dir is None:
        output_dir = get_cache_dir() / "models" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check if cache already exists
    if os.path.exists(output_dir) and not force:
        print(f"Cache already exists at {output_dir}. Use --force to overwrite.")
        return

    # Determine parameters to fix
    if not static:
        with open("config/params.json", "r") as f:
            params_to_fix = json.load(f)
    else:
        params_to_fix = STATIC_PARAMS

    # Step 1: Export ONNX model using optimum-cli
    export_with_optimum_cli(model_name, output_dir, opset)

    # Step 2: Fix static shapes for encoder and decoder models
    encoder_model_path = os.path.join(output_dir, "encoder_model.onnx")
    decoder_model_path = os.path.join(output_dir, "decoder_model.onnx")
    decoder_init_model_path = os.path.join(output_dir, "decoder_init_model.onnx")
    if os.path.exists(encoder_model_path):
        force_set_static(encoder_model_path, encoder_model_path, params_to_fix)

    if os.path.exists(decoder_model_path):

        # Create a copy of decoder_model.onnx before setting it to static
        shutil.copy(decoder_model_path, decoder_init_model_path)

        # Set the copied model to static using STATIC_PARAMS_KV
        force_set_static(
            decoder_init_model_path, decoder_init_model_path, STATIC_PARAMS_KV
        )

        static_decoder_model_path = os.path.join(output_dir, "decoder_model.onnx")
        force_set_static(decoder_model_path, static_decoder_model_path, STATIC_PARAMS)

    print("Model export and static shape conversion completed successfully.")


if __name__ == "__main__":
    args = parse_args()
    export_whisper_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        opset=args.opset,
        static=args.static,
        force=args.force,
    )
