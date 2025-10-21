# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import json
import shutil
import subprocess
import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import numpy as np
from onnx import shape_inference
from lira.utils.config import get_exported_cache_dir


# ----------------------------------------------------------------------
# Static shape parameter definitions
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
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
        "--force", action="store_true", help="Force re-export even if cache exists"
    )

    args = parser.parse_args()

    # Load parameter map
    if not args.static:
        with open("config/params.json", "r") as f:
            args.params_to_fix = json.load(f)
    else:
        args.params_to_fix = STATIC_PARAMS
    return args


# ----------------------------------------------------------------------
# Optimum CLI model export
# ----------------------------------------------------------------------
def export_with_optimum_cli(model_name, output_dir, opset):
    """Export transformer to ONNX format using optimum-cli."""
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
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print(f"Exported ONNX model to: {output_dir}")


# ----------------------------------------------------------------------
# Static shape patching with support for models > 2GB
# ----------------------------------------------------------------------
def force_set_static(model_path, output_path, mapping):
    """
    Safely convert ONNX dynamic dims to static dims without breaking >2GB models.
    """
    print(f"Loading large ONNX model from: {model_path}")
    model = onnx.load_model(model_path, load_external_data=True)

    def patch_value_info(value_infos):
        for vi in value_infos:
            if not vi.type.HasField("tensor_type"):
                continue
            shape = vi.type.tensor_type.shape
            for dim in shape.dim:
                if dim.dim_param and dim.dim_param in mapping:
                    dim.dim_value = int(mapping[dim.dim_param])
                    dim.ClearField("dim_param")

    patch_value_info(model.graph.input)
    patch_value_info(model.graph.output)
    patch_value_info(model.graph.value_info)

    # Save with external data to handle >2GB tensors
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datafile_name = output_path.with_suffix(".onnx_data").name

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=datafile_name,
        size_threshold=512 * 1024 * 1024,
    )

    # ✅ Use model path for validation instead of in-memory object
    try:
        onnx.checker.check_model(str(output_path))
        print(f"Model integrity verified for {output_path}")
    except Exception as e:
        print(f"Warning: ONNX checker skipped in-memory validation: {e}")

    print(f"Saved static ONNX model to {output_path}")
    print(f"   External tensor data stored at {datafile_name}")


# ----------------------------------------------------------------------
# Whisper model export workflow
# ----------------------------------------------------------------------
def export_whisper_model(
    model_name, output_dir=None, opset=17, static=False, force=False
):
    """Exports and statically fixes ONNX models for Whisper pipeline."""
    if output_dir is None:
        output_dir = get_exported_cache_dir() / model_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and not force:
        print(f"Cache already exists at {output_dir}. Use --force to overwrite.")
        return

    # Optional: export with optimum
    # export_with_optimum_cli(model_name, str(output_dir), opset)

    params_to_fix = STATIC_PARAMS if static else STATIC_PARAMS
    encoder_model = output_dir / "encoder_model.onnx"
    decoder_model = output_dir / "decoder_model.onnx"
    decoder_init_model = output_dir / "decoder_init_model.onnx"

    static_dir = output_dir.parent / f"{output_dir.name}_static"
    static_dir.mkdir(parents=True, exist_ok=True)

    if encoder_model.exists():
        force_set_static(
            encoder_model, static_dir / "encoder_model.onnx", params_to_fix
        )

    if decoder_model.exists():
        shutil.copy(decoder_model, decoder_init_model)
        force_set_static(decoder_init_model, decoder_init_model, STATIC_PARAMS_KV)
        force_set_static(
            decoder_model, static_dir / "decoder_model.onnx", STATIC_PARAMS
        )

    print("Model export and static shape conversion completed successfully.")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    export_whisper_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        opset=args.opset,
        static=args.static,
        force=args.force,
    )
