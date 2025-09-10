import numpy as np
import subprocess
import onnx
import onnxruntime as ort
import json
import os
import argparse
import shutil

from onnx import shape_inference

# Global variable for static shape parameters
STATIC_PARAMS = {
    "batch_size": "1",
    "encoder_sequence_length / 2": "1500",
    "decoder_sequence_length": "448"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Export and fix ONNX models for Whisper")
    parser.add_argument("--model_name", required=True, help="Model name (e.g., openai/whisper-base.en, openai/whisper-medium, etc.)")
    parser.add_argument("--output_dir", required=True, help="Directory to save the exported ONNX models")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--static", action="store_true", help="Use static shape parameters")
    args = parser.parse_args()

    if not args.static:
        with open('config/params.json', 'r') as f:
            args.params_to_fix = json.load(f)
    else:
        args.params_to_fix = STATIC_PARAMS

    return args

def directorycreation(directory_name):
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
        print(f"Directory '{directory_name}' already exists. It has been deleted.")
    os.mkdir(directory_name)

def generate_dummy_data(input_tensor, name):
    input_shape = input_tensor.shape
    input_type = input_tensor.type
    print(f"Generating dummy data for: {name}")
    if input_type == 'tensor(float)':
        return np.random.randint(0, 64, size=input_shape).astype(np.float32)
    elif input_type == 'tensor(float16)':
        return np.random.rand(*input_shape).astype(np.float32)
    elif input_type == 'tensor(int32)':
        return np.random.randint(0, 100, size=input_shape).astype(np.int32)
    elif input_type == 'tensor(int64)' and name == 'attention_mask':
        return np.random.randint(0, 32, size=input_shape).astype(np.int64)
    elif input_type == 'tensor(int64)' and name == 'token_type_ids':
        return np.random.randint(0, 32, size=input_shape).astype(np.int64)
    elif input_type == 'tensor(int64)':
        return np.random.randint(0, 9, size=input_shape).astype(np.int64)
    elif input_type == 'tensor(bool)':
        return np.ones(input_shape).astype(np.bool_)
    elif input_type == 'tensor(double)':
        return np.random.rand(*input_shape).astype(np.float64)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

def validate_onnx_model(model):
    try:
        onnx.checker.check_model(model)
        print("ONNX model is valid.")
        return True
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        print("ONNX model is not valid.")
        print(e)
        return False

def export_with_optimum_cli(model_name, output_dir, opset):
    """Run the optimum-cli export command to generate ONNX models."""
    command = [
        "optimum-cli", "export", "onnx",
        "--model", model_name,
        "--opset", str(opset),
        output_dir
    ]
    print(f"Running optimum-cli export for model: {model_name}")
    subprocess.run(command, check=True)
    print(f"Exported ONNX model to: {output_dir}")

def fix_static_shapes(input_model_path, output_model_path, params_to_fix):
    """Fix static shapes for the given ONNX model."""
    print(f"Fixing static shapes for: {input_model_path}")
    command_base = ["python", "-m", "onnxruntime.tools.make_dynamic_shape_fixed"]
    for param, value in params_to_fix.items():
        command = command_base + [input_model_path, output_model_path, "--dim_param", str(param), "--dim_value", str(value)]
        subprocess.run(command, check=True)
    print(f"Static shapes fixed and saved to: {output_model_path}")

if __name__ == "__main__":
    args = parse_args()

    # Step 1: Export ONNX model using optimum-cli
    export_with_optimum_cli(args.model_name, args.output_dir, args.opset)

    # Step 2: Fix static shapes for encoder and decoder models
    encoder_model_path = os.path.join(args.output_dir, "encoder_model.onnx")
    decoder_model_path = os.path.join(args.output_dir, "decoder_model.onnx")

    if os.path.exists(encoder_model_path):
        fix_static_shapes(encoder_model_path, encoder_model_path, args.params_to_fix)

    if os.path.exists(decoder_model_path):
        fix_static_shapes(decoder_model_path, decoder_model_path, args.params_to_fix)

    print("Model export and static shape conversion completed successfully.")