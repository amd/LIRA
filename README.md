# LIRA: Local Inference tool for Realtime Audio

![LIRA logo](images/logo.png)

Bring the power of speech locally — fast, high-performance ASR that's simple to run on your machine.

LIRA is a CLI-first, developer-friendly tool: run and serve ASR models locally with `lira run` and `lira serve` to integrate with your apps and tools.

---
## Getting Started

**Prerequisites**:

- Python 3.10 is required.
- We recommend using conda for environment management.
- If you plan to use the RyzenAI NPU flow, follow the [RyzenAI installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html) and verify drivers/runtime are installed for your device.

After the prerequisites are satisfied, the minimal install steps are:

1. Clone the repo and change directory:

    ```bash
    git clone https://github.com/aigdat/LIRA.git
    cd LIRA
    ```

2. Activate your conda environment (example used in development):

    ```bash
    conda activate ryzen-ai-1.5.0
    ```

3. Install LIRA in editable mode so the `lira` entrypoint is available:

    ```bash
    pip install -e .
    ```

Now you can run `lira --help` to see available commands.

---
## Running the server

LIRA includes a FastAPI-based HTTP server designed for rapid integration with your applications. The server offers OpenAI API compatibility, making it easy to connect existing tools and workflows for real-time speech recognition.

Start the server:

```bash
lira serve --backend openai --model whisper-base --device cpu --host 0.0.0.0 --port 5000
```

- Configure models via `config/model_config.json`.
- Set API keys (dummy) as environment variables for protected backends.


## How to use the CLI
To run a model using the CLI, use the following syntax:

```bash
lira run <model> [options]
```

Replace `<model>` with the model name or path, and specify any required options or flags as needed.

Core flags (common to most models):

- `--device` — target device (`cpu`, `gpu`, `npu`).
- `-m` / `--model` — path to a local exported model directory (ONNX files + tokenizer/config).
- `--audio` — path to the input audio file (wav).
- `--profile` — enable simple timing/profiling output.

Tip: run `lira run <model> --help` to see model-specific flags.

---

### Running Whisper

Whisper has built-in export/optimization support and a few model-specific flags.

#### Example

To export and run Whisper:

```bash
# Export the Whisper base model to ONNX format, optimize and run on NPU
lira run whisper --model-type whisper-base --export --device npu --audio <input/.wav file> --use-kv-cache
# Run inference on a sample audio file
lira run whisper -m exported_models/whisper_base --device cpu --audio "audio_files/test.wav"
```

**Usage:**

- The first command exports the Whisper base model to the specified directory.
- The second command runs the exported model on a WAV audio file using the CPU.
- Replace `"audio_files/test.wav"` with your own audio file path as needed.
- Use `--help` with any command for more options and model-specific flags.

Key Whisper flags:

- `--model-type` — Hugging Face model id (e.g. `whisper-base`, `whisper-small`).
- `--export` — export/prepare a Whisper model into an ONNX export (goes to `--export-dir`).
- `--export-dir` — output path for the export (default: `exported_models`).
- `--force` — when used with `--export`, overwrite an existing export.
- `--use-kv-cache` — enable KV-cache decoding (requires KV-capable exports).
- `--static` — request static shapes during export.
- `--opset` — ONNX opset version for export (default: `17`).
- `--eval-dir` / `--results-dir` — run dataset evaluation and store results.
---

### Running Zipformer

Zipformer enables streaming, low-latency transcription for real-time ASR tasks.

#### Example Usage

To run a Zipformer model:

```bash
lira run zipformer -m <exported_model_dir> --device cpu --audio "audio_files/stream_sample.wav"
```

Replace `<exported_model_dir>` with the path to your exported Zipformer model directory.

### Common CLI Flags

- `-m`, `--model` — Path to the exported Zipformer model directory.
- `--device` — Target device (`cpu`, `gpu`, `npu`).
- `--audio` — Path to the input audio file (WAV format).
- `--cache` — Optional cache directory for intermediate data.
- `--profile` — Enable simple timing/profiling output.

Tip: Run `lira run zipformer --help` for a full list of supported flags and options.

---

## Configuration

Model and runtime configuration files live in the `config/` directory. Example files:

- `config/model_config.json` — top-level model routing and defaults
- `vitisai_config_*.json` — Vitis AI specific configurations used for NPU exports

When running the server or CLI, you can point to custom config files or modify the ones in this repo for experiments.

---

## Supported Model Architectures & Runtimes

LIRA includes integrations and exported artifacts for multiple common speech-model architectures. Runtime support depends on the specific exported model variant and the chosen runtime (CPU/GPU/NPU). Always verify the exact export and config on this table before deploying.

| Model | Typical use case | Runs on | Supported datatypes |
|---|---|---:|---:|
| Whisper (small) | Low-latency, resource-constrained inference | CPU, GPU, NPU* | float32, float16, int8 (quantized variants)
| Whisper (base) | Balanced accuracy and performance | CPU, GPU, NPU* | float32, float16, int8 (quantized variants)
| Whisper (medium) | Higher accuracy for challenging audio | CPU, GPU, NPU* | float32, float16
| Whisper (large) | Highest accuracy (more compute) | CPU, GPU | float32, float16
| Zipformer | Streaming / low-latency ASR encoder | CPU, GPU, NPU* | float32, float16

* NPU support depends on available Vitis AI export artifacts and the target NPU hardware. 


## Datasets & Examples

- `datasets/LibriSpeech` contains sample audio and transcripts used for quick testing. Replace or augment these with your own data when benchmarking.

---

## Early Access & Open Source Intentions

LIRA is released with the intent to be an open, community-driven project. Important notes:

- Some features and model exports are marked early-access — they may be experimental, incomplete, or optimized for internal hardware (e.g., Vitis AI flows).
- The codebase and APIs may evolve; breaking changes can happen while we stabilize interfaces.
- We encourage community feedback, issues, and PRs to expand model compatibility, improve user experience, and add more tooling.

If you rely on specific exported models or NPU backends for production, treat the early-access pieces with caution and verify performance and accuracy for your use case.

---

## Contributing

We want contributors! Suggested ways to help:

- Open issues for bugs, UX problems, or feature requests
- Submit PRs with tests and clear descriptions
- Add new model integrations or tooling chains
- Improve documentation and examples

Guidelines:

- Follow existing code style and add unit tests where practical
- Make small, focused PRs with clear motivation and reproducible steps
- Sign the Contributor License Agreement (CLA) if requested by maintainers

See `CONTRIBUTING.md` (if present) for more details and templates.

---

## Roadmap

Planned improvements and focus areas:

- Stabilize CLI flags and server API
- Add more model backends and runtime targets (quantized, NPU)
- Improve documentation, quickstart guides, and reproducible examples
- CI for model export verification and unit tests
- UX improvements: clearer error messages, progress reporting, and model comparison tools

Contributions that align with these goals are highly welcome.

## Getting Help

- Open an issue on GitHub with reproduction steps and logs
- For design and roadmap discussions, open a feature request or contact maintainers via GitHub discussions (if enabled)

Thank you for trying LIRA — we look forward to your contributions and feedback as we build a robust, community-driven speech toolkit.

---

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.