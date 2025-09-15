# LIRA

LIRA is a lightweight, modular toolkit for running and comparing speech models and chained tooling. It provides an extensible CLI and server components to run ONNX-based models (Whisper, Zipformer, and others), experiment with exports, and integrate additional processing tools.

This repository contains model export artifacts, example datasets, utilities, and a production-capable server aimed at making speech-model experimentation and deployment frictionless.

Note: parts of LIRA are early-access and under active development. We welcome contributions and collaboration to make the project more stable and user-friendly.

---

## Key Features

- Simple CLI to run models and utilities:
  - `lira run` — run a single model and optionally chain tools
  - `lira serve` — start the LIRA server for programmatic access
- Supports ONNX-exported models for CPU/NPU execution paths
- Example audio and transcripts in `datasets/` and `audio_files/`
- Modular code organization to add new models, tools, and backends

---

## Quickstart

Minimum steps to get started from source (Windows PowerShell example):

1. Clone the repository:

   git clone https://github.com/aigdat/LIRA.git
   cd LIRA

2. Create a Python environment and install dependencies:

   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

3. Run the CLI help to see available commands:

   python -m lira.cli -h

Or run a model (example):

   python -m lira.cli run whisper --input audio_files/61-70968-0000.wav

Start the local server (example):

   python -m lira.cli serve --backend openai --model whisper-base --device cpu --host 0.0.0.0 --port 5000

---

## CLI Overview

All CLI entry points are implemented in `lira/cli.py`. Primary commands:

- `run` — Runs a single model. Subcommands are provided per model (e.g., `whisper`, `zipformer`) and expose model-specific flags.
  - Example: `lira run whisper --input <file> --device cpu`

- `compare` — Compare multiple models on the same audio input.
  - Example: `lira compare --models whisper zipformer --input audio.wav`

- `tool` — Run a standalone helper tool or chain tools together.
  - Example: `lira tool --name normalize --args --level 0.8`

- `serve` — Start the LIRA HTTP server to serve model inference and tool endpoints.
  - Example: `lira serve --backend openai --model whisper-base --device cpu`

Refer to `lira/cli.py` to see model-specific CLI extensions and flags.

---

## Configuration

Model and runtime configuration files live in the `config/` directory. Example files:

- `config/model_config.json` — top-level model routing and defaults
- `vitisai_config_*.json` — Vitis AI specific configurations used for NPU exports

When running the server or CLI, you can point to custom config files or modify the ones in this repo for experiments.

---

## Models & Exports

The repo contains several exported model artifacts for experimentation under `exported_models/` and `exported_models_small/`. These include encoder/decoder ONNX files and tokenizer/artifacts required to run the models.

If you add additional exports, please follow the existing layout and add a short README or config entry describing source model, export tool, and verification steps.

---

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

---

## License

This project is intended to be open-source. Please confirm the LICENSE file in this repository for the exact terms. If a `LICENSE` file is missing, please contact maintainers before using for commercial purposes.

---

## Getting Help

- Open an issue on GitHub with reproduction steps and logs
- For design and roadmap discussions, open a feature request or contact maintainers via GitHub discussions (if enabled)

Thank you for trying LIRA — we look forward to your contributions and feedback as we build a robust, community-driven speech toolkit.

---

## Supported Model Architectures & Runtimes

LIRA includes integrations and exported artifacts for multiple common speech-model architectures. Runtime support depends on the specific exported model variant and the chosen runtime (CPU/GPU/NPU). Always verify the exact export and config under `exported_models/` before deploying.

| Model | Typical use case | Runs on | Supported datatypes |
|---|---|---:|---:|
| Whisper (small) | Low-latency, resource-constrained inference | CPU, GPU, NPU* | float32, float16, int8 (quantized variants)
| Whisper (base) | Balanced accuracy and performance | CPU, GPU, NPU* | float32, float16, int8 (quantized variants)
| Whisper (medium) | Higher accuracy for challenging audio | CPU, GPU, NPU* | float32, float16
| Whisper (large) | Highest accuracy (more compute) | CPU, GPU | float32, float16
| Zipformer | Streaming / low-latency ASR encoder | CPU, GPU, NPU* | float32, float16

* NPU support depends on available Vitis AI export artifacts and the target NPU hardware. Many NPU flows favor float16 or int8 quantized exports — check `exported_modelsvitisai_cache/` and the corresponding `vitisai_config_*.json` files for details.
