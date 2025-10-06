# LIRA: Local Inference for Realtime Audio
<p align="center">
    <p align="center">
        <img src="images/logo.png" alt="LIRA logo" width="1280" style="border-radius:24px; height:400px; object-fit:cover;">
    </p>

**Local, efficient automatic speech recognition (ASR).  Run ASR models on your local machine—fast, simple, and developer-friendly.**

LIRA is a **CLI-first, developer-friendly tool**: run and serve ASR models locally simply with the `lira run` and `lira serve` commands to integrate with your apps and tools.

---

## 🧩 Supported Model Architectures & Runtimes

LIRA supports multiple speech-model architectures. Runtime support depends on the exported model and chosen runtime.

| Model                | Typical use case                        | Runs on         | Supported datatypes                |
|----------------------|-----------------------------------------|-----------------|------------------------------------|
| [whisper-small](https://huggingface.co/openai/whisper-small)      | Low-latency, resource-constrained       | CPU, GPU, NPU*  | FP32, BFP16                        |
| [whisper-base](https://huggingface.co/openai/whisper-base)       | Balanced accuracy and performance       | CPU, GPU, NPU*  | FP32, BFP16                        |
| [whisper-medium](https://huggingface.co/openai/whisper-medium)     | Higher accuracy for challenging audio   | CPU, GPU, NPU*  | FP32, BFP16                        |
| [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)      | Highest accuracy (more compute)         | CPU, GPU        | FP32, BFP16                        |
| [zipformer](https://huggingface.co/papers/2310.11230)            | Streaming / low-latency ASR encoder     | CPU, GPU, NPU*  | FP32, BFP16                        |

<sub>*NPU support depends on available Vitis AI export artifacts and target hardware.</sub>

---

## 🚀 Getting Started

**Prerequisites:**

- **Python 3.11** is required.
- We recommend using **conda** for environment management.
- For Ryzen™ AI NPU flow, follow the [Ryzen AI installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html) and verify drivers/runtime for your device. Ensure that you have a Ryzen AI 300 Series machine to enable NPU use cases.
- Current recommended Ryzen AI version 1.5.0 with the 32.0.203.280 NPU driver.

**Minimal install steps:**

1. **Clone the repo and change directory:**
    ```bash
    git clone https://github.com/amd/LIRA.git
    cd LIRA
    ```

2. **Activate your conda environment:**
This conda environment should already be installed from the Ryzen AI SW installation mentioned earlier. 
    ```bash
    conda activate ryzen-ai-*.*.*
    ```
Replace the `ryzen-ai-*.*.*` with the version number you are using, such as `ryzen-ai-1.5.0`

3. **Install LIRA in editable mode:**
    ```bash
    pip install -e .
    ```

You can run `lira --help` to see available commands.

---

## ⚡ CLI-first Design

LIRA is a CLI-first toolkit focused on simple developer workflows for exporting, running, and serving speech models.

**Primary commands:**

1. Run models locally with `lira run`:

- Run, export, or benchmark models locally directly from the command line.  
- Use for local inference, ONNX export, or rapid prototyping.
```bash
lira run whisper --model-type whisper-base --export --device cpu --audio audio_files/test.wav
```

2. OpenAI API-compatible local model serving with `lira serve`:

- Launch and serve the model with a FastAPI server with OpenAI API-compatible endpoints.  
- Expose models as HTTP APIs for real-time transcription and seamless integration.  
- Add speech recognition to your apps, automate workflows, or build custom endpoints using standard REST calls.
```bash
lira serve --backend openai --model whisper-base --device cpu --host 0.0.0.0 --port 5000
```

**NPU Acceleration:**

For NPU acceleration, change `--device cpu` to `--device npu`.

> Interested in more server features?  
> Try the **LIRA server demo** with Open WebUI.  
> See [examples/openwebui](examples/openwebui) for setup instructions.

- Configure models via `config/model_config.json`.
- Set API keys (dummy) as environment variables for protected backends.

---

## 🏃 Running Models with `lira run`

To run a model using the CLI:
```bash
lira run <model> [options]
```
Replace `<model>` with the model name or path.

**Core flags:**
- `--device` — target device (`cpu`, `gpu`, `npu`)
- `-m` / `--model` — path to local exported model directory
- `--audio` — path to input audio file (wav)
- `--profile` — enable timing/profiling output

_Tip: run `lira run <model> --help` for model-specific flags._

---

### 🗣️ Running Whisper

Whisper supports export/optimization and model-specific flags.

**Example:**
Export Whisper base model to ONNX, optimize and run on NPU.
```bash
lira run whisper --model-type whisper-base --export --device npu --audio <input/.wav file> --use-kv-cache
```

Run inference on a sample audio file on CPU.
```bash
lira run whisper -m exported_models/whisper_base --device cpu --audio "audio_files/test.wav"
```

**Key Whisper flags:**
- `--model-type` — Hugging Face model id
- `--export` — export/prepare Whisper model to ONNX
- `--export-dir` — output path for export
- `--force` — overwrite existing export
- `--use-kv-cache` — enable KV-cache decoding
- `--static` — request static shapes during export
- `--opset` — ONNX opset version
- `--eval-dir` / `--results-dir` — run dataset evaluation

---

### 🔄 Running Zipformer

Zipformer enables streaming, low-latency transcription.

**Example:**
```bash
lira run zipformer -m <exported_model_dir> --device cpu --audio "audio_files/stream_sample.wav"
```

**Common CLI Flags:**
- `-m`, `--model` — exported Zipformer model directory
- `--device` — target device
- `--audio` — input audio file (WAV)
- `--cache` — cache directory (optional)
- `--profile` — enable profiling

_Tip: Run `lira run zipformer --help` for all options._

---

## ⚙️ Configuration

Model and runtime configs live in `config/`:

- [config/model_config.json](config/model_config.json) — model routing and defaults
- [config/vitisai_config_whisper_base_encoder.json](config/vitisai_config_whisper_base_encoder.json) — Vitis AI Whisper encoder config for NPU exports
- [config/vitisai_config_whisper_base_decoder.json](config/vitisai_config_whisper_base_decoder.json) — Vitis AI Whisper decoder config for NPU exports
- [config/vitisai_config_zipformer_encoder.json](config/vitisai_config_zipformer_encoder.json) — Vitis AI Zipformer encoder config for NPU exports

You can point to custom config files or modify those in the repo.

---



## 🧪 Early Access & Open Source Intentions

LIRA is released as an open, community-driven project.

- Some features and model exports are **early-access** — may be experimental or optimized for internal hardware.
- APIs may evolve; breaking changes can happen while stabilizing.
- Community feedback, issues, and PRs are encouraged!

_If you rely on specific exported models or NPU backends for production, treat early-access pieces with caution and verify performance._

---

## 🤝 Contributing

We welcome contributors!

- Open issues for bugs, UX problems, or feature requests
- Submit PRs with tests and clear descriptions
- Add new model integrations or tooling chains
- Improve documentation and examples

**Guidelines:**
- Follow code style and add unit tests where practical
- Make small, focused PRs with clear motivation
- Sign the Contributor License Agreement (CLA) if requested

See `CONTRIBUTING.md` for more details.

---

## 🗺️ Roadmap

Planned improvements:

- Stabilize CLI flags and server API
- Add more model backends and runtime targets
- Improve documentation and quickstart guides
- CI for model export verification and unit tests
- UX improvements: error messages, progress reporting, model comparison tools

Contributions aligned with these goals are highly welcome.

---

## 💡 Getting Help

- Open an issue on GitHub with reproduction steps and logs
- For design/roadmap, open a feature request or contact maintainers via GitHub discussions

Thank you for trying LIRA — we look forward to your contributions and feedback!

---

## 📄 License

This project is licensed under the terms of the MIT license.  
See the [LICENSE](LICENSE) file for details.

<sub>Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.</sub>

<sub>SPDX-License-Identifier: MIT</sub>
