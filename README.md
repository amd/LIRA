# LIRA: Local Inference tool for Realtime Audio
<p align="center">
    <p align="center">
        <img src="images/logo.png" alt="LIRA logo" width="1280" style="border-radius:24px; height:400px; object-fit:cover;">
    </p>

**Local, efficient speech recognition.  
Run ASR models on your machine—fast, simple, and developer-friendly.**

LIRA is a **CLI-first, developer-friendly tool**: run and serve ASR models locally with `lira run` and `lira serve` to integrate with your apps and tools.

---


## 🧩 Supported Model Architectures & Runtimes

LIRA supports multiple speech-model architectures. Runtime support depends on the exported model and chosen runtime.

| Model                | Typical use case                        | Runs on         | Supported datatypes                |
|----------------------|-----------------------------------------|-----------------|------------------------------------|
| whisper-base         | Low-latency, resource-constrained       | CPU, GPU, NPU  | FP32, BFP16                        |
| whisper-small        | Balanced accuracy and performance       | CPU, GPU, NPU  | FP32, BFP16                        |
| whisper-medium     | Higher accuracy for challenging audio   | CPU, GPU, NPU  | FP32, BFP16                        |
| whisper-large-v3-turbo      | Highest accuracy (more compute)         | CPU, GPU, NPU        | FP32, BFP16                        |
| Zipformer            | Streaming / low-latency ASR encoder     | CPU, GPU, NPU  | FP32, BFP16                        |

<sub>*NPU support depends on available Vitis AI export artifacts and target hardware.</sub>

---

## 🚀 Getting Started

**Prerequisites:**

- **Python 3.10** is required.
- We recommend using **conda** for environment management.
- To use Ryzen AI powered NPUs, you must have a RYzen AI 300 series laptop.
- For RyzenAI NPU flow, follow the [RyzenAI installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html) and verify drivers/runtime for your device.
- Requires **Ryzen AI 1.6.0 or above**

**Minimal install steps:**

1. **Activate your conda environment:**  
    If you have followed Ryzen AI setup instructions and installed the latest Ryzen AI, you'll have the latest environment available in `conda envs list`.

    ```bash
    conda activate <latest-ryzen-ai-environment>
    ```
    For example, use the latest
    ```bash
    conda activate ryzen-ai-1.6.0
    ```

2. **Clone the repo and change directory:**
    ```bash
    git clone https://github.com/aigdat/LIRA.git
    cd LIRA
    ```

3. **Install LIRA in editable mode:**
    ```bash
    pip install -e .
    ```

Now you can run `lira --help` to see available commands.

---

## ⚡ CLI-first Design

LIRA is a CLI-first toolkit focused on simple developer workflows for exporting, running, and serving speech models.

**Primary commands:**

- **`lira run`**  
    Run, export, or benchmark models directly from the command line.  
    Use for local inference, ONNX export, or rapid prototyping.

- **`lira serve`**  
    Launch a FastAPI server with OpenAI-compatible endpoints.  
    Expose models as HTTP APIs for real-time transcription and seamless integration.  
    Add speech recognition to your apps, automate workflows, or build custom endpoints using standard REST calls.

**Quick examples:**
```bash
# Run a model locally (inference)
lira run whisper --model-type whisper-base --export --device cpu --audio audio_files/test.wav

# Serve the model for local apps (OpenAI-compatible endpoints)
lira serve --backend openai --model whisper-base --device cpu --host 0.0.0.0 --port 5000
```

## 🖥️ LIRA Server

LIRA includes a FastAPI-based HTTP server for rapid integration with your applications. The server offers **OpenAI API compatibility** for real-time speech recognition.

**Start the server:**

- **CPU acceleration:**

    ```bash
    lira serve --backend openai --model whisper-base --device cpu --host 0.0.0.0 --port 5000
    ```
- **NPU acceleration:**

    ```bash
    lira serve --backend openai --model whisper-base --device npu --host 0.0.0.0 --port 5000
    ```
<details>
<summary><span style="color:#FFF9C4; font-size:1em;">Test &amp; Debug Your LIRA Server with a Sample Audio File</span></summary>

Open a new command prompt and run the following `curl` command to verify your server is working. Replace `audio_files\test.wav` with your actual audio file path if needed:

```bash
curl -X POST "http://localhost:5000/v1/audio/transcriptions" ^
        -H "accept: application/json" ^
        -H "Content-Type: multipart/form-data" ^
        -F "file=@audio_files\test.wav" ^
        -F "model=whisper-onnx"
```

If everything is set up correctly, you should receive a JSON response containing the transcribed text. 
</details>

**Notes:**  
- Models are configured in `config/model_config.json`.  
- For protected backends, set API keys as environment variables.

<div>

#### 🌐 **LIRA Server Demo: Try Open WebUI with LIRA**

Interested in more? Try the **LIRA server demo** with Open WebUI for interactive transcription, monitoring, and management.

<p>
    <a href="docs/OpenWebUI_README.md" style="font-size:1.1em; font-weight:bold; background:#1976D2; color:#FFF; padding:0.5em 1em; border-radius:8px; text-decoration:none;">
        👉 Get Started: Step-by-step Setup Guide
    </a>
</p>

---

</div>

## 🗣️ Running Models with `lira run`

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

### Running Whisper Locally using `lira run whisper`

To export, optimize, compile, and run Whisper models locally, follow this section for examples.

Check out `lira run whisper --help` for more details on running Whisper and Whisper-specific CLI flags.

**Key Whisper flags:**
- `--model-type` — Hugging Face model ID
- `--export` — export/prepare Whisper model to ONNX
- `--export-dir` — output path for export
- `--force` — overwrite existing export
- `--use-kv-cache` — enable KV-cache decoding
- `--static` — request static shapes during export
- `--opset` — ONNX opset version
**Examples:**  

- **Run `whisper-base` on CPU (auto-download, export, and run):**
    ```bash
    # Export Whisper base model to ONNX, optimize, and run on CPU with KV caching enabled
    lira run whisper --model-type whisper-base --export --device cpu --audio audio_files/test.wav --use-kv-cache
    ```

- **Run Whisper on NPU (encoder and decoder on NPU):**
    ```bash
    # Export Whisper base model to ONNX, optimize, and run on NPU
    lira run whisper --model-type whisper-base --export --device npu --audio <input.wav file>
    ```
    *Note:*  
    KV caching is not supported on NPU. If you use `--use-kv-cache` with `--device npu`, only the encoder runs on NPU; the decoder runs on CPU.

- **Run a locally exported Whisper ONNX model:**
    ```bash
    lira run whisper -m <path to local Whisper ONNX> --device cpu --audio "audio_files/test.wav"
    ```

---

### 🔄 Running Zipformer using `lira run zipformer`

Zipformer enables streaming, low-latency transcription. 

Run `lira run zipformer -h` for
more details on runnign Zipformer.

**Example:**
Using Zipformer English model exported in AMD Huggingface
```bash
lira run zipformer -m aigdat/AMD-zipformer-en  --device cpu --audio "audio_files/test.wav"
```

## ⚙️ Configuration

Model and runtime configs live in `config/`:

- `config/model_config.json` — model routing and defaults
- `vitisai_config_*.json` — Vitis AI configs for NPU exports

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
