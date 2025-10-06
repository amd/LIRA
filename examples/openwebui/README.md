# 🚀 Quick Start: Local Audio Transcription with LIRA & OpenWebUI on NPU 

Ready to turn your browser into a voice-powered AI playground? With LIRA and OpenWebUI, you’ll be transcribing audio locally in no time—no cloud required! This guide shows you how to connect everything for instant, private voice transcription right from your desktop.

> **💡 Tip:** Make sure you’ve checked out the main `README.md` for setup and installation of Ryzen AI + LIRA.

---

## 1. Set up environments

**Recommended:** Use separate conda environments to avoid dependency conflicts.  
- For LIRA, reuse `ryzen-ai-1.6.0` to leverage NPU support.  
- For OpenWebUI, create a new environment.

### LIRA and OpenWebUI setup:
Follow the instructions in the [Getting Started](../README.md#getting-started) section of the main README.md to install and set up the Ryzen AI environment.

Let's set up OpenWebUI by first cloning the ryzen-ai-1.6.0 environment, and then installing `open-webui`.
```powershell
conda create -n openwebui python=3.11 -y --clone ryzen-ai-1.6.0
conda activate openwebui
pip install open-webui
```

---

## 2️. Start OpenWebUI 🌐

```powershell
open-webui serve
```
OpenWebUI will be available at [`http://localhost:8080`](http://localhost:8080) by default.

---

## 3️. Start the LIRA Whisper server 🗣️

Activate the LIRA environment and start the server using the CLI tool. The server exposes an OpenAI-compatible transcription route at `/v1/audio/transcriptions`.

**To target Whisper on CPU:**
```powershell
lira serve --backend openai --model whisper-base --device cpu --host 0.0.0.0 --port 5000
```
**For NPU:**
```powershell
lira serve --backend openai --model whisper-base --device npu --host 0.0.0.0 --port 5000
```
Once running, your API endpoint will be:

```
http://localhost:5000/api/v0
```

---

## 4️. Configure OpenWebUI for LLM and Audio Transcription 🛠️

1. Open your browser and navigate to [`http://localhost:8080`](http://localhost:8080)
2. Sign-in with your credentials

To enable chat and audio transcription features in OpenWebUI, connect it to your local LLM server (for chat) and LIRA server (for speech-to-text).

### a. Connect OpenWebUI to your LLM server

- Go to **⚙️ Admin Settings → Connections → OpenAI API**
- Click **➕ Add Connection**
        - **URL:** Enter your LLM server endpoint (e.g., for Lemonade: `http://localhost:8000/api/v0`)
        - **API Key:** Leave blank or use a placeholder like `fake-key` if required.
- Click **Save**

### b. Connect OpenWebUI to your LIRA ASR server

- Go to **⚙️ Admin Settings → Audio**
- Under **Speech-to-Text**, select **OpenAI**
        - **URL:** Enter your LIRA server endpoint (e.g., `http://localhost:5000/v1`)
- Save your settings

---

🎉 You can now chat with your local LLM and transcribe audio using LIRA directly in OpenWebUI.  
Record from your mic or upload audio files (`.wav`, `.mp3`)—OpenWebUI will send them to LIRA for transcription.

---

## 📝 Notes & Tips

- If you exported a Whisper ONNX model to a custom directory, set `LIRA_MODEL_DIR` before starting the server, or use `lira serve` flags to point at the export.
- For NPU runs, start `lira serve` from `ryzen-ai-1.6.0` so Vitis AI tooling and drivers are available.
- If running behind a reverse proxy, update OpenWebUI's API Base URL accordingly.

See the main [README.md](../README.md) for full LIRA setup and model export instructions.


Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT