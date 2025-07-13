# ASR Readme

## Objective

This repository is ongoing work to build, track, and experiment with lightweight ASR (Automatic Speech Recognition) projects, focusing on efficient deployment and rapid prototyping using models optimized for Ryzen AI (RAI).

## Main Options

### 1. Quick CLI Tool

The `run_dual.py` script provides a simple command-line interface for running ASR tasks. It supports both `zipformer` and `whisper` model architectures, using models from Hugging Face that are cached locally for speed. You can process audio from files or (experimentally) from a microphone.

**Example usage:**
```sh
python run_dual.py --encoder <encoder.onnx> --decoder <decoder.onnx> --joiner <joiner.onnx> --tokens <tokens.txt> --input <audio.wav|mic> --duration <seconds> --model-type <zipformer|whisper> --device <cpu|npu>
```

### 2. Live Demo App

A Gradio-based demo app is included for interactive ASR testing. You can upload a wave file, select your preferred model, and choose the target device. 
(Note: Microphone input is not yet supported in the demo.)

![ASR System Overview](images/asrimage.png)


These tools enable fast experimentation and deployment of ASR models tailored for RAI.

```bash
(ryzen-ai-1.4.0ga) C:\Users\ISWAALEX\asr>python main.py

* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```
