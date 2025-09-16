# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import gradio as gr
import torchaudio
import torch
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

from lira.zipformer.transcribe import EncoderWrapper, DecoderWrapper, JoinerWrapper
from lira.whisper.transcribe import WhisperONNX
from lira.utils.audio import extract_fbank
from lira.utils.audio import get_model_providers
from lira.cli.run_asr import greedy_search, mic_stream

SAMPLE_RATE = 16000


def download_and_prepare_models(model_type, target_device, config_path="config/model_config.json"):
    providers = get_model_providers(
        model_type=model_type,
        device=target_device,
        config_path=config_path
    )

    if model_type == "whisper":
        model_dir = snapshot_download(repo_id="aigdat/AMD-Whisper-Base", cache_dir="./hf_models")
        encoder_path = str(Path(model_dir / "whisper-encoder.onnx"))
        decoder_path = str(Path(model_dir / "whisper-decoder.onnx"))
        whisper_model = WhisperONNX(
            str(encoder_path),
            str(decoder_path),
            encoder_provider=providers["encoder"],
            decoder_provider=providers["decoder"]
        )
        return {"model": whisper_model}

    elif model_type == "zipformer":
        model_dir = snapshot_download(repo_id="aigdat/AMD-zipformer-en", cache_dir="./hf_models")
        encoder = EncoderWrapper(str(Path(model_dir) / "encoder.onnx"), providers=providers["encoder"])
        decoder = DecoderWrapper(str(Path(model_dir) / "decoder.onnx"), providers=providers["decoder"])
        joiner = JoinerWrapper(str(Path(model_dir) / "joiner.onnx"), providers=providers["joiner"])
        tokens = []
        with open(Path(model_dir) / "tokens.txt", 'r', encoding='utf-8') as f:
            tokens = [line.strip().split()[0] for line in f]
        return {"encoder": encoder, "decoder": decoder, "joiner": joiner, "tokens": tokens}

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def transcribe_audio(audio, model_type, target_device, duration=5):
    with gr.Row():
        status = gr.Textbox(value="⏳ Downloading and preparing model...", visible=True, interactive=False)
    models = download_and_prepare_models(model_type, target_device)
    status.value = "✅ Model ready!"

    if isinstance(audio, np.ndarray):
        # Mic input from Gradio
        sr = SAMPLE_RATE
        audio = audio[:, 0] if audio.ndim > 1 else audio

    elif isinstance(audio, tuple):
        # Uploaded file
        if len(audio) != 2 or not isinstance(audio[0], (np.ndarray, torch.Tensor)):
            return "⚠️ Invalid audio data."
        audio, sr = audio
        audio = audio[:, 0] if audio.ndim > 1 else audio

    elif isinstance(audio, str) and os.path.exists(audio):
        # File path
        waveform, sr = torchaudio.load(audio)
        audio = waveform.squeeze(0).numpy()

    else:
        return "⚠️ No valid audio input detected."

    if sr != SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(torch.tensor(audio)).numpy()

    if model_type == "whisper":
        whisper_model = models["model"]
        text = whisper_model.transcribe(audio)
        return text

    elif model_type == "zipformer":
        encoder = models["encoder"]
        decoder = models["decoder"]
        joiner = models["joiner"]
        tokens = models["tokens"]
        features = extract_fbank(audio)
        state = {}
        text = greedy_search(encoder, decoder, joiner, features, tokens, state)
        return text


def start_mic_stream(model_type, target_device, duration=5):
    models = download_and_prepare_models(model_type, target_device)

    if model_type == "whisper":
        whisper_model = models["model"]

        def transcribe_fn(audio, state):
            return whisper_model.transcribe(audio)

    elif model_type == "zipformer":
        encoder = models["encoder"]
        decoder = models["decoder"]
        joiner = models["joiner"]
        tokens = models["tokens"]

        def transcribe_fn(features, state):
            return greedy_search(encoder, decoder, joiner, features, tokens, state)

    mic_stream(transcribe_fn, duration)
    return "🗣️ Transcription from microphone completed. See console for live text."


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🗣️ LIAR: Local Interface for Audio Recognition")
        gr.Markdown("### ASR Demo supporting Zipformer and Whisper")

        model_selector = gr.Dropdown(["zipformer", "whisper"], label="Model Type", value="zipformer")
        device_selector = gr.Dropdown(["cpu", "npu"], label="Device Selector", value="cpu")
        duration_input = gr.Number(label="Mic Input Duration (seconds)", value=5)

        with gr.Tab("Upload WAV File"):
            file_input = gr.Audio(sources="upload", type="filepath", label="Upload a .wav file")
            file_button = gr.Button("Transcribe File")
            file_output = gr.Textbox(label="Transcription")

        with gr.Tab("Mic Input"):
            mic_button = gr.Button("Start Mic Transcription")
            mic_output = gr.Textbox(label="Mic Output")

        file_button.click(
            fn=transcribe_audio,
            inputs=[file_input, model_selector, device_selector],
            outputs=file_output
        )

        mic_button.click(
            fn=start_mic_stream,
            inputs=[model_selector, device_selector, duration_input],
            outputs=mic_output
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
