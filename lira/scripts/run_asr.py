import argparse
import json
import numpy as np
import torchaudio
import time
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from lira.whisper.transcribe import WhisperONNX
from lira.zipformer.transcribe import EncoderWrapper, DecoderWrapper, JoinerWrapper
from lira.utils.audio import extract_fbank, greedy_search, mic_stream, get_model_providers
from lira.utils.tokens import load_tokens

SAMPLE_RATE = 16000

def get_model_dir(model_dir):
    """
    Resolves the model directory. If the input is an HF repo name, downloads the model.
    """
    if not os.path.exists(model_dir):
        print(f"⏳ Downloading model from Hugging Face: {model_dir}")
        model_dir = snapshot_download(repo_id=model_dir, cache_dir="./hf_models")
    return Path(model_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="WAV file path or 'mic'")
    parser.add_argument("--model-dir", required=True, help="Path to local model directory or HF repo name")
    parser.add_argument("--duration", type=int, default=0, help="Duration for mic input (seconds)")
    parser.add_argument("--device", choices=['cpu', 'npu'], default='cpu', help="Target device")
    parser.add_argument("--model-type", choices=['zipformer', 'whisper-base', 'whisper-small', 'whisper-medium'], required=True, help="Model type")
    parser.add_argument("--config-path", type=str, default="model_config.json", help="Path to the JSON configuration file")
    args = parser.parse_args()

    model_dir = get_model_dir(args.model_dir)

    providers = get_model_providers(
        args.model_type,
        args.device,
        config_path=args.config_path
    )

    if args.model_type == "zipformer":
        encoder = EncoderWrapper(str(model_dir / "encoder.onnx"), providers=providers["encoder"])
        decoder = DecoderWrapper(str(model_dir / "decoder.onnx"), providers=providers["decoder"])
        joiner = JoinerWrapper(str(model_dir / "joiner.onnx"), providers=providers["joiner"])
        tokens = load_tokens(str(model_dir / "tokens.txt"))

        def transcribe_fn(audio, state):
            features = extract_fbank(audio)
            return greedy_search(encoder, decoder, joiner, features, tokens, state)
    elif args.model_type.startswith("whisper"):
        whisper_model = WhisperONNX(
            str(model_dir / "whisper-encoder.onnx"),
            str(model_dir / "whisper-decoder.onnx"),
            encoder_provider=providers["encoder"],
            decoder_provider=providers["decoder"],
            model_type=args.model_type
        )

        def transcribe_fn(audio, state):
            return whisper_model.transcribe(audio)

    if args.input.lower() == 'mic':
        mic_stream(transcribe_fn, args.duration)
    else:
        waveform, sr = torchaudio.load(args.input)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        audio = waveform.squeeze(0).numpy()
        audio_duration = len(audio) / SAMPLE_RATE 
        state = {}

        start_time = time.time()
        text = transcribe_fn(audio, state)
        end_time = time.time()

        transcription_time = end_time - start_time
        rtf = transcription_time / audio_duration 

        print("\n🗣️ Transcription:", text)
        print(f"⏱️ RTF Performance: {rtf:.4f}")
if __name__ == "__main__":
    main()