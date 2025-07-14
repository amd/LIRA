import argparse
import json
import numpy as np
import torchaudio
import time  # Add this import
from pathlib import Path
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from lira.whisper.transcribe import WhisperONNX
from lira.zipformer.transcribe import EncoderWrapper, DecoderWrapper, JoinerWrapper
from lira.utils.audio import extract_fbank, greedy_search, mic_stream, get_model_providers
from lira.utils.tokens import load_tokens

SAMPLE_RATE = 16000



# ---------------- Main ---------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="WAV file path or 'mic'")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--joiner")
    parser.add_argument("--tokens")
    parser.add_argument("--duration", type=int, default=0)
    parser.add_argument("--device", choices=['cpu', 'npu'], default='cpu')
    parser.add_argument("--model-type", choices=['zipformer', 'whisper'], required=True)
    parser.add_argument("--config-path", type=str, default="model_config.json", help="Path to the JSON configuration file")
    args = parser.parse_args()

    # Get providers for the model
    providers = get_model_providers(
        args.model_type,
        args.device,
        config_path=args.config_path
    )

    if args.model_type == "zipformer":
        encoder = EncoderWrapper(args.encoder, providers=providers["encoder"])
        decoder = DecoderWrapper(args.decoder, providers=providers["decoder"])
        joiner = JoinerWrapper(args.joiner, providers=providers["joiner"])
        tokens = load_tokens(args.tokens)

        def transcribe_fn(audio, state):
            features = extract_fbank(audio)
            return greedy_search(encoder, decoder, joiner, features, tokens, state)
    else:
        whisper_model = WhisperONNX(
            args.encoder,
            args.decoder,
            encoder_provider=providers["encoder"],
            decoder_provider=providers["decoder"]
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
        audio_duration = len(audio) / SAMPLE_RATE  # Calculate audio duration in seconds
        state = {}

        start_time = time.time()  # Start timing
        text = transcribe_fn(audio, state)
        end_time = time.time()  # End timing

        transcription_time = end_time - start_time
        rtf = transcription_time / audio_duration  # Calculate RTF

        print("\n🗣️ Transcription:", text)
        
        print(f"⏱️ RTF Performance: {rtf:.4f}")  # Print RTF

if __name__ == "__main__":
    main()