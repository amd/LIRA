import argparse
import numpy as np
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from lira.whisper.transcribe import WhisperONNX
from lira.zipformer.transcribe import EncoderWrapper, DecoderWrapper, JoinerWrapper
from lira.utils.audio import extract_fbank, greedy_search, mic_stream, get_providers
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
    parser.add_argument("--device", choices=['cpu', 'cuda', 'npu'], default='cpu')
    parser.add_argument("--model-type", choices=['zipformer', 'whisper'], required=True)
    args = parser.parse_args()

    providers = get_providers(args.device)

    if args.model_type == "zipformer":
        encoder = EncoderWrapper(args.encoder, providers=providers)
        decoder = DecoderWrapper(args.decoder, providers=providers)
        joiner = JoinerWrapper(args.joiner, providers=providers)
        tokens = load_tokens(args.tokens)

        def transcribe_fn(audio, state):
            features = extract_fbank(audio)
            return greedy_search(encoder, decoder, joiner, features, tokens, state)
    else:
        whisper_model = WhisperONNX(args.encoder, args.decoder, providers=providers)

        def transcribe_fn(audio, state):
            return whisper_model.transcribe(audio)

    if args.input.lower() == 'mic':
        mic_stream(transcribe_fn, args.duration)
    else:
        waveform, sr = torchaudio.load(args.input)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        audio = waveform.squeeze(0).numpy()
        state = {}
        text = transcribe_fn(audio, state)
        print("\n🗣️ Transcription:", text)

if __name__ == "__main__":
    main()
