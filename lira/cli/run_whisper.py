import argparse
import json
import numpy as np
import onnxruntime as ort
import torchaudio
import sounddevice as sd
import queue
import threading
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 0.1 sec chunks


class WhisperONNX:
    def __init__(self, encoder_path, decoder_path, hf_model="openai/whisper-base",
                 encoder_providers=None, decoder_providers=None):
        # Load ONNX encoder and decoder with separate provider options
        self.encoder = ort.InferenceSession(encoder_path, providers=encoder_providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=decoder_providers)

        # Hugging Face feature extractor & tokenizer
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(hf_model)
        self.tokenizer = WhisperTokenizer.from_pretrained(hf_model)

        self.decoder_start_token = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = 448  # Adjust if needed for specific models

    def preprocess(self, audio):
        """
        Convert raw audio to Whisper log-mel spectrogram
        """
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return inputs["input_features"]

    def encode(self, input_features):
        """
        Run encoder ONNX model
        """
        return self.encoder.run(None, {"input_features": input_features})[0]

    def decode(self, encoder_out):
        """
        Greedy decode with fixed-length input_ids
        """
        tokens = [self.decoder_start_token]
        for _ in range(100):
            # Pad input_ids to (1, max_length)
            decoder_input = np.full((1, self.max_length), self.eos_token, dtype=np.int64)
            decoder_input[0, :len(tokens)] = tokens

            outputs = self.decoder.run(None, {
                "input_ids": decoder_input,
                "encoder_hidden_states": encoder_out
            })
            logits = outputs[0]
            next_token = int(np.argmax(logits[0, len(tokens)-1]))  # Current position logits
            if next_token == self.eos_token:
                break
            tokens.append(next_token)
        return tokens

    def transcribe(self, audio):
        """
        Full encode-decode pipeline
        """
        input_features = self.preprocess(audio)
        encoder_out = self.encode(input_features)
        tokens = self.decode(encoder_out)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def load_provider_options(config, model_name, device):
    """
    Load provider options for encoder and decoder from JSON config
    """
    # Extract model type (e.g., base, small, medium) from the model name
    model_key = model_name.split("-")[-1]  # e.g., whisper-base -> base
    if model_key not in config["whisper"]:
        raise ValueError(f"Model type '{model_key}' not found in config")

    if device not in config["whisper"][model_key]:
        raise ValueError(f"Device '{device}' not found in config for model type '{model_key}'")

    model_config = config["whisper"][model_key][device]
    encoder_opts = model_config["encoder"]
    decoder_opts = model_config["decoder"]

    def build_provider_opts(opts):
        if opts.get("config_file"):
            return [
                (
                    "VitisAIExecutionProvider",
                    {
                        "config_file": opts["config_file"],
                        "cache_dir": opts.get("cache_dir", ""),
                        "cache_key": opts.get("cache_key", "")
                    }
                )
            ]
        else:
            return ["CPUExecutionProvider"]

    return build_provider_opts(encoder_opts), build_provider_opts(decoder_opts)


def mic_stream(model, duration=0):
    """
    Capture microphone audio and transcribe in real time
    """
    q_audio = queue.Queue()
    stop_flag = threading.Event()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        q_audio.put(indata.copy())

    def feeder():
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=CHUNK_SIZE, callback=audio_callback):
            if duration > 0:
                sd.sleep(int(duration * 1000))
                stop_flag.set()
            else:
                while not stop_flag.is_set():
                    sd.sleep(100)

    threading.Thread(target=feeder, daemon=True).start()

    buffer = np.zeros((0,), dtype=np.float32)
    print("\n🗣️ Real-time Transcription:")
    while not stop_flag.is_set():
        try:
            chunk = q_audio.get(timeout=0.1).squeeze()
            buffer = np.concatenate((buffer, chunk))
            if len(buffer) >= SAMPLE_RATE * 5:  # Process every 5 seconds
                text = model.transcribe(buffer)
                print(text)
                buffer = np.zeros((0,), dtype=np.float32)
        except queue.Empty:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="WAV file path or 'mic'")
    parser.add_argument("--encoder", required=True, help="Path to Whisper encoder ONNX model")
    parser.add_argument("--decoder", required=True, help="Path to Whisper decoder ONNX model")
    parser.add_argument("--hf_model", default="openai/whisper-base",
                        help="Hugging Face model name (e.g., openai/whisper-small)")
    parser.add_argument("--config-file", default="./config/model_config.json", help="Path to Model provider configs")
    parser.add_argument("--device", choices=['cpu', 'npu'], default='cpu')
    parser.add_argument("--duration", type=int, default=0, help="Mic duration in seconds (0 = unlimited)")
    args = parser.parse_args()

    # Load VitisAI config
    if Path(args.config_file).exists():
        with open(args.config_file) as f:
            vitis_config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file {args.config_file} not found")

    # Load provider options
    encoder_providers, decoder_providers = load_provider_options(
        vitis_config, args.hf_model, args.device
    )

    # Load Whisper ONNX model with separate provider configs
    model = WhisperONNX(args.encoder, args.decoder,
                        hf_model=args.hf_model,
                        encoder_providers=encoder_providers,
                        decoder_providers=decoder_providers)

    # Process input
    if args.input.lower() == 'mic':
        mic_stream(model, args.duration)
    else:
        waveform, sr = torchaudio.load(args.input)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        audio = waveform.squeeze(0).numpy()
        text = model.transcribe(audio)
        print("\n🗣️ Transcription:", text)


if __name__ == "__main__":
    main()