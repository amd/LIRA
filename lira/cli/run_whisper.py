import argparse
import numpy as np
import onnxruntime as ort
import torchaudio
import sounddevice as sd
import queue
import threading
from transformers import WhisperFeatureExtractor, WhisperTokenizer

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 0.1 sec chunks

class WhisperONNX:
    def __init__(self, encoder_path, decoder_path, providers=None):
        # Load ONNX encoder and decoder
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

        # Hugging Face feature extractor & tokenizer
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

        self.decoder_start_token = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = 448  # Decoder max token length

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
            # Pad input_ids to (1, 448)
            decoder_input = np.full((1, 448), self.eos_token, dtype=np.int64)
            decoder_input[0, :len(tokens)] = tokens

            outputs = self.decoder.run(None, {
                "input_ids": decoder_input,
                "encoder_hidden_states": encoder_out
            })
            logits = outputs[0]
            next_token = int(np.argmax(logits[0, len(tokens)-1]))  # Use current position logits
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
    parser.add_argument("--device", choices=['cpu', 'npu'], default='cpu')
    parser.add_argument("--duration", type=int, default=0, help="Mic duration in seconds (0 = unlimited)")
    args = parser.parse_args()

    # Set ONNX Runtime providers
    if args.device == 'cpu':
        providers = ["CPUExecutionProvider"]

    elif args.device == 'npu':
        providers = ["VitisAIExecutionProvider"]
    else:
        raise ValueError(f"Unknown device: {args.device}")

    # Load Whisper ONNX model
    model = WhisperONNX(args.encoder, args.decoder, providers=providers)

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
