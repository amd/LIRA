# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
import numpy as np
import onnxruntime as ort
import torchaudio
import torch
import sounddevice as sd
import queue
import threading

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 0.1 second chunks
CHUNK_LEN = 151  # number of frames
BLANK_ID = 0


def get_providers(device):
    if device == "npu":
        return [
            (
                "VitisAIExecutionProvider",
                {
                    "config_file": "",  # Replace with actual config file if required
                    "cache_dir": "/tmp/vitis_cache",  # Customize as needed
                    "cache_key": "asr_zipformer",
                },
            )
        ]
    return ["CPUExecutionProvider"]


class EncoderWrapper:
    def __init__(self, model_path, providers=None):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.streaming = any(
            "cached_" in name or "cached_len_" in name for name in self.input_names
        )
        self.cache = self._init_cache() if self.streaming else None

    def _init_cache(self):
        cache = {}
        for inp in self.session.get_inputs():
            if inp.name == "x":
                continue
            shape = [1 if isinstance(s, str) else s for s in inp.shape]
            dtype = np.float32 if "float" in inp.type else np.int64
            canonical_name = inp.name.replace("cached_", "")
            cache[canonical_name] = np.zeros(shape, dtype=dtype)
        return cache

    def run(self, x):
        inputs = {"x": x.astype(np.float32)}
        if self.streaming:
            expanded_cache = {f"cached_{k}": v for k, v in self.cache.items()}
            inputs.update(expanded_cache)
        outs = self.session.run(None, inputs)
        if self.streaming:
            self.cache = {
                o.name.replace("new_cached_", ""): val
                for o, val in zip(self.session.get_outputs()[1:], outs[1:])
            }
        return outs[0]


class DecoderWrapper:
    def __init__(self, model_path, providers=None):
        self.session = ort.InferenceSession(model_path, providers=providers)

    def run(self, tokens):
        return self.session.run(None, {"y": tokens.astype(np.int64)})[0]


class JoinerWrapper:
    def __init__(self, model_path, providers=None):
        self.session = ort.InferenceSession(model_path, providers=providers)

    def run(self, encoder_out, decoder_out):
        if encoder_out.ndim == 3 and encoder_out.shape[1] == 1:
            encoder_out = encoder_out[:, 0, :]
        if decoder_out.ndim == 3 and decoder_out.shape[1] == 1:
            decoder_out = decoder_out[:, 0, :]
        return self.session.run(
            None, {"encoder_out": encoder_out, "decoder_out": decoder_out}
        )[0]


def extract_fbank(audio):
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    return torchaudio.compliance.kaldi.fbank(
        audio_tensor,
        num_mel_bins=80,
        sample_frequency=SAMPLE_RATE,
        dither=0.0,
        frame_length=25.0,
        frame_shift=10.0,
        snip_edges=True,
        use_energy=False,
    ).numpy()


def load_tokens(token_path):
    with open(token_path, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f]


def greedy_search(encoder, decoder, joiner, features, tokens, state):
    context_size = 2
    hyp = state.get("hyp", [BLANK_ID])
    context = [BLANK_ID] * (context_size - len(hyp)) + hyp[-context_size:]
    decoder_input = np.array(context, dtype=np.int64)[None, :]
    decoder_out = decoder.run(decoder_input)

    chunk_len = CHUNK_LEN if encoder.streaming else features.shape[0]
    for start in range(0, features.shape[0], chunk_len):
        chunk = features[start : start + chunk_len]
        if chunk.shape[0] < chunk_len:
            chunk = np.pad(
                chunk, ((0, chunk_len - chunk.shape[0]), (0, 0)), mode="constant"
            )
        chunk = chunk[np.newaxis, :, :]
        encoder_out = encoder.run(chunk)

        for t in range(encoder_out.shape[1]):
            step = 0
            while True:
                logits = joiner.run(encoder_out[:, t : t + 1, :], decoder_out)
                y = int(np.argmax(logits, axis=-1)[0])
                if y == BLANK_ID or step > 5:
                    break
                hyp.append(y)
                context = [BLANK_ID] * (context_size - len(hyp)) + hyp[-context_size:]
                decoder_input = np.array(context, dtype=np.int64)[None, :]
                decoder_out = decoder.run(decoder_input)
                step += 1

        if not encoder.streaming:
            break

    state["hyp"] = hyp
    state["text"] = (
        "".join([tokens[i] for i in hyp[1:] if i < len(tokens)])
        .replace("▁", " ")
        .strip()
    )
    return state["text"]


def mic_stream(callback, duration=0):
    q_audio = queue.Queue()
    stop_flag = threading.Event()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        q_audio.put(indata.copy())

    def feeder():
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):
            if duration > 0:
                sd.sleep(int(duration * 1000))
                stop_flag.set()
            else:
                while not stop_flag.is_set():
                    sd.sleep(100)

    threading.Thread(target=feeder, daemon=True).start()

    buffer = np.zeros((0,), dtype=np.float32)
    state = {}
    prev_text = ""
    print("\n🗣️ Transcription:")
    while not stop_flag.is_set():
        try:
            chunk = q_audio.get(timeout=0.1).squeeze()
            buffer = np.concatenate((buffer, chunk))
            if len(buffer) >= SAMPLE_RATE:
                features = extract_fbank(buffer)
                text = callback(features, state)
                new_text = text[len(prev_text) :]
                print(new_text, end="", flush=True)
                prev_text = text
                buffer = np.zeros((0,), dtype=np.float32)
        except queue.Empty:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="WAV path or 'mic'")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--joiner", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    args = parser.parse_args()

    providers = get_providers(args.device)

    encoder = EncoderWrapper(args.encoder, providers=providers)
    decoder = DecoderWrapper(args.decoder, providers=providers)
    joiner = JoinerWrapper(args.joiner, providers=providers)
    tokens = load_tokens(args.tokens)

    if args.input.lower() == "mic":

        def print_result(features, state):
            return greedy_search(encoder, decoder, joiner, features, tokens, state)

        mic_stream(print_result, args.duration)
    else:
        waveform, sr = torchaudio.load(args.input)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=SAMPLE_RATE
            )(waveform)
        audio = waveform.squeeze(0).numpy()
        features = extract_fbank(audio)
        state = {}
        text = greedy_search(encoder, decoder, joiner, features, tokens, state)
        print("\n🗣️ Transcription:", text)


if __name__ == "__main__":
    main()
