import torch
import torchaudio
import os
import queue
import threading
import numpy as np
import json
import sounddevice as sd
from lira.utils.config import MODEL_CONFIG_PATH
from pathlib import Path
from lira.utils.cache import get_cache_dir

BLANK_ID = 0
CHUNK_LEN = 151
SAMPLE_RATE = 16000


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


def mic_stream(transcribe_fn, duration=0):
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
            blocksize=CHUNK_LEN,
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
    print("\n🗣️ Real-time Transcription:")
    while not stop_flag.is_set():
        try:
            chunk = q_audio.get(timeout=0.1).squeeze()
            buffer = np.concatenate((buffer, chunk))
            if len(buffer) >= SAMPLE_RATE * 5:
                text = transcribe_fn(buffer, state)
                new_text = text[len(prev_text) :]
                print(new_text, end="", flush=True)
                prev_text = text
                buffer = np.zeros((0,), dtype=np.float32)
        except queue.Empty:
            continue
