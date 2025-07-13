import os
import json
import torchaudio
import numpy as np
from lira.whisper.transcribe import WhisperONNX
from lira.zipformer.transcribe import EncoderWrapper, DecoderWrapper, JoinerWrapper, CHUNK_LEN
from lira.utils.audio import extract_fbank
from lira.utils.tokens import load_tokens
from jiwer import wer, cer
import argparse
from lira.cli.run_asr import greedy_search

SAMPLE_RATE = 16000

def transcribe_whisper(encoder_path, decoder_path, audio_path, providers):
    whisper_model = WhisperONNX(encoder_path, decoder_path, providers=providers)
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    audio = waveform.squeeze(0).numpy()
    return whisper_model.transcribe(audio)

def transcribe_zipformer(encoder_path, decoder_path, joiner_path, tokens_path, audio_path, providers):
    encoder = EncoderWrapper(encoder_path, providers=providers)
    decoder = DecoderWrapper(decoder_path, providers=providers)
    joiner = JoinerWrapper(joiner_path, providers=providers)
    tokens = load_tokens(tokens_path)

    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    audio = waveform.squeeze(0).numpy()
    features = extract_fbank(audio)

    state = {}
    return greedy_search(encoder, decoder, joiner, features, tokens, state)

def evaluate_models(dataset_path, ground_truths, model_configs):
    results = []
    for model_config in model_configs:
        model_type = model_config['type']
        print(f"Evaluating {model_type} model...")
        for wav_file, ground_truth in ground_truths.items():
            audio_path = os.path.join(dataset_path, wav_file+".wav")
            if model_type == 'whisper':
                transcription = transcribe_whisper(
                    model_config['encoder'],
                    model_config['decoder'],
                    audio_path,
                    model_config['providers']
                )
            elif model_type == 'zipformer':
                transcription = transcribe_zipformer(
                    model_config['encoder'],
                    model_config['decoder'],
                    model_config['joiner'],
                    model_config['tokens'],
                    audio_path,
                    model_config['providers']
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            wer_score = wer(ground_truth, transcription)
            cer_score = cer(ground_truth, transcription)
            results.append({
                'model': model_type,
                'file': wav_file,
                'wer': wer_score,
                'cer': cer_score,
                'transcription': transcription,
                'ground_truth': ground_truth
            })

    return results

def parse_ground_truth(ground_truth_path):
    ground_truths = {}
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wav_file, transcript = parts
                ground_truths[wav_file] = transcript
    return ground_truths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="Path to the folder containing .wav files")
    parser.add_argument("--ground-truth-path", required=True, help="Path to the ground truth .txt file")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    ground_truth_path = args.ground_truth_path

    ground_truths = parse_ground_truth(ground_truth_path)

    model_configs = [
        # {
        #     'type': 'whisper',
        #     'encoder': 'path/to/whisper-encoder.onnx',
        #     'decoder': 'path/to/whisper-decoder.onnx',
        #     'providers': ['CPUExecutionProvider']
        # },
        {
            'type': 'zipformer',
            'encoder': 'C:\\Users\\ISWAALEX\\DAToolkit\\sandbox\\asr_sandbox\\encoder.onnx',
            'decoder': 'C:\\Users\\ISWAALEX\\DAToolkit\\sandbox\\asr_sandbox\\decoder.onnx',
            'joiner': 'C:\\Users\\ISWAALEX\\DAToolkit\\sandbox\\asr_sandbox\\joiner.onnx',
            'tokens': 'C:\\Users\\ISWAALEX\\DAToolkit\\sandbox\\asr_sandbox\\tokens.txt',
            'providers': ['CPUExecutionProvider']
        }
    ]

    results = evaluate_models(dataset_path, ground_truths, model_configs)
    for result in results:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
