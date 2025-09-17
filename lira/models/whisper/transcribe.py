# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import time
import json
import torchaudio
from pathlib import Path
from jiwer import wer, cer

from lira.utils.config import get_provider, get_cache_dir
from lira.models.whisper.export import export_whisper_model

SAMPLE_RATE = 16000
MAX_DECODE_STEPS = 100


class WhisperONNX:
    def __init__(
        self,
        encoder_path,
        decoder_path,
        encoder_provider,
        decoder_provider,
        decoder_init_provider,
        decoder_init_path=None,
        decoder_past_path=None,
        use_kv_cache=False,
        profile=False,
    ):
        self.use_kv_cache = use_kv_cache

        # Initialize encoder FIRST to match run_whisper.py order
        self.encoder = ort.InferenceSession(encoder_path, providers=encoder_provider)

        if self.use_kv_cache:
            self.decoder_init = ort.InferenceSession(
                decoder_init_path,
                providers=decoder_init_provider,
            )
            self.decoder_past = ort.InferenceSession(
                decoder_past_path,
                providers=["CPUExecutionProvider"],  # Not supported on NPU
            )
            self.num_layers = (
                len(
                    [inp for inp in self.decoder_past.get_inputs() if "key" in inp.name]
                )
                // 4
            )
        else:
            self.decoder = ort.InferenceSession(
                decoder_path,
                providers=decoder_provider,
            )
            self.max_length = min(448, self.decoder.get_inputs()[0].shape[1])
            if not isinstance(self.max_length, int):
                raise ValueError("Invalid/Dynamic input shapes")

        tokenizer_dir = Path(encoder_path).parent
        print(
            f"\nLoading tokenizer and feature extractor from: {Path(tokenizer_dir).resolve()}"
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            tokenizer_dir, local_files_only=True
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True
        )
        self.profile = profile
        self.decoder_start_token = self.sot_token = (
            self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        )
        self.eos_token = self.tokenizer.eos_token_id

    def preprocess(self, audio):
        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
        )
        return inputs["input_features"]

    def encode(self, input_features):
        result = self.encoder.run(None, {"input_features": input_features})[0]
        return result

    def _extract_past(self, past_outputs):
        pkv = {}
        num_tensors = len(past_outputs)

        if num_tensors == 4 * self.num_layers:
            # Full encoder + decoder KV
            idx = 0
            for i in range(self.num_layers):
                pkv[f"past_key_values.{i}.decoder.key"] = past_outputs[idx]
                idx += 1
                pkv[f"past_key_values.{i}.decoder.value"] = past_outputs[idx]
                idx += 1
                pkv[f"past_key_values.{i}.encoder.key"] = past_outputs[idx]
                idx += 1
                pkv[f"past_key_values.{i}.encoder.value"] = past_outputs[idx]
                idx += 1

        elif num_tensors == 2 * self.num_layers:
            # Only decoder KV
            idx = 0
            for i in range(self.num_layers):
                pkv[f"past_key_values.{i}.decoder.key"] = past_outputs[idx]
                idx += 1
                pkv[f"past_key_values.{i}.decoder.value"] = past_outputs[idx]
                idx += 1

        else:
            raise RuntimeError(
                f"Unexpected number of past tensors: {num_tensors}. Expected {4 * self.num_layers} or {2 * self.num_layers}."
            )

        return pkv

    def decode(self, encoder_out):
        tokens = [self.decoder_start_token]
        token_id = self.sot_token
        past_kv = None
        encoder_kv = {}

        if self.use_kv_cache:
            print("Using KV Cache supported models")
            for step in range(MAX_DECODE_STEPS):
                if step == 0:
                    ort_inputs = {
                        "input_ids": np.array([[token_id]], dtype=np.int64),
                        "encoder_hidden_states": encoder_out,
                    }
                    outputs = self.decoder_init.run(None, ort_inputs)
                else:
                    ort_inputs = {"input_ids": np.array([[token_id]], dtype=np.int64)}
                    ort_inputs.update(past_kv)
                    ort_inputs.update(encoder_kv)
                    outputs = self.decoder_past.run(None, ort_inputs)

                logits = outputs[0]
                pkv = self._extract_past(outputs[1:])

                decoder_only = {}
                for k, v in pkv.items():
                    if ".encoder." in k:
                        if step == 0:
                            encoder_kv[k] = v  # IA store once
                    else:
                        decoder_only[k] = v

                past_kv = decoder_only

                next_token = int(np.argmax(logits[0, -1]))
                if next_token == self.eos_token:
                    print("EOS reached, stopping.")
                    break

                tokens.append(next_token)
                token_id = next_token
        else:
            print(f"Using non-KV Cache logic")
            for i in range(self.max_length):
                decoder_input = np.full(
                    (1, self.max_length), self.eos_token, dtype=np.int64
                )
                decoder_input[0, : len(tokens)] = tokens
                logits = self.decoder.run(
                    None,
                    {"input_ids": decoder_input, "encoder_hidden_states": encoder_out},
                )[0]
                next_token = int(np.argmax(logits[0, len(tokens) - 1]))
                if next_token == self.eos_token:
                    break
                tokens.append(next_token)

        return tokens

    def transcribe(self, audio_input):
        """
        Transcribe the given audio file or audio data.
        """
        if isinstance(audio_input, (str, Path)):
            # File path input
            input_features, audio_duration = self.preprocess_audio(audio_input)
        else:
            # Raw audio data input (numpy array)
            input_features = self.preprocess(audio_input)
            audio_duration = len(audio_input) / SAMPLE_RATE

        # Time only the decoding step if profiling is enabled
        start_time = time.time() if self.profile else None
        encoder_out = self.encode(input_features)
        tokens = self.decode(encoder_out)
        end_time = time.time() if self.profile else None

        transcription = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

        rtf = (end_time - start_time) / audio_duration if self.profile else None

        return transcription, rtf

    def preprocess_audio(self, audio_path):
        """
        Preprocess the audio file by loading, resampling, and extracting features.
        Returns input features and audio duration.
        """
        print(f"Loading audio from {audio_path}")
        audio, sr = torchaudio.load(audio_path)
        audio_duration = audio.shape[1] / sr  # Calculate duration in seconds

        # Resample if not 16k
        if sr != SAMPLE_RATE:
            print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz")
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(
                audio
            )
            sr = SAMPLE_RATE
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Expected {SAMPLE_RATE}Hz audio but got {sr}Hz")

        features = self.feature_extractor(
            audio.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="np"
        )
        print(f"Extracted features shape: {features['input_features'].shape}")

        return features["input_features"], audio_duration

    @staticmethod
    def parse_cli(subparsers):
        whisper_parser = subparsers.add_parser("whisper", help="Run Whisper model")
        whisper_parser.add_argument("-m", "--model", help="Model name or path")
        whisper_parser.add_argument("--audio", help="Path to audio file")
        whisper_parser.add_argument(
            "--model-type",
            default="whisper-base",
            help="Model type (default: whisper-base)",
        )
        whisper_parser.add_argument(
            "--device",
            default="cpu",
            choices=["cpu", "npu", "igpu"],
            help="Device to run the model on (default: cpu)",
        )
        whisper_parser.add_argument(
            "--cache",
            default=None,
            help="Path to cache directory (default: ~/.cache/lira)",
        )
        whisper_parser.add_argument(
            "--eval-dir", help="Dataset directory with wavs/ and transcripts.txt"
        )
        whisper_parser.add_argument(
            "--results-dir",
            default="results",
            help="Directory to store evaluation results",
        )
        whisper_parser.add_argument(
            "--use-kv-cache", action="store_true", help="Enable KV cache"
        )
        whisper_parser.add_argument(
            "--profile", action="store_true", help="Enable profiling mode"
        )
        whisper_parser.add_argument(
            "--export",
            action="store_true",
            help="Export the model instead of running it",
        )
        whisper_parser.add_argument(
            "--export-dir",
            default=None,
            help="Directory to export the model",
        )
        whisper_parser.add_argument(
            "--opset", type=int, default=17, help="ONNX opset version (default: 17)"
        )
        whisper_parser.add_argument(
            "--static",
            default=True,
            action="store_true",
            help="Use static shape parameters for export",
        )
        whisper_parser.set_defaults(func=WhisperONNX.run)

    @staticmethod
    def run(args):
        print(f"Running Whisper model: {args.model_type}")
        if args.model:
            print("Model Location: ", args.model)

        if not args.model and not args.export:
            print(
                "Error: You must provide a model "
                "directory with -m/--model or use --export to export a model first."
            )
            return

        if args.export:
            if args.export_dir is None:
                args.export_dir = get_cache_dir() / "models" / args.model_type
            print("Exporting model...")
            export_whisper_model(
                model_name=f"openai/{args.model_type}",
                output_dir=args.export_dir,
                opset=args.opset,
                static=args.static,
            )
            print("Model export completed.")
            args.model = args.export_dir

        # Load providers based on device
        whisper = WhisperONNX(
            encoder_path=f"{args.model}/encoder_model.onnx",
            decoder_path=f"{args.model}/decoder_model.onnx",
            decoder_init_path=f"{args.model}/decoder_init_model.onnx",
            decoder_past_path=f"{args.model}/decoder_with_past_model.onnx",
            encoder_provider=get_provider(
                args.device, args.model_type, "encoder", cache_dir=args.cache
            ),
            decoder_provider=get_provider(
                args.device, args.model_type, "decoder", cache_dir=args.cache
            ),
            decoder_init_provider=get_provider(
                args.device, args.model_type, "decoder_init", cache_dir=args.cache
            ),
            use_kv_cache=args.use_kv_cache,
            profile=args.profile,
        )

        # Ensure at least one input is provided
        if not args.audio and not args.eval_dir:
            print("Error: You must provide either --audio or --eval_dir.")
            return

        if args.eval_dir:
            WhisperONNX.evaluate(whisper, args.eval_dir, args.results_dir)
            return

        if args.audio:
            transcription, rtf = whisper.transcribe(
                args.audio,
            )
            print("\nTranscription:", transcription)
            if args.profile:
                print(f"Real-Time Factor (RTF): {rtf:.2f}")

    @staticmethod
    def evaluate(model, dataset_dir, results_dir):
        dataset_name = Path(dataset_dir).name
        wav_dir = Path(dataset_dir) / "wav"
        transcript_file = Path(dataset_dir) / "transcripts.txt"

        if not transcript_file.exists() or not wav_dir.exists():
            print(f"Missing transcripts.txt or wav folder in {dataset_dir}")
            return

        with open(transcript_file, "r", encoding="utf-8") as f:
            references = {
                line.split()[0]: " ".join(line.strip().split()[1:])
                for line in f.readlines()
            }

        output_dir = Path(results_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / "results.txt"

        total_wer, total_cer, total_rtf, count = 0, 0, 0, 0

        with result_file.open("w", encoding="utf-8") as out_f:
            for wav_path in sorted(wav_dir.glob("*.wav")):
                key = wav_path.stem
                if key not in references:
                    print(f"Reference for {key} not found in transcripts.txt")
                    continue
                reference = references[key].lower()
                waveform, sr = torchaudio.load(str(wav_path))
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=SAMPLE_RATE
                    )(waveform)
                audio = waveform.squeeze(0).numpy()
                predicted, rtf = model.transcribe(audio)

                sample_wer = wer(reference, predicted)
                sample_cer = cer(reference, predicted)
                total_wer += sample_wer
                total_cer += sample_cer
                total_rtf += rtf
                count += 1

                out_f.write(f"{key}\n")
                out_f.write(f"Reference: {reference}\n")
                out_f.write(f"Predicted: {predicted}\n")
                out_f.write(
                    f"WER: {sample_wer:.3f}, CER: {sample_cer:.3f}, RTF: {rtf:.3f}\n\n"
                )

            if count:
                avg_wer = total_wer / count
                avg_cer = total_cer / count
                avg_rtf = total_rtf / count
                print(f"Evaluation completed for {count} files.")
                print(
                    f"Average WER: {avg_wer:.3f}, Average CER: {avg_cer:.3f}, Average RTF: {avg_rtf:.3f}"
                )
                out_f.write(
                    f"Summary:\nAverage WER: {avg_wer:.3f}\nAverage CER: {avg_cer:.3f}\nAverage RTF: {avg_rtf:.3f}\n"
                )
            else:
                print("No valid audio-transcript pairs found.")
