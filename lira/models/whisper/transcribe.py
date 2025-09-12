import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from lira.utils.audio import get_providers
import json
import torchaudio
from pathlib import Path
from jiwer import wer, cer
import time

SAMPLE_RATE = 16000
MAX_DECODE_STEPS = 100


class WhisperONNX:
    def __init__(
        self,
        encoder_path,
        decoder_path,
        encoder_provider,
        decoder_provider,
        decoder_init_path=None,
        decoder_past_path=None,
        model_type="whisper-base",
        use_kv_cache=False,
        debug=False,
    ):
        self.encoder = ort.InferenceSession(encoder_path, providers=encoder_provider)
        self.decoder = ort.InferenceSession(decoder_path, providers=decoder_provider)

        self.use_kv_cache = use_kv_cache
        if self.use_kv_cache:
            self.decoder_init = ort.InferenceSession(
                decoder_init_path, providers=decoder_provider
            )
            self.decoder_past = ort.InferenceSession(
                decoder_past_path, providers=decoder_provider
            )
            self.num_layers = (
                len(
                    [inp for inp in self.decoder_past.get_inputs() if "key" in inp.name]
                )
                // 4
            )

        # Determine tokenizer directory
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
        self.debug = debug
        self.decoder_start_token = self.sot_token = (
            self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        )
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = self.max_length = min(
            448, self.decoder.get_inputs()[0].shape[1]
        )

    def preprocess(self, audio):
        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
        )
        return inputs["input_features"]

    def encode(self, input_features):
        return self.encoder.run(None, {"input_features": input_features})[0]

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
        tokens = [self.sot_token]
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
                if self.debug:
                    print(
                        f"Step {step} → {next_token} ({self.tokenizer.decode([next_token])})"
                    )

                if next_token == self.eos_token:
                    print("EOS reached, stopping.")
                    break

                tokens.append(next_token)
                token_id = next_token
        else:
            print("Using non-KV Cache logic")
            for _ in range(self.max_length):
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

        transcription = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if self.debug:
            print("\n🗣️ Tokens:", tokens)
            print("📝 Transcription:", transcription)
        return tokens

    def transcribe(self, audio, chunk_length_s=30, is_mic=False):
        """
        Full encode-decode pipeline with support for long-form transcription using chunking.
        """
        chunk_size = SAMPLE_RATE * chunk_length_s
        total_samples = len(audio)
        transcription = []
        chunk_idx = 0
        total_start_time = time.time()

        overlap = SAMPLE_RATE * 1  # Tune this
        for start in range(0, total_samples, chunk_size - overlap):
            end = min(start + chunk_size, total_samples)
            audio_chunk = audio[start:end]

            # Process the chunk
            input_features = self.preprocess(audio_chunk)
            encoder_out = self.encode(input_features)
            tokens = self.decode(encoder_out)
            transcription.append(
                self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            )
            chunk_idx += 1

        total_end_time = time.time()
        input_audio_duration = total_samples / SAMPLE_RATE
        rtf = (total_end_time - total_start_time) / input_audio_duration

        # Combine all transcriptions
        return " ".join(transcription), rtf

    @staticmethod
    def parse_cli(subparsers):
        whisper_parser = subparsers.add_parser("whisper", help="Run Whisper model")
        whisper_parser.add_argument(
            "-m", "--model", required=True, help="Model name or path"
        )
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
            "--eval-dir", help="Dataset directory with wavs/ and transcripts.txt"
        )
        whisper_parser.add_argument(
            "--results-dir",
            default="results",
            help="Directory to store evaluation results",
        )
        whisper_parser.add_argument(
            "--chunk-length",
            type=int,
            default=30,
            help="Chunk length in seconds for long-form transcription",
        )
        whisper_parser.add_argument(
            "--use-kv-cache", action="store_true", help="Enable KV cache"
        )
        whisper_parser.set_defaults(func=WhisperONNX.run)

    @staticmethod
    def run(args):
        print(f"Running Whisper model: {args.model}")

        # Load providers based on device
        with open("config/model_config.json", "r") as f:
            model_config = json.load(f)
        providers = get_providers(args.device, model_config)

        whisper = WhisperONNX(
            encoder_path=f"{args.model}/encoder_model.onnx",
            decoder_path=f"{args.model}/decoder_model.onnx",
            decoder_init_path=f"{args.model}/decoder_init_model.onnx",
            decoder_past_path=f"{args.model}/decoder_with_past_model.onnx",
            encoder_provider=providers,
            decoder_provider=providers,
            model_type=args.model_type,
            use_kv_cache=args.use_kv_cache,
            debug=args.debug,
        )

        # Ensure at least one input is provided
        if not args.audio and not args.eval_dir:
            print("Error: You must provide either --audio or --eval_dir.")
            return

        if args.eval_dir:
            WhisperONNX.evaluate(whisper, args.eval_dir, args.results_dir)
            return

        if args.audio:
            waveform, sr = torchaudio.load(args.audio)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=SAMPLE_RATE
                )(waveform)
            audio = waveform.squeeze(0).numpy()
            transcription, rtf = whisper.transcribe(
                audio, chunk_length_s=args.chunk_length
            )
            print("\nTranscription:", transcription)
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
