import numpy as np
import onnxruntime as ort
import torchaudio
from lira.utils.config import get_provider
from lira.utils.audio import extract_fbank
from lira.utils.tokens import load_tokens
from lira.utils.audio import greedy_search
from lira.utils.cache import get_cache_dir

CHUNK_LEN = 151
BLANK_ID = 0
SAMPLE_RATE = 16000


class EncoderWrapper:
    def __init__(self, model_path, providers=None):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.streaming = any("cached_" in i.name for i in self.session.get_inputs())
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


class ZipformerONNX:
    def __init__(
        self,
        encoder_path,
        decoder_path,
        joiner_path,
        tokens,
        device,
        config_path="config/model_config.json",
    ):
        # Fetch providers from model configuration based on the device
        print("Using providers:", get_provider(device, "zipformer", "encoder"))
        self.encoder = EncoderWrapper(
            encoder_path, providers=get_provider(device, "zipformer", "encoder")
        )
        self.decoder = DecoderWrapper(
            decoder_path, providers=get_provider(device, "zipformer", "decoder")
        )
        self.joiner = JoinerWrapper(
            joiner_path, providers=get_provider(device, "zipformer", "joiner")
        )
        self.tokens_path = tokens
        self.device = device

    def transcribe(self, audio):
        print("Starting transcription with Zipformer...")

        # Preprocess audio
        waveform, sr = torchaudio.load(audio)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=SAMPLE_RATE
            )(waveform)
        audio = waveform.squeeze(0).numpy()
        features = extract_fbank(audio)

        # Initialize state for greedy search
        state = {}
        tokens = load_tokens(self.tokens_path)
        # Perform greedy search
        transcription = greedy_search(
            self.encoder, self.decoder, self.joiner, features, tokens, state
        )
        print("Transcription completed.")
        print("Transcription:", transcription)
        return transcription

    @staticmethod
    def parse_cli(subparsers):
        zipformer_parser = subparsers.add_parser(
            "zipformer", help="Run Zipformer model"
        )
        zipformer_parser.add_argument(
            "--model-dir", help="Path to model directory (local or Hugging Face repo)"
        )
        zipformer_parser.add_argument(
            "--encoder", help="Path to encoder model (overrides model-dir)"
        )
        zipformer_parser.add_argument(
            "--decoder", help="Path to decoder model (overrides model-dir)"
        )
        zipformer_parser.add_argument(
            "--joiner", help="Path to joiner model (overrides model-dir)"
        )
        zipformer_parser.add_argument(
            "--tokens", help="Path to tokens file (overrides model-dir)"
        )
        zipformer_parser.add_argument(
            "--audio", required=True, help="Path to audio file"
        )
        zipformer_parser.add_argument(
            "--device",
            choices=["cpu", "npu", "igpu"],
            default="cpu",
            help="Device to run the model on",
        )
        zipformer_parser.set_defaults(func=ZipformerONNX.run)

    @staticmethod
    def run(args):
        import os
        from huggingface_hub import snapshot_download

        if not args.model_dir and not (
            args.encoder and args.decoder and args.joiner and args.tokens
        ):
            raise ValueError(
                "Either --model-dir or all of --encoder, --decoder, --joiner, and --tokens must be provided."
            )

        model_dir = args.model_dir
        required_files = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]

        if model_dir:
            if os.path.isdir(model_dir):
                print("Using local model directory.")
            else:
                print("Downloading model from Hugging Face repository...")
                cache_dir = get_cache_dir() / "hf"
                model_dir = snapshot_download(
                    repo_id=model_dir, cache_dir=str(cache_dir)
                )

            for file in required_files:
                file_path = os.path.join(model_dir, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file not found: {file_path}")

        # Override with explicitly provided paths if available
        encoder_path = (
            args.encoder if args.encoder else os.path.join(model_dir, "encoder.onnx")
        )
        decoder_path = (
            args.decoder if args.decoder else os.path.join(model_dir, "decoder.onnx")
        )
        joiner_path = (
            args.joiner if args.joiner else os.path.join(model_dir, "joiner.onnx")
        )
        tokens_path = (
            args.tokens if args.tokens else os.path.join(model_dir, "tokens.txt")
        )

        print(f"Running Zipformer model on {args.device}")
        zipformer = ZipformerONNX(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            joiner_path=joiner_path,
            tokens=tokens_path,
            device=args.device,
        )
        zipformer.transcribe(args.audio)
