import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import soundfile as sf

SAMPLE_RATE = 16000
MAX_LENGTH = 100  # Adjust as needed

class WhisperONNXDecoderWithCache:
    def __init__(self, encoder_path, decoder_path, tokenizer_repo="onnx-community/whisper-base", device='cpu'):
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']

        print(f"Loading encoder from {encoder_path}")
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        print(f"Loading decoder from {decoder_path}")
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

        print(f"Loading tokenizer from {tokenizer_repo}")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_repo)
        self.tokenizer = WhisperTokenizer.from_pretrained(tokenizer_repo)

        print(f"pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"bos_token_id: {self.tokenizer.bos_token_id}")
        print(f"eos_token_id: {self.tokenizer.eos_token_id}")

        self.decoder_start_token = 50258  # <|startoftranscript|>
        self.language_token = 50259       # <|en|>
        self.task_token = 50359           # <|transcribe|>
        self.eos_token = self.tokenizer.eos_token_id  # Usually 50257

        self.num_layers = 6  # Whisper base has 6 decoder layers

        print("Decoder input names:", [inp.name for inp in self.decoder.get_inputs()])
        print("Decoder output names:", [out.name for out in self.decoder.get_outputs()])

    def preprocess_audio(self, audio_path):
        print(f"Loading audio from {audio_path}")
        audio, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Expected {SAMPLE_RATE}Hz audio but got {sr}Hz")
        features = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        print(f"Extracted features shape: {features['input_features'].shape}")
        return features["input_features"]

    def encode(self, input_features):
        print(f"Running encoder ONNX")
        encoder_out = self.encoder.run(None, {"input_features": input_features})[0]
        print(f"Encoder output shape: {encoder_out.shape}")
        print(f"Encoder output sample values: {encoder_out.flatten()[:10]}")
        return encoder_out

    def initialize_past_key_values(self, batch_size, encoder_seq_len):
        print("Initializing past key values with zeros...")
        past_key_values = {}
        for i in range(self.num_layers):
            # Decoder past keys/values (empty initially)
            past_key_values[f"past_key_values.{i}.decoder.key"] = np.zeros((batch_size, 8, 0, 64), dtype=np.float32)
            past_key_values[f"past_key_values.{i}.decoder.value"] = np.zeros((batch_size, 8, 0, 64), dtype=np.float32)
            # Encoder past keys/values (fixed after encode)
            past_key_values[f"past_key_values.{i}.encoder.key"] = np.zeros((batch_size, 8, encoder_seq_len, 64), dtype=np.float32)
            past_key_values[f"past_key_values.{i}.encoder.value"] = np.zeros((batch_size, 8, encoder_seq_len, 64), dtype=np.float32)
        return past_key_values

    def decode(self, encoder_out):
        batch_size, encoder_seq_len, _ = encoder_out.shape

        # Initialize past key values
        past_key_values = self.initialize_past_key_values(batch_size, encoder_seq_len)

        # Set start tokens: <|startoftranscript|>, <|en|>, <|transcribe|>
        tokens = [self.decoder_start_token, self.language_token, self.task_token]
        print(f"Starting decode loop with tokens: {[self.tokenizer.decode([t]) for t in tokens]}")

        for step in range(MAX_LENGTH):
            if step == 0:
                # Step 0: full sequence
                input_ids = np.array([tokens], dtype=np.int64)
                use_cache = np.array([0], dtype=np.bool_)
            else:
                # Step >0: last token only
                input_ids = np.array([[tokens[-1]]], dtype=np.int64)
                use_cache = np.array([1], dtype=np.bool_)

            ort_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_out,
                "use_cache_branch": use_cache,
            }
            ort_inputs.update(past_key_values)

            print(f"\nStep {step}")
            print(f"Input ids: {input_ids}")
            print(f"use_cache_branch: {use_cache}")
            print(f"Past key values keys: {list(past_key_values.keys())}")

            outputs = self.decoder.run(None, ort_inputs)

            logits = outputs[0]
            print(f"logits shape: {logits.shape}")

            next_token = int(np.argmax(logits[0, -1]))  # Last position
            print(f"Next token: {next_token} ({self.tokenizer.decode([next_token])})")

            if next_token == self.eos_token:
                print("EOS token predicted, stopping decoding.")
                break

            tokens.append(next_token)

            # Update decoder past key/values only
            for i in range(self.num_layers):
                decoder_key_index = 1 + i * 4
                decoder_value_index = 2 + i * 4
                past_key_values[f"past_key_values.{i}.decoder.key"] = outputs[decoder_key_index]
                past_key_values[f"past_key_values.{i}.decoder.value"] = outputs[decoder_value_index]
                # Do NOT update encoder keys/values

        transcription = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        print("\n🗣️ Final decoded tokens:", tokens)
        print("📝 Transcription:", transcription)
        return transcription

    def transcribe(self, audio_path):
        input_features = self.preprocess_audio(audio_path)
        encoder_out = self.encode(input_features)
        return self.decode(encoder_out)


if __name__ == "__main__":
    ENCODER_PATH = "C:\\Users\\ISWAALEX\\Downloads\\encoder_model.onnx"
    DECODER_PATH = "C:\\Users\\ISWAALEX\\Downloads\\decoder_model_merged.onnx"
    AUDIO_PATH = "../../audio_files/1089-134686-0001.wav"

    model = WhisperONNXDecoderWithCache(ENCODER_PATH, DECODER_PATH)
    transcription = model.transcribe(AUDIO_PATH)
    print("\nFinal transcription:", transcription)
