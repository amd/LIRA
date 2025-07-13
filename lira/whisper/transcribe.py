import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

SAMPLE_RATE = 16000

class WhisperONNX:
    def __init__(self, encoder_path, decoder_path, providers=None):
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
        self.decoder_start_token = self.tokenizer.pad_token_id  # Usually 50257
        self.eos_token = self.tokenizer.eos_token_id            # Usually 50256
        self.max_length = 448
        self.num_layers = 6           # Whisper-base has 6 decoder layers
        self.num_heads = 8            # As per your past_key_values shape
        self.head_dim = 64            # Last dim in past_key_values shape

    def preprocess(self, audio):
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return inputs["input_features"]

    def encode(self, input_features):
        # encoder output shape: (batch_size, encoder_seq_len/2, 512)
        return self.encoder.run(None, {"input_features": input_features})[0]

    def decode(self, encoder_out):
        batch_size = encoder_out.shape[0]
        encoder_seq_len = encoder_out.shape[1]

        tokens = [self.decoder_start_token]

        # Past key shapes
        zero_decoder_shape = (batch_size, self.num_heads, 0, self.head_dim)  # empty seq_len=0
        encoder_kv_shape = (batch_size, self.num_heads, encoder_seq_len, self.head_dim)

        past_key_values = {}
        for i in range(self.num_layers):
            past_key_values[f"past_key_values.{i}.decoder.key"] = np.zeros(zero_decoder_shape, dtype=np.float32)
            past_key_values[f"past_key_values.{i}.decoder.value"] = np.zeros(zero_decoder_shape, dtype=np.float32)
            past_key_values[f"past_key_values.{i}.encoder.key"] = np.zeros(encoder_kv_shape, dtype=np.float32)
            past_key_values[f"past_key_values.{i}.encoder.value"] = np.zeros(encoder_kv_shape, dtype=np.float32)

        for step in range(self.max_length):
            input_ids = np.array([[tokens[-1]]], dtype=np.int64)
            use_cache = np.array([step > 0], dtype=bool)

            ort_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_out,
                "use_cache_branch": use_cache,
                **past_key_values,
            }

            outputs = self.decoder.run(None, ort_inputs)

            logits = outputs[0]  # (batch, 1, vocab_size)
            next_token = int(np.argmax(logits[0, 0]))
            if next_token == self.eos_token:
                break
            tokens.append(next_token)

            # Update decoder past_key_values from outputs (starting at index 1)
            # Each layer has 2 outputs: key and value, so 2 * num_layers total
            for i in range(self.num_layers):
                past_key_values[f"past_key_values.{i}.decoder.key"] = outputs[1 + i * 2]
                past_key_values[f"past_key_values.{i}.decoder.value"] = outputs[2 + i * 2]

            # encoder key/values are static and unchanged

        return tokens


    def transcribe(self, audio):
        input_features = self.preprocess(audio)
        encoder_out = self.encode(input_features)
        tokens = self.decode(encoder_out)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
