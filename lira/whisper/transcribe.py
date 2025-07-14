import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

SAMPLE_RATE = 16000

class WhisperONNX:
    def __init__(self, encoder_path, decoder_path, encoder_provider, decoder_provider):
        self.encoder = ort.InferenceSession(encoder_path, providers=encoder_provider)
        self.decoder = ort.InferenceSession(decoder_path, providers=decoder_provider)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
        self.decoder_start_token = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = 448

    def preprocess(self, audio):
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return inputs["input_features"]

    def encode(self, input_features):
        return self.encoder.run(None, {"input_features": input_features})[0]

    def decode(self, encoder_out):
        tokens = [self.decoder_start_token]
        for _ in range(self.max_length):
            decoder_input = np.full((1, self.max_length), self.eos_token, dtype=np.int64)
            decoder_input[0, :len(tokens)] = tokens

            logits = self.decoder.run(None, {
                "input_ids": decoder_input,
                "encoder_hidden_states": encoder_out
            })[0]
            next_token = int(np.argmax(logits[0, len(tokens)-1]))
            if next_token == self.eos_token:
                break
            tokens.append(next_token)
        return tokens

    def transcribe(self, audio):
        input_features = self.preprocess(audio)
        encoder_out = self.encode(input_features)
        tokens = self.decode(encoder_out)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()