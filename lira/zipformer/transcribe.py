import numpy as np
import onnxruntime as ort

CHUNK_LEN = 151
BLANK_ID = 0

class EncoderWrapper:
    def __init__(self, model_path, providers=None):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.streaming = any('cached_' in i.name for i in self.session.get_inputs())
        self.cache = self._init_cache() if self.streaming else None

    def _init_cache(self):
        cache = {}
        for inp in self.session.get_inputs():
            if inp.name == 'x':
                continue
            shape = [1 if isinstance(s, str) else s for s in inp.shape]
            dtype = np.float32 if 'float' in inp.type else np.int64
            canonical_name = inp.name.replace('cached_', '')
            cache[canonical_name] = np.zeros(shape, dtype=dtype)
        return cache

    def run(self, x):
        inputs = {"x": x.astype(np.float32)}
        if self.streaming:
            expanded_cache = {f"cached_{k}": v for k, v in self.cache.items()}
            inputs.update(expanded_cache)
        outs = self.session.run(None, inputs)
        if self.streaming:
            self.cache = {o.name.replace('new_cached_', ''): val
                          for o, val in zip(self.session.get_outputs()[1:], outs[1:])}
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
        return self.session.run(None, {
            "encoder_out": encoder_out,
            "decoder_out": decoder_out
        })[0]
