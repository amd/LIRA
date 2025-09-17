# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 
import unittest
import os
import torchaudio
from lira.models.whisper.transcribe import WhisperONNX
from lira.models.whisper.export import export_whisper_model
from lira.utils.config import get_provider

class TestWhisperONNX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Export the ONNX models using the export_model function directly

        cls.export_dir = "exported_models"
        export_whisper_model(
            model_name="whisper-base.en",
            output_dir=cls.export_dir,
            opset=17,
            static=True,
            force=True
        )

        cls._encoder = os.path.join(cls.export_dir, "encoder_model.onnx")
        cls._decoder = os.path.join(cls.export_dir, "decoder_model.onnx")
        cls._decoder_init = os.path.join(cls.export_dir, "decoder_init_model.onnx")
        cls._decoder_past = os.path.join(cls.export_dir, "decoder_with_past_model.onnx")

    def test_02_whisper_base_transcribe_npu_kv_cache(self):
        # Pass the audio file path directly
        audio_path = "audio_files/test.wav"

        # Get model providers for encoder and decoder based on device
        device = "npu"
        model = "whisper-base"

        whisper = WhisperONNX(
            encoder_path=self._encoder,
            decoder_path=self._decoder,
            decoder_init_path=self._decoder_init,
            decoder_past_path=self._decoder_past,
            encoder_provider=get_provider(device, model, "encoder", cache_dir=self.export_dir + "_vitisai_cache"),
            decoder_provider=get_provider("cpu", model, "decoder", cache_dir=self.export_dir + "_vitisai_cache"),
            decoder_init_provider=get_provider("cpu", model, "decoder_init", cache_dir=self.export_dir + "_vitisai_cache"),
            use_kv_cache=True
        )
        transcription, _ = whisper.transcribe(audio_path)
        print(transcription)
        self.assertIsInstance(transcription, str)
        self.assertGreater(len(transcription), 0)

if __name__ == "__main__":
    unittest.main()
