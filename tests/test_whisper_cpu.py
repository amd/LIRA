import unittest
from lira.models.whisper.transcribe import WhisperONNX
from lira.models.whisper.export import export_whisper_model
import os
import torchaudio

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

    def test_01_whisper_base_transcribe_cpu(self):
        # Pass the audio file path directly
        audio_path = "audio_files/61-70968-0000.wav"

        whisper = WhisperONNX(
            encoder_path=self._encoder,
            decoder_path=self._decoder,
            encoder_provider=["CPUExecutionProvider"],
            decoder_provider=["CPUExecutionProvider"],
            decoder_init_provider=["CPUExecutionProvider"]
        )
        transcription, _ = whisper.transcribe(audio_path)
        print(transcription)
        self.assertIsInstance(transcription, str)
        self.assertGreater(len(transcription), 0)

    def test_02_whisper_base_transcribe_cpu_kv_cache(self):
        # Pass the audio file path directly
        audio_path = "audio_files/61-70968-0000.wav"

        whisper = WhisperONNX(
            encoder_path=self._encoder,
            decoder_path=self._decoder,
            decoder_init_path=self._decoder_init,
            decoder_past_path=self._decoder_past,
            encoder_provider=["CPUExecutionProvider"],
            decoder_provider=["CPUExecutionProvider"],
            decoder_init_provider=["CPUExecutionProvider"],
            use_kv_cache=True
        )
        transcription, _ = whisper.transcribe(audio_path)
        print(transcription)
        self.assertIsInstance(transcription, str)
        self.assertGreater(len(transcription), 0)

if __name__ == "__main__":
    unittest.main()
