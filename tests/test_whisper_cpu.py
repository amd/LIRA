import unittest
from lira.models.whisper.transcribe import WhisperONNX
import os
import subprocess
import torchaudio

class TestWhisperONNX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run the export script to generate ONNX models
        cls.export_dir = "exported_models"
        subprocess.run([
            "python", "lira/models/whisper/export.py",
            "--model_name", "openai/whisper-base.en",
            "--output_dir", cls.export_dir,
            "--opset", "17",
            "--static"
        ], check=True)

        cls._encoder = os.path.join(cls.export_dir, "encoder_model.onnx")
        cls._decoder = os.path.join(cls.export_dir, "decoder_model.onnx")
        cls._decoder_init = os.path.join(cls.export_dir, "decoder_init_model.onnx")
        cls._decoder_past = os.path.join(cls.export_dir, "decoder_with_past_model.onnx")

    def test_01_whisper_base_transcribe_cpu(self):
        # Load real audio file for testing
        audio_path = "audio_files/61-70968-0000.wav"
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        audio = waveform.squeeze(0).numpy()

        whisper = WhisperONNX(
            encoder_path=self._encoder,
            decoder_path=self._decoder,
            encoder_provider=["CPUExecutionProvider"],
            decoder_provider=["CPUExecutionProvider"],
            model_type="whisper-base"
        )
        transcription, _ = whisper.transcribe(audio)
        print(transcription)
        self.assertIsInstance(transcription, str)
        self.assertGreater(len(transcription), 0)

    def test_02_whisper_base_transcribe_cpu_kv_cache(self):
        # Load real audio file for testing
        audio_path = "audio_files/61-70968-0000.wav"
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        audio = waveform.squeeze(0).numpy()

        whisper = WhisperONNX(
            encoder_path=self._encoder,
            decoder_path=self._decoder,
            decoder_init_path=self._decoder_init,
            decoder_past_path=self._decoder_past,
            encoder_provider=["CPUExecutionProvider"],
            decoder_provider=["CPUExecutionProvider"],
            model_type="whisper-base",
            use_kv_cache=True
        )
        transcription, _ = whisper.transcribe(audio)
        print(transcription)
        self.assertIsInstance(transcription, str)
        self.assertGreater(len(transcription), 0)

if __name__ == "__main__":
    unittest.main()
