import unittest
from fastapi.testclient import TestClient
from lira.server.openai_server import setup_openai_server
from lira.server import openai_server
import os


class TestOpenAIServerReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
       
        cls.app = setup_openai_server("whisper-base.en", "cpu")
        cls.client = TestClient(cls.app)
        openai_server.load_model()

    def test_transcribe_sample_wav(self):
        sample_path = os.path.join("audio_files", "test.wav")
        with open(sample_path, "rb") as fh:
            files = {"file": ("test.wav", fh, "audio/wav")}
            data = {"model": "whisper-onnx"}
            resp = self.client.post("/v1/audio/transcriptions", files=files, data=data)

        self.assertEqual(resp.status_code, 200)
        json = resp.json()
        self.assertIn("text", json)
        self.assertIsInstance(json["text"], str)
        self.assertGreater(len(json["text"]), 0)

    def test_unsupported_model(self):
        sample_path = os.path.join("audio_files", "test.wav")
        with open(sample_path, "rb") as fh:
            files = {"file": ("test.wav", fh, "audio/wav")}
            data = {"model": "unsupported-model"}
            resp = self.client.post("/v1/audio/transcriptions", files=files, data=data)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())


if __name__ == "__main__":
    unittest.main()
