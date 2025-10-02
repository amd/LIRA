# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import unittest
from lira.models.zipformer.transcribe import ZipformerONNX
from huggingface_hub import snapshot_download
import os

class TestZipformerONNX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use Hugging Face to download a small model for testing
        cls.model_dir = snapshot_download(repo_id="aigdat/AMD-zipformer-en")
        cls.encoder_path = os.path.join(cls.model_dir, "encoder.onnx")
        cls.decoder_path = os.path.join(cls.model_dir, "decoder.onnx")
        cls.joiner_path = os.path.join(cls.model_dir, "joiner.onnx")
        cls.tokens_path = os.path.join(cls.model_dir, "tokens.txt")

    def test_transcribe_npu(self):
        zipformer = ZipformerONNX(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            joiner_path=self.joiner_path,
            tokens=self.tokens_path,
            device="npu"
        )
        transcription = zipformer.transcribe("audio_files/test.wav")

        self.assertIsInstance(transcription, str)
        self.assertGreater(len(transcription), 0)

if __name__ == "__main__":
    unittest.main()
