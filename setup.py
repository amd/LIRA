# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from setuptools import setup, find_packages

setup(
    name="lira",
    version="0.1.0",
    description="ASR package supporting Whisper and Zipformer models",
    author="Your Name",
    packages=find_packages(include=["lira", "lira.*"]),
    install_requires=[
        # "numpy",
        # "onnxruntime",
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "sounddevice",
        "transformers==4.52.4",
        "soundfile",
        "gradio",
        "jiwer",
        # dev tools
        "onnx",
        "optimum==1.26.1",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "run_asr=lira.scripts.run_asr:main",
            "lira=lira.cli:main",
        ],
    },
)
