from setuptools import setup, find_packages

setup(
    name="lira",
    version="0.1.0",
    description="ASR package supporting Whisper and Zipformer models",
    author="Your Name",
    packages=find_packages(include=["lira", "lira.*"]),
    install_requires=[
        "numpy",
        "onnxruntime",
        "torch",
        "torchaudio",
        "sounddevice",
        "transformers"
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "run_asr=lira.cli.run_asr:main",
        ],
    },
)
