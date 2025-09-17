# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import os
from lira.models.whisper.transcribe import WhisperONNX, export_whisper_model
from pathlib import Path
from lira.utils.config import get_provider, get_cache_dir

app = FastAPI()
_model = None


def load_model():
    global _model
    model_dir = os.getenv("LIRA_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("LIRA_MODEL_DIR not set")

    encoder = os.path.join(model_dir, "encoder_model.onnx")
    decoder = os.path.join(model_dir, "decoder_model.onnx")
    decoder_init = os.path.join(model_dir, "decoder_init_model.onnx")
    decoder_past = os.path.join(model_dir, "decoder_with_past_model.onnx")
    device = "cpu"

    _model = WhisperONNX(
        encoder_path=encoder,
        decoder_path=decoder,
        decoder_init_path=decoder_init,
        decoder_past_path=decoder_past,
        encoder_provider=get_provider(
            device,
            "whisper",
            "encoder",
        ),
        decoder_provider=get_provider(
            device,
            "whisper",
            "decoder",
        ),
        decoder_init_provider=get_provider(
            device,
            "whisper",
            "decoder_init",
        ),
        use_kv_cache=True,
    )


@app.on_event("startup")
def _startup():
    load_model()


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile, model: str = Form(...)):
    if model not in ["whisper-1", "whisper-onnx"]:
        return JSONResponse({"error": "Unsupported model"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcription, _ = _model.transcribe(tmp_path)
        return JSONResponse({"text": transcription})
    finally:
        os.remove(tmp_path)


def get_app():
    return app


def setup_openai_server(model_type, device):
    if model_type.startswith("whisper-"):
        # Use centralized cache directory
        cache_dir = get_cache_dir() / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_dir = cache_dir / model_type
        print(f"Checking cache for model {model_type} at {output_dir}...")

        if not output_dir.exists():
            print(
                f"Exporting model {model_type} for device {device} to {output_dir}..."
            )
            export_whisper_model(
                model_name=model_type,
                output_dir=str(output_dir),
                opset=17,
                static=True,
            )
        else:
            print(f"Model {model_type} already exists in cache. Reusing cached model.")

        os.environ["LIRA_MODEL_DIR"] = str(output_dir)
        return get_app()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
