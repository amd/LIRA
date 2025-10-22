from fastapi import FastAPI, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
import tempfile
import os
from lira.models.whisper.transcribe import WhisperONNX, export_whisper_model
from pathlib import Path
from lira.utils.config import get_provider, get_cache_dir

class WhisperService:
    def __init__(self, model_type: str, device: str):
        self.model_type = model_type
        self.device = device
        self.model_dir = None
        self.model = None
        self._setup_model_dir_and_export()

    def _setup_model_dir_and_export(self):
        # Use centralized cache directory
        cache_dir = get_cache_dir() / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir = cache_dir / self.model_type

        if not output_dir.exists():
            print(f"Exporting model {self.model_type} for device {self.device} to {output_dir}...")
            export_whisper_model(
                model_name=self.model_type,
                output_dir=str(output_dir),
                opset=17,
                static=True,
            )
        else:
            print(f"Model {self.model_type} already exists in cache. Reusing cached model.")
        self.model_dir = str(output_dir)
        self._load_model()

    def _load_model(self):
        encoder = os.path.join(self.model_dir, "encoder_model.onnx")
        decoder = os.path.join(self.model_dir, "decoder_model.onnx")
        decoder_init = os.path.join(self.model_dir, "decoder_init_model.onnx")
        decoder_past = os.path.join(self.model_dir, "decoder_with_past_model.onnx")

        self.model = WhisperONNX(
            encoder_path=encoder,
            decoder_path=decoder,
            decoder_init_path=decoder_init,
            decoder_past_path=decoder_past,
            encoder_provider=get_provider(self.device, "whisper", "encoder"),
            decoder_provider=get_provider(self.device, "whisper", "decoder"),
            decoder_init_provider=get_provider(self.device, "whisper", "decoder_init"),
            use_kv_cache=True,
        )

    def transcribe(self, wav_path: str):
        transcription, _ = self.model.transcribe(wav_path)
        return transcription

def create_app(model_type: str, device: str):
    app = FastAPI()
    whisper_service = WhisperService(model_type, device)

    def get_whisper_service():
        return whisper_service

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile,
        model: str = Form(...),
        svc: WhisperService = Depends(get_whisper_service)
    ):
        if model not in ["whisper-1", "whisper-onnx"]:
            return JSONResponse({"error": "Unsupported model"}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            text = svc.transcribe(tmp_path)
            return JSONResponse({"text": text})
        finally:
            os.remove(tmp_path)

    return app

# Entry function (to match your OpenAI-style signature)
def setup_openai_server(model_type: str, device: str):
    if not model_type.startswith("whisper-"):
        raise ValueError(f"Unsupported model type: {model_type}")
    return create_app(model_type=model_type, device=device)
