from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.json"

# DATASETS_DIR = PROJECT_ROOT / "datasets"
# EXPORTED_MODELS_DIR = PROJECT_ROOT / "exported_models"
