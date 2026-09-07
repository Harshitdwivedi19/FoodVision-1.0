"""
FoodVision 1.0 Configuration Module
Centralized configuration management for paths, model parameters, API endpoints,
and runtime settings.
"""

from pathlib import Path
from pydantic import BaseModel, Field
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ModelConfig(BaseModel):
    """Configuration for Deep Learning classification model."""
    default_model_name: str = "nateraw/food"
    fallback_model_name: str = "ashaduzzaman/vit-finetuned-food101"
    top_k: int = 5
    image_size: int = 224
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    confidence_threshold: float = 0.20


class LLMConfig(BaseModel):
    """Configuration for Local LLM Diagnosis service."""
    ollama_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3:latest"))
    timeout_seconds: float = 20.0
    fallback_to_clinical_engine: bool = True
    temperature: float = 0.3


class AppConfig(BaseModel):
    """Global Application Configuration."""
    app_name: str = "FoodVision 1.0"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    nutrition_db_path: Path = DATA_DIR / "nutrition_db.json"
    log_file_path: Path = LOGS_DIR / "foodvision.log"
    model: ModelConfig = ModelConfig()
    llm: LLMConfig = LLMConfig()


config = AppConfig()
