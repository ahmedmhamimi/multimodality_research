from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "week5-multimodal-secret")
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024

    DEVICE = os.getenv("DEVICE", "cpu")
    ENABLE_REMOTE_MODEL_DOWNLOAD = os.getenv("ENABLE_REMOTE_MODEL_DOWNLOAD", "false").lower() == "true"

    TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH", str(BASE_DIR / "models" / "bert_multiwoz_best"))
    IMAGE_MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", str(BASE_DIR / "models" / "sir_yes_sir.pt"))
    AUDIO_MODEL_PATH = os.getenv("AUDIO_MODEL_PATH", str(BASE_DIR / "models" / "wav2vec2-base"))
    AUDIO_CHECKPOINT_PATH = os.getenv("AUDIO_CHECKPOINT_PATH", str(BASE_DIR / "models" / "best_model.pt"))

    DEFAULT_TEXT_MODEL_NAME = os.getenv("DEFAULT_TEXT_MODEL_NAME", "bert-base-uncased")
    DEFAULT_IMAGE_MODEL_NAME = os.getenv("DEFAULT_IMAGE_MODEL_NAME", "openai/clip-vit-base-patch32")
    DEFAULT_AUDIO_MODEL_NAME = os.getenv("DEFAULT_AUDIO_MODEL_NAME", "facebook/wav2vec2-base-960h")

    EVAL_CASES_PATH = BASE_DIR / "data" / "sample_eval_cases.json"
    FUSION_OUTPUT_DIM = 512
    RANDOM_SEED = 42

    FUSION_METHODS = ["concatenation", "cross_attention", "late_fusion"]
    TASK_OPTIONS = [
        "auto",
        "detailed_response",
        "classification",
        "task_execution",
        "retrieval_summary",
    ]
    EXPLAINABLE_METRICS = [
        "task_success_rate",
        "precision",
        "recall",
        "f1",
        "average_task_time_seconds",
        "cross_modal_relevance",
        "modality_coverage",
    ]
