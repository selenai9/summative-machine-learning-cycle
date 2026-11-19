"""Paths and settings. Everything can be overridden with environment variables (useful in Docker)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

MODEL_PATH = MODELS_DIR / "heart_cnn.keras"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
REPLAY_PATH = MODELS_DIR / "replay_set.npz"
ARCHIVE_DIR = MODELS_DIR / "archive"

UPLOAD_DIR = DATA_DIR / "uploads"          # new recordings waiting for retraining (uploads/normal, uploads/abnormal)
TRAIN_DIR = DATA_DIR / "train"             # recordings that were used for training (train/normal, train/abnormal)
DB_PATH = DATA_DIR / "app.db"

RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "20"))   # auto-retrain when this many new recordings are waiting
MAX_CLIPS_TRAIN = 3                        # clips taken from each uploaded recording
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MIN_PROMOTE_MARGIN = 0.01                  # new model is kept unless its validation AUC drops by more than this
MIN_RECALL = 0.85                          # target recall when choosing the threshold after retraining
