"""Prediction for one heart-sound recording (used by the notebook and the API)."""
import os
import threading
import time

import numpy as np

from .preprocessing import load_audio, make_clips, quality_report, scale_spec, to_melspec
from .model import load_json, load_model

DISCLAIMER = ("Educational demo, not a medical device. A normal result does not rule out heart disease; "
              "please see a clinician.")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
MODEL_PATH = os.path.join(MODELS_DIR, "heart_cnn.keras")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.json")


def predict_recording(model, threshold, source, max_clips=12):
    """Cut a recording into 5 s clips, predict each, and average the probabilities.

    `source` is a file path. Returns a dict with the label, probability and quality warnings.
    """
    raw = load_audio(source, bandpass=False)
    report = quality_report(raw)
    y = load_audio(source, bandpass=True)
    clips = make_clips(y, max_clips=max_clips)
    X = np.stack([scale_spec(to_melspec(c)) for c in clips])[..., np.newaxis]
    clip_probs = np.asarray(model(X, training=False)).ravel()
    prob = float(clip_probs.mean())
    return {
        "label": "abnormal" if prob >= threshold else "normal",
        "probability_abnormal": round(prob, 4),
        "threshold": round(float(threshold), 4),
        "n_clips": int(len(clips)),
        "clip_probabilities": [round(float(p), 4) for p in clip_probs],
        "duration_s": report["duration_s"],
        "warnings": report["warnings"],
        "disclaimer": DISCLAIMER,
    }


class Predictor:
    """Holds the current model in memory and reloads it automatically when the model file changes.

    This matters for retraining: when a new model is saved, every running copy of the API picks it up
    on the next request without a restart.
    """

    def __init__(self, model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH):
        self.model_path = model_path
        self.threshold_path = threshold_path
        self.model = None
        self.threshold = 0.5
        self._mtimes = (None, None)
        self._lock = threading.Lock()
        self.loaded_at = None

    def _current_mtimes(self):
        def mt(p):
            return os.path.getmtime(p) if os.path.exists(p) else None
        return mt(self.model_path), mt(self.threshold_path)

    def ensure_fresh(self):
        """Load (or reload) the model if the files on disk changed. Returns True if a model is available."""
        mtimes = self._current_mtimes()
        if mtimes[0] is None:
            return False
        if self.model is None or mtimes != self._mtimes:
            with self._lock:
                mtimes = self._current_mtimes()
                if self.model is None or mtimes != self._mtimes:
                    self.model = load_model(self.model_path)
                    info = load_json(self.threshold_path, {}) or {}
                    self.threshold = float(info.get("threshold", 0.5))
                    self._mtimes = mtimes
                    self.loaded_at = time.time()
        return True

    def predict(self, source, max_clips=12):
        if not self.ensure_fresh():
            raise FileNotFoundError("No trained model found. Train it in the notebook and put it in models/.")
        return predict_recording(self.model, self.threshold, source, max_clips=max_clips)
