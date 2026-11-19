"""Locust flood test for the Heart Sound API.

Open the Locust web page:   locust -f locustfile.py --host http://localhost:8080
Headless (no browser):      locust -f locustfile.py --host http://localhost:8080 --headless -u 40 -r 10 -t 60s --csv results/flood

Every simulated user mostly uploads a heart-sound recording to /predict (the expensive request)
and sometimes calls /health. Sample recordings come from data/test/ (or a synthetic one if that is empty).
"""
import glob
import io
import os
import random

import numpy as np
import soundfile as sf
from locust import HttpUser, between, task


def load_samples(limit=10):
    root = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(root, "data", "test", "**", "*.wav"), recursive=True))[:limit]
    samples = [(os.path.basename(f), open(f, "rb").read()) for f in files]
    if not samples:                                           # fallback so the test still runs
        t = np.arange(0, 10, 1 / 2000)
        y = (np.sin(2 * np.pi * 50 * t) * (np.sin(2 * np.pi * 1.1 * t) > 0.95) + 0.05 * np.random.randn(len(t))).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, y, 2000, format="WAV")
        samples = [("synthetic.wav", buf.getvalue())]
    return samples


SAMPLES = load_samples()


class HeartSoundUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(8)
    def predict(self):
        name, data = random.choice(SAMPLES)
        self.client.post("/predict", files={"file": (name, data, "audio/wav")}, name="/predict")

    @task(2)
    def health(self):
        self.client.get("/health", name="/health")
