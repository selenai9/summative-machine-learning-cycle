"""FastAPI service: predict, upload labelled recordings, retrain, health/model info.

Run:  uvicorn app.api:app --host 0.0.0.0 --port 8000      (interactive docs at /docs)
"""
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import db, retrain
from .config import (MAX_UPLOAD_MB, METRICS_PATH, MODEL_PATH, RETRAIN_THRESHOLD, THRESHOLD_PATH, UPLOAD_DIR)
from src.model import load_json
from src.prediction import DISCLAIMER, Predictor

ALLOWED_EXT = {".wav", ".flac", ".ogg"}
LABELS = {"normal", "abnormal"}
START_TIME = time.time()
STATS = {"predictions": 0, "errors": 0, "total_latency_ms": 0.0}
_stats_lock = threading.Lock()
predictor = Predictor(str(MODEL_PATH), str(THRESHOLD_PATH))


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    try:
        predictor.ensure_fresh()           # load the model once at start-up (skipped if there is no model yet)
    except Exception as exc:
        print("Could not load model at start-up:", exc)
    yield


app = FastAPI(title="Heart Sound Classifier API", version="1.0", lifespan=lifespan)


# ------------------------------------------------------------------ helpers
def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "file.wav").name)[:80]


async def _read_limited(upload: UploadFile, limit_mb=MAX_UPLOAD_MB):
    data = await upload.read()
    if len(data) > limit_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (limit {limit_mb} MB).")
    return data


def _save_recording(data: bytes, filename: str, label: str):
    """Validate an audio file and store it in data/uploads/<label>/. Returns (ok, message)."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return False, f"{filename}: unsupported type {ext or '(none)'}; use .wav, .flac or .ogg"
    dest_dir = UPLOAD_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex[:8]}_{_safe_name(filename)}"
    dest.write_bytes(data)
    try:
        info = sf.info(str(dest))
        if info.duration < 1.0:
            raise ValueError("shorter than 1 second")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return False, f"{filename}: not a usable audio file ({exc})"
    db.add_upload(filename, label, dest, round(info.duration, 2))
    return True, f"{filename}: saved as {label}"


def _maybe_auto_retrain():
    """Retraining trigger: start automatically once enough new recordings are waiting."""
    pending = len(db.pending_uploads())
    if pending >= RETRAIN_THRESHOLD and MODEL_PATH.exists() and not retrain.STATE["running"]:
        return retrain.start_retrain(f"auto: {pending} new recordings reached the threshold of {RETRAIN_THRESHOLD}")
    return False


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    """Uptime and status (the UI shows this as 'model uptime')."""
    up = time.time() - START_TIME
    metrics = load_json(METRICS_PATH, {}) or {}
    with _stats_lock:
        n = STATS["predictions"]
        avg = STATS["total_latency_ms"] / n if n else None
    return {
        "status": "ok",
        "uptime_seconds": round(up, 1),
        "started_at": datetime.fromtimestamp(START_TIME, timezone.utc).isoformat(timespec="seconds"),
        "model_file_present": MODEL_PATH.exists(),
        "model_loaded": predictor.model is not None,
        "model_version": metrics.get("current_version", 0),
        "predictions_served": n,
        "prediction_errors": STATS["errors"],
        "avg_prediction_ms": round(avg, 1) if avg else None,
        "retraining_running": retrain.STATE["running"],
    }


@app.get("/model")
def model_info():
    """Saved evaluation numbers, current threshold and retraining history."""
    metrics = load_json(METRICS_PATH, {}) or {}
    thr = load_json(THRESHOLD_PATH, {}) or {}
    return {"threshold": thr, "metrics": metrics, "versions": db.list_versions()[:20]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload ONE heart-sound recording and get normal / abnormal."""
    data = await _read_limited(file, limit_mb=30)
    suffix = Path(file.filename or "x.wav").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, "Please upload a .wav, .flac or .ogg file.")
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = predictor.predict(tmp_path)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        with _stats_lock:
            STATS["errors"] += 1
        raise HTTPException(400, str(exc))
    finally:
        os.unlink(tmp_path)
    ms = (time.time() - t0) * 1000
    with _stats_lock:
        STATS["predictions"] += 1
        STATS["total_latency_ms"] += ms
    result["latency_ms"] = round(ms, 1)
    return result


@app.post("/upload")
async def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    """Upload several recordings that all share one label ('normal' or 'abnormal')."""
    label = label.strip().lower()
    if label not in LABELS:
        raise HTTPException(400, "label must be 'normal' or 'abnormal'")
    results = []
    for f in files:
        ok, msg = _save_recording(await _read_limited(f, 30), f.filename or "file.wav", label)
        results.append({"file": f.filename, "saved": ok, "message": msg})
    auto = _maybe_auto_retrain()
    return {"saved": sum(r["saved"] for r in results), "failed": sum(not r["saved"] for r in results),
            "results": results, "auto_retrain_started": bool(auto), "pending": len(db.pending_uploads())}


@app.post("/upload_zip")
async def upload_zip(file: UploadFile = File(...)):
    """Bulk upload: a .zip with two folders, 'normal/' and 'abnormal/', each holding .wav files."""
    data = await _read_limited(file, MAX_UPLOAD_MB)
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.write(data)
    tmp.close()
    results = []
    try:
        with zipfile.ZipFile(tmp.name) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            if len(members) > 2000 or sum(m.file_size for m in members) > 600 * 1024 * 1024:
                raise HTTPException(413, "Zip is too large (max 2000 files / 600 MB unpacked).")
            for m in members:
                parts = [p.lower() for p in Path(m.filename).parts[:-1]]
                label = next((p for p in reversed(parts) if p in LABELS), None)
                name = Path(m.filename).name
                if name.startswith(".") or "__MACOSX" in m.filename:
                    continue
                if label is None:
                    results.append({"file": m.filename, "saved": False,
                                    "message": f"{m.filename}: not inside a 'normal' or 'abnormal' folder, skipped"})
                    continue
                if m.file_size > 30 * 1024 * 1024:
                    results.append({"file": m.filename, "saved": False, "message": f"{m.filename}: larger than 30 MB, skipped"})
                    continue
                ok, msg = _save_recording(z.read(m), name, label)
                results.append({"file": m.filename, "saved": ok, "message": msg})
    except zipfile.BadZipFile:
        raise HTTPException(400, "That is not a valid .zip file.")
    finally:
        os.unlink(tmp.name)
    auto = _maybe_auto_retrain()
    return {"saved": sum(r["saved"] for r in results), "failed": sum(not r["saved"] for r in results),
            "results": results[:50], "auto_retrain_started": bool(auto), "pending": len(db.pending_uploads())}


@app.get("/uploads")
def uploads():
    """How many labelled recordings are stored, and when automatic retraining will start."""
    counts = db.upload_counts()
    pending = sum(counts["pending"].values())
    return {"counts": counts, "pending": pending, "auto_retrain_threshold": RETRAIN_THRESHOLD,
            "recordings_until_auto_retrain": max(RETRAIN_THRESHOLD - pending, 0)}


@app.post("/retrain")
def retrain_now():
    """Manual retraining trigger (the 'Retrain' button in the UI)."""
    if not db.pending_uploads():
        raise HTTPException(400, "No new recordings to train on. Upload labelled recordings first.")
    if not MODEL_PATH.exists():
        raise HTTPException(503, "No base model found in models/.")
    started = retrain.start_retrain("manual")
    return {"started": started, "message": "Retraining started." if started else "Retraining is already running."}


@app.get("/retrain/status")
def retrain_status():
    s = dict(retrain.STATE)
    return s


@app.get("/versions")
def versions():
    return db.list_versions()
