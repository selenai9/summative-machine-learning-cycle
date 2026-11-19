"""Retraining: new uploads -> database -> preprocessing -> fine-tune the saved model -> keep it only if it is not worse.

Steps (each one is reported in STATE["message"] so the UI can show progress):
 1. read the pending uploads from the database
 2. preprocess each recording into 5 s spectrogram clips (src/preprocessing.py)
 3. load the saved model and a fixed validation set + a replay sample of old training data (models/replay_set.npz)
 4. fine-tune a COPY of the model on new clips + replay clips (src/model.py: fine_tune)
 5. compare old and new model on the same validation set (recording-level AUC and recall)
 6. if the new model is not worse (within a small margin): archive the old model, save the new one,
    update the threshold and metrics, move the uploads into data/train/.
    Otherwise keep the old model and leave the uploads pending for the next run.
"""
import shutil
import threading
import time
import traceback
from datetime import datetime, timezone

import numpy as np

from . import db
from .config import (ARCHIVE_DIR, METRICS_PATH, MAX_CLIPS_TRAIN, MIN_PROMOTE_MARGIN, MIN_RECALL,
                     MODEL_PATH, REPLAY_PATH, THRESHOLD_PATH, TRAIN_DIR)
from src.model import (aggregate_by_recording, choose_threshold, evaluate_binary, fine_tune,
                       load_json, load_model, predict_clips, save_json, save_model)
from src.preprocessing import recording_to_array

_lock = threading.Lock()
STATE = {"running": False, "message": "idle", "trigger": None, "started_at": None,
         "finished_at": None, "result": None, "error": None}


def _set(**kw):
    STATE.update(kw)


def _rec_eval(model, X_val, y_val, val_rec, threshold=None):
    probs = predict_clips(model, X_val)
    _, p, y = aggregate_by_recording(val_rec, probs, y_val)
    return p, y


def retrain_once(trigger):
    _set(message="1/6 reading pending uploads from the database")
    pending = db.pending_uploads()
    if not pending:
        raise ValueError("No new recordings to train on. Upload some labelled recordings first.")
    if not MODEL_PATH.exists() or not REPLAY_PATH.exists():
        raise FileNotFoundError("models/heart_cnn.keras and models/replay_set.npz are required for retraining.")

    _set(message=f"2/6 preprocessing {len(pending)} recordings into spectrogram clips")
    X_parts, y_parts, used_ids = [], [], []
    for row in pending:
        try:
            arr = recording_to_array(row["path"], max_clips=MAX_CLIPS_TRAIN)
        except Exception:
            continue                                   # unreadable file: skip it
        X_parts.append(arr)
        y_parts.append(np.full(len(arr), 1 if row["label"] == "abnormal" else 0, dtype="float32"))
        used_ids.append(row)
    if not X_parts:
        raise ValueError("None of the pending recordings could be read.")
    X_new, y_new = np.concatenate(X_parts), np.concatenate(y_parts)

    _set(message="3/6 loading the saved model, validation set and replay sample")
    npz = np.load(REPLAY_PATH)
    X_replay, y_replay = npz["X_train"].astype("float32"), npz["y_train"].astype("float32")
    X_val, y_val, val_rec = npz["X_val"].astype("float32"), npz["y_val"].astype("float32"), npz["val_rec"]
    old_info = load_json(THRESHOLD_PATH, {}) or {}
    old_thr = float(old_info.get("threshold", 0.5))

    old_model = load_model(MODEL_PATH)
    p_old, y_rec = _rec_eval(old_model, X_val, y_val, val_rec)
    old_m = evaluate_binary(y_rec, p_old, old_thr)

    _set(message=f"4/6 fine-tuning on {len(X_new)} new clips + replay data")
    new_model = load_model(MODEL_PATH, compile_for_training=True, lr=1e-4)   # a fresh copy of the old model
    new_model, _ = fine_tune(new_model, X_new, y_new, X_replay, y_replay, X_val, y_val, verbose=0)

    _set(message="5/6 comparing old and new model on the validation set")
    p_new, _ = _rec_eval(new_model, X_val, y_val, val_rec)
    new_thr = choose_threshold(y_rec, p_new, min_recall=MIN_RECALL)
    new_m = evaluate_binary(y_rec, p_new, new_thr)
    promoted = (new_m["roc_auc"] >= old_m["roc_auc"] - MIN_PROMOTE_MARGIN)

    version_id = db.add_version(
        trigger=trigger, n_new_recordings=len(used_ids), n_new_clips=int(len(X_new)),
        old_auc=old_m["roc_auc"], new_auc=new_m["roc_auc"],
        old_recall=old_m["recall"], new_recall=new_m["recall"], promoted=promoted,
        note="new model kept" if promoted else "new model was worse; old model kept, uploads stay pending")

    if promoted:
        _set(message="6/6 saving the new model")
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODEL_PATH, ARCHIVE_DIR / f"heart_cnn_before_v{version_id}.keras")
        save_model(new_model, str(MODEL_PATH))
        save_json({"threshold": new_thr, "min_recall": MIN_RECALL, "chosen_on": "validation",
                   "version": version_id}, str(THRESHOLD_PATH))
        metrics = load_json(METRICS_PATH, {}) or {}
        metrics.setdefault("retraining_history", []).append(
            {"version": version_id, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "validation_after_retraining": new_m, "validation_before_retraining": old_m})
        metrics["current_version"] = version_id
        save_json(metrics, str(METRICS_PATH))
        for row in used_ids:                            # keep the data: move it into data/train/<label>/
            dest_dir = TRAIN_DIR / row["label"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / row["path"].replace("\\", "/").split("/")[-1]
            try:
                shutil.move(row["path"], dest)
            except Exception:
                dest = row["path"]
            db.mark_used(row["id"], dest, version_id)
    return {"version": version_id, "promoted": bool(promoted), "old": old_m, "new": new_m,
            "n_new_recordings": len(used_ids), "n_new_clips": int(len(X_new))}


def _worker(trigger):
    try:
        result = retrain_once(trigger)
        _set(result=result, error=None,
             message="done: new model saved" if result["promoted"] else "done: new model was worse, old model kept")
    except Exception as exc:
        traceback.print_exc()
        _set(result=None, error=str(exc), message="failed")
    finally:
        _set(running=False, finished_at=time.time())


def start_retrain(trigger="manual"):
    """Start retraining in a background thread. Returns False if a run is already in progress."""
    with _lock:
        if STATE["running"]:
            return False
        _set(running=True, trigger=trigger, started_at=time.time(), finished_at=None,
             result=None, error=None, message="starting")
    threading.Thread(target=_worker, args=(trigger,), daemon=True).start()
    return True
