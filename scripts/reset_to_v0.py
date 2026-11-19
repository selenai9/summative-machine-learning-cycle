"""Put the repo back to the original (version 0) model after trying retraining in the app.

Run from the repo root with the API stopped:   python scripts/reset_to_v0.py

It: restores models/heart_cnn.keras from models/archive/heart_cnn_before_v1.keras, rewrites
models/threshold.json from the threshold saved in models/metrics.json, removes the retraining history from
metrics.json, deletes the app database and uploads, and removes the demo recordings the retraining moved into
data/train/. It asks before changing anything.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models"
archive = M / "archive" / "heart_cnn_before_v1.keras"
if not archive.exists():
    sys.exit(f"Not found: {archive}. Nothing to restore (no retraining was promoted, or it was already reset).")

metrics = json.loads((M / "metrics.json").read_text())
thr = metrics.get("threshold")
if thr is None:
    sys.exit("metrics.json has no 'threshold'; rerun Step 18 of the notebook instead.")

print("This will:")
print(f" - copy {archive.name} over models/heart_cnn.keras")
print(f" - set models/threshold.json to threshold {thr:.4f} (version 0)")
print(" - remove retraining history from models/metrics.json")
print(" - delete data/app.db, data/uploads/ and the .wav/.flac/.ogg files in data/train/normal and data/train/abnormal")
if input("Type yes to continue: ").strip().lower() != "yes":
    sys.exit("Cancelled.")

shutil.copy2(archive, M / "heart_cnn.keras")
(M / "threshold.json").write_text(json.dumps(
    {"threshold": float(thr), "min_recall": 0.85, "chosen_on": "validation, recording level", "version": 0}, indent=2))
metrics.pop("retraining_history", None)
metrics["current_version"] = 0
(M / "metrics.json").write_text(json.dumps(metrics, indent=2))

(ROOT / "data" / "app.db").unlink(missing_ok=True)
shutil.rmtree(ROOT / "data" / "uploads", ignore_errors=True)
for name in ("normal", "abnormal"):
    for p in (ROOT / "data" / "train" / name).glob("*"):
        if p.suffix.lower() in {".wav", ".flac", ".ogg"}:
            p.unlink()
shutil.rmtree(M / "archive", ignore_errors=True)
print("Done. Restart the API and check the Status page: model version should say 'initial' and the threshold 0.643.")
