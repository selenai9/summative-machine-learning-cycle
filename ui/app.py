"""Streamlit UI: model status/uptime, data insights, single prediction, bulk upload + retraining.

Run:  streamlit run ui/app.py        (set API_URL if the API is not on http://localhost:8000)
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing import extract_features, load_audio, middle_clip, to_melspec, SR    # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
FEATURES_CSV = ROOT / "data" / "features.csv"
DISCLAIMER = "Educational demo, not a medical device. A normal result does not rule out heart disease."

st.set_page_config(page_title="Heart Sound Classifier", page_icon="🫀", layout="wide")


FEATURE_TEXT = {
    "low_share": ("Low-frequency share (20-150 Hz)", "energy in the low band, where the main heart sounds (S1 and S2) sit"),
    "mid_share": ("Mid-frequency share (150-400 Hz)", "energy in the middle band"),
    "high_share": ("High-frequency share (400-1000 Hz)", "energy in the high band, where noise and some murmurs add energy"),
    "spectral_centroid_hz": ("Spectral centroid (Hz)", "the 'centre of mass' of the spectrum: higher means a brighter, higher-pitched sound"),
    "spectral_flatness": ("Spectral flatness", "0 = a few clear tones, 1 = noise-like, spread evenly over frequencies"),
}


def feature_panel(audio):
    """Describe one recording with 5 interpretable numbers and compare them with the training medians.

    This is a description of the sound, NOT a diagnosis: the numbers are weak on their own and the
    recording setup can change them as much as the heart does.
    """
    feats = extract_features(audio)
    ref = None
    if FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV)
        df = df[df["split"] == "train"]
        if df["abnormal"].nunique() == 2:
            ref = df.groupby("abnormal")[list(FEATURE_TEXT)].median()
    rows = []
    for key, (name, _) in FEATURE_TEXT.items():
        row = {"feature": name, "this recording": round(feats[key], 3)}
        if ref is not None:
            n, a = float(ref.loc[0, key]), float(ref.loc[1, key])
            row["typical normal"] = round(n, 3)
            row["typical abnormal"] = round(a, 3)
            row["closer to"] = "normal" if abs(feats[key] - n) <= abs(feats[key] - a) else "abnormal"
        rows.append(row)
    return feats, pd.DataFrame(rows)


# ------------------------------------------------------------------ API helpers
def api_get(path, timeout=15):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


def api_post(path, **kw):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=kw.pop("timeout", 120), **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            return {"_error": f"{r.status_code}: {detail}"}
        return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


def human(seconds):
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{d}d " if d else "") + f"{h}h {m}m {s}s"


# --------------------------------------------------------------------- sidebar
st.sidebar.title("🫀 Heart Sound Classifier")
page = st.sidebar.radio("Page", ["Status", "Data insights", "Predict", "Retrain"])
st.sidebar.caption(f"API: {API_URL}")
st.sidebar.info(DISCLAIMER)


# ------------------------------------------------------------------- Status
if page == "Status":
    st.header("Model status and uptime")
    health = api_get("/health")
    if "_error" in health:
        st.error(f"The API is not reachable: {health['_error']}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("API status", health["status"].upper())
        c2.metric("Uptime", human(health["uptime_seconds"]))
        c3.metric("Model version", health["model_version"] or "initial")
        c4.metric("Predictions served", health["predictions_served"])
        c5, c6, c7 = st.columns(3)
        c5.metric("Model loaded", "yes" if health["model_loaded"] else "no")
        c6.metric("Average prediction time", f"{health['avg_prediction_ms']} ms" if health["avg_prediction_ms"] else "n/a")
        c7.metric("Retraining running", "yes" if health["retraining_running"] else "no")
        st.caption(f"Started at {health['started_at']} (UTC). The page does not refresh by itself; reload to update.")

        info = api_get("/model")
        if "_error" not in info and info.get("metrics"):
            m = info["metrics"]
            st.subheader("Saved evaluation of the current model")
            thr = info.get("threshold", {}).get("threshold")
            st.write(f"Decision threshold (chosen on validation data): **{thr:.3f}**" if thr else "")
            rows = {}
            for split in ("validation", "test"):
                if split in m:
                    rows[split] = {k: m[split].get(k) for k in
                                   ["roc_auc", "recall", "specificity", "precision", "f1", "accuracy"]}
            if rows:
                st.dataframe(pd.DataFrame(rows).T.round(3))
            st.caption("Recall = share of abnormal recordings that were caught. Specificity = share of normal recordings "
                       "correctly called normal. Test numbers were computed once, at the end of the notebook.")


# ------------------------------------------------------------- Data insights
elif page == "Data insights":
    st.header("Data insights (training recordings only)")
    if not FEATURES_CSV.exists():
        st.warning("data/features.csv is missing. Run the notebook (Step 10) and copy the file into data/.")
        st.stop()

    feats = pd.read_csv(FEATURES_CSV)
    train = feats[feats["split"] == "train"].copy()
    train["class"] = np.where(train["abnormal"] == 1, "abnormal", "normal")
    st.write(f"{len(train)} training recordings. The validation and test recordings are not used on this page.")

    # 1. class balance per source
    st.subheader("1. Abnormal rate depends on where the recording came from")
    by_src = train.groupby("source").agg(recordings=("abnormal", "size"), abnormal_rate=("abnormal", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(by_src["source"], by_src["abnormal_rate"] * 100, color="#4c78a8")
    for i, r in by_src.iterrows():
        ax.text(i, r["abnormal_rate"] * 100 + 1, f"n={r['recordings']}", ha="center", fontsize=8)
    ax.set_ylabel("% abnormal"); ax.set_xlabel("source (a to f)"); ax.set_ylim(0, 105)
    st.pyplot(fig)
    hi, lo = by_src.loc[by_src["abnormal_rate"].idxmax()], by_src.loc[by_src["abnormal_rate"].idxmin()]
    st.info(f"Source **{hi['source']}** is {hi['abnormal_rate']*100:.0f}% abnormal but source **{lo['source']}** only "
            f"{lo['abnormal_rate']*100:.0f}%. A model could cheat by recognising the recording setup instead of the heart. "
            "That is why we split by source and class, and why results are also reported per source.")

    # 2. duration
    st.subheader("2. Recording length")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.boxplot([train[train["class"] == c]["duration"] for c in ["normal", "abnormal"]])
    ax.set_xticklabels(["normal", "abnormal"])
    ax.set_ylabel("duration (seconds)")
    st.pyplot(fig)
    med = train.groupby("class")["duration"].median()
    st.info(f"Median length: normal {med['normal']:.1f} s vs abnormal {med['abnormal']:.1f} s. Lengths differ between classes "
            "mostly because of the source mix, so length could act as a shortcut. The model therefore always sees fixed 5 s clips.")

    # 3. frequency features
    st.subheader("3. Frequency features: what is different between normal and abnormal?")
    choices = ["all sources"] + sorted(train["source"].unique())
    src = st.selectbox("Compare within source", choices, help="Comparing inside one source avoids source differences.")
    sub = train if src == "all sources" else train[train["source"] == src]
    labels = {"low_share": "Low-frequency share (20-150 Hz)", "high_share": "High-frequency share (400-1000 Hz)",
              "spectral_centroid_hz": "Spectral centroid (Hz)"}
    feature_names = ["low_share", "mid_share", "high_share", "spectral_centroid_hz", "spectral_flatness"]
    from sklearn.metrics import roc_auc_score

    cols = st.columns(3)
    for col, f in zip(cols, ["low_share", "high_share", "spectral_centroid_hz"]):
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.boxplot([sub[sub["class"] == c][f] for c in ["normal", "abnormal"]], showfliers=False)
        ax.set_xticklabels(["normal", "abnormal"])
        ax.set_title(labels[f], fontsize=9)
        col.pyplot(fig)
        if sub["abnormal"].nunique() == 2:
            mn = sub[sub["class"] == "normal"][f].median(); ma = sub[sub["class"] == "abnormal"][f].median()
            auc = roc_auc_score(sub["abnormal"], sub[f])
            direction = "higher" if ma > mn else "lower"
            strength = "barely separates" if abs(auc - 0.5) < 0.05 else ("weakly separates" if abs(auc - 0.5) < 0.15 else "clearly separates")
            col.caption(f"Median {mn:.3f} (normal) vs {ma:.3f} (abnormal): abnormal is **{direction}**. "
                        f"AUC alone = {auc:.2f}, so this feature {strength} the classes.")

    if sub["abnormal"].nunique() == 2:
        auc_tab = pd.DataFrame({"feature": feature_names,
                                "AUC vs target": [roc_auc_score(sub["abnormal"], sub[f]) for f in feature_names]})
        auc_tab["strength (distance from 0.5)"] = (auc_tab["AUC vs target"] - 0.5).abs().round(3)
        st.write("**Each feature on its own against the label** (0.5 = no information, 1.0 or 0.0 = perfect):")
        st.dataframe(auc_tab.round(3), hide_index=True)
    st.warning("Differences between classes can come from noise or the recording device, not only from heart murmurs. "
               "No single feature is enough; the CNN looks at the whole spectrogram and is tested on unseen recordings.")


# ------------------------------------------------------------------- Predict
elif page == "Predict":
    st.header("Classify one heart-sound recording")
    st.caption("Upload a stethoscope recording (.wav). Best results: 5 seconds or longer, recorded on the chest, quiet room.")
    up = st.file_uploader("Heart-sound recording", type=["wav", "flac", "ogg"])
    if up is not None:
        with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
            tmp.write(up.getvalue())
            tmp_path = tmp.name
        try:
            y = load_audio(tmp_path)
            clip = middle_clip(y)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3))
            axes[0].plot(np.arange(len(clip)) / SR, clip, linewidth=0.6); axes[0].set_title("Waveform (middle 5 s)")
            axes[0].set_xlabel("seconds")
            axes[1].imshow(to_melspec(clip), origin="lower", aspect="auto", cmap="magma")
            axes[1].set_title("Spectrogram (what the CNN sees)"); axes[1].set_xlabel("time"); axes[1].set_ylabel("frequency band")
            st.pyplot(fig)
            feats, table = feature_panel(y)
            with st.expander("Sound description: which frequency bands are strong in this recording?", expanded=True):
                parts = {"low (20-150 Hz)": feats["low_share"], "mid (150-400 Hz)": feats["mid_share"],
                         "high (400-1000 Hz)": feats["high_share"]}
                top = max(parts, key=parts.get)
                st.write(f"Most of the energy in the middle 5 s is in the **{top}** band "
                         f"({parts[top] * 100:.0f}% of the energy between 20 and 1000 Hz).")
                st.bar_chart(pd.DataFrame({"share of energy": parts}))
                st.dataframe(table, hide_index=True)
                st.caption("Typical values are the medians of the training recordings in each class. "
                           "What each number means: " + "; ".join(f"{v[0]} = {v[1]}" for v in FEATURE_TEXT.values()) + ".")
                st.warning("This describes the sound. It is not a diagnosis and it does not name a murmur or a condition. "
                           "Each of these numbers separates normal from abnormal only weakly on its own (see Data insights), and the "
                           "recording device and noise change them too. The decision comes from the CNN, not from this table.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
        if st.button("Classify", type="primary"):
            with st.spinner("Analysing..."):
                res = api_post("/predict", files={"file": (up.name, up.getvalue())})
            if "_error" in res:
                st.error(res["_error"])
            else:
                label = res["label"]
                (st.error if label == "abnormal" else st.success)(
                    f"Result: **{label.upper()}**  (probability abnormal = {res['probability_abnormal']:.2f}, "
                    f"threshold = {res['threshold']:.2f})")
                st.write(f"{res['n_clips']} clip(s) of 5 s analysed; response time {res['latency_ms']} ms.")
                if res["n_clips"] > 1:
                    st.bar_chart(pd.DataFrame({"probability abnormal": res["clip_probabilities"]}))
                for w in res["warnings"]:
                    st.warning(w)
                st.caption(res["disclaimer"])
        os.unlink(tmp_path)


# ------------------------------------------------------------------- Retrain
elif page == "Retrain":
    st.header("Upload new recordings and retrain")
    up_info = api_get("/uploads")
    if "_error" in up_info:
        st.error(f"The API is not reachable: {up_info['_error']}")
        st.stop()
    c = up_info["counts"]["pending"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Waiting: normal", c.get("normal", 0))
    c2.metric("Waiting: abnormal", c.get("abnormal", 0))
    c3.metric("Until automatic retraining", up_info["recordings_until_auto_retrain"],
              help=f"Retraining starts by itself when {up_info['auto_retrain_threshold']} new recordings are waiting.")

    st.subheader("1. Upload labelled recordings")
    mode = st.radio("How do you want to upload?", ["Many files with one label", "A .zip with normal/ and abnormal/ folders"], horizontal=True)
    if mode.startswith("Many"):
        label = st.selectbox("Label of these recordings", ["normal", "abnormal"])
        files = st.file_uploader("Recordings", type=["wav", "flac", "ogg"], accept_multiple_files=True)
        if files and st.button("Upload recordings"):
            res = api_post("/upload", data={"label": label}, files=[("files", (f.name, f.getvalue())) for f in files])
            st.error(res["_error"]) if "_error" in res else st.success(f"Saved {res['saved']}, failed {res['failed']}. "
                                                                       f"Auto-retrain started: {res['auto_retrain_started']}")
            if "_error" not in res:
                for r in res["results"]:
                    if not r["saved"]:
                        st.warning(r["message"])
    else:
        z = st.file_uploader("Zip file", type=["zip"])
        if z and st.button("Upload zip"):
            res = api_post("/upload_zip", files={"file": (z.name, z.getvalue())}, timeout=600)
            st.error(res["_error"]) if "_error" in res else st.success(f"Saved {res['saved']}, failed {res['failed']}. "
                                                                       f"Auto-retrain started: {res['auto_retrain_started']}")
            if "_error" not in res:
                for r in res["results"]:
                    if not r["saved"]:
                        st.warning(r["message"])

    st.subheader("2. Retrain")
    st.caption("The saved model is fine-tuned on the new recordings plus a sample of the old training data, then compared with the "
               "old model on the same validation set. It is only kept if it is not worse.")
    if st.button("Retrain now", type="primary"):
        res = api_post("/retrain")
        st.error(res["_error"]) if "_error" in res else st.info(res["message"])

    status = api_get("/retrain/status")
    if "_error" not in status:
        st.write(f"**Status:** {status['message']}")
        if status["error"]:
            st.error(status["error"])
        if status["result"]:
            r = status["result"]
            st.success(f"Version {r['version']}: validation AUC {r['old']['roc_auc']:.3f} -> {r['new']['roc_auc']:.3f}, "
                       f"recall {r['old']['recall']:.2f} -> {r['new']['recall']:.2f}. "
                       f"{'New model is now live.' if r['promoted'] else 'New model was worse; the old model was kept.'}")
        if status["running"]:
            time.sleep(3)
            st.rerun()

    st.subheader("Model history")
    vers = api_get("/versions")
    if isinstance(vers, list) and vers:
        df = pd.DataFrame(vers)[["id", "created_at", "trigger", "n_new_recordings", "n_new_clips",
                                 "old_auc", "new_auc", "old_recall", "new_recall", "promoted", "note"]]
        st.dataframe(df.round(3), hide_index=True)
    else:
        st.write("No retraining has happened yet.")