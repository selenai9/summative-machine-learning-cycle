"""Streamlit UI: model status/uptime, data insights, single prediction, bulk upload + retraining.

Run:  streamlit run ui/app.py        (set API_URL if the API is not on http://localhost:8000)

Look: "chart paper" - a warm paper background, navy ink, serif headings, and two meaning colours that are
used for nothing else: teal = normal, vermilion = abnormal.
"""
import html
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
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

# palette (teal / vermilion validated for colour-blind separation; they only ever mean normal / abnormal)
PAPER, PAPER_2 = "#F7F5F0", "#EFECE4"
INK, MUTED, FAINT, RULE = "#14213D", "#5B6575", "#8A94A3", "#D8D3C6"
NORMAL, ABNORMAL = "#08908A", "#D9531E"

st.set_page_config(page_title="Heart Sound Classifier", layout="wide")

matplotlib.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --paper:#F7F5F0; --paper2:#EFECE4; --ink:#14213D; --muted:#5B6575; --faint:#8A94A3; --rule:#D8D3C6;
        --normal:#08908A; --abnormal:#D9531E; }
.stApp { background: var(--paper); color: var(--ink); font-family: 'IBM Plex Sans', system-ui, sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
h1, h2, h3, h4 { font-family: 'Newsreader', Georgia, serif !important; color: var(--ink) !important; letter-spacing: -0.01em; }
p, li, label, span, div { font-family: inherit; }
a { color: var(--ink); }
.mono, .ledger dd, .num { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--paper2); border-right: 1px solid var(--rule); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.6rem; }
.wordmark { font-family:'Newsreader',Georgia,serif; font-size: 1.55rem; line-height: 1.05; font-weight: 600; color: var(--ink); }
.wordmark-sub { font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); margin: .45rem 0 1.4rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .1rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .35rem .2rem; border-bottom: 1px solid var(--rule); width: 100%; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: .95rem; }
.side-note { border-left: 3px solid var(--abnormal); padding: .15rem 0 .15rem .75rem; margin-top: 1.6rem;
             font-size: .8rem; color: var(--muted); line-height: 1.45; }
.side-api { font-size: .72rem; color: var(--faint); margin-top: 1.2rem; word-break: break-all; }

/* page header with ECG-paper grid */
.ph { position: relative; padding: 1.5rem 1.6rem 1.1rem; margin: 0 0 1.8rem; border: 1px solid var(--rule);
      background-color: var(--paper);
      background-image: linear-gradient(rgba(20,33,61,.055) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(20,33,61,.055) 1px, transparent 1px),
                        linear-gradient(rgba(20,33,61,.10) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(20,33,61,.10) 1px, transparent 1px);
      background-size: 10px 10px, 10px 10px, 50px 50px, 50px 50px; }
.kicker { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.ph h1 { font-size: 2.15rem; line-height: 1.12; margin: .35rem 0 .5rem; font-weight: 500; max-width: 24ch; }
.ph .lede { font-size: .98rem; color: var(--muted); max-width: 62ch; margin: .2rem 0 0; line-height: 1.5; }
.ph svg { display:block; width: 100%; height: 30px; margin-top: .8rem; }

/* section titles */
.sec { display:flex; align-items: baseline; gap: .8rem; margin: 2.1rem 0 .7rem; padding-top: .9rem; border-top: 1px solid var(--ink); }
.sec .n { font-family:'IBM Plex Mono',monospace; font-size: .8rem; color: var(--muted); }
.sec h3 { margin: 0; font-size: 1.35rem; font-weight: 500; }
.sub { color: var(--muted); font-size: .88rem; margin: -.2rem 0 .8rem; max-width: 70ch; line-height: 1.5; }

/* ledger (label ..... value) */
.ledger { margin: 0; padding: 0; }
.ledger > div { display:flex; align-items: baseline; gap: .5rem; padding: .5rem 0; border-bottom: 1px solid var(--rule); }
.ledger dt { color: var(--muted); font-size: .9rem; white-space: nowrap; margin: 0; }
.ledger .lead { flex: 1; border-bottom: 1px dotted var(--faint); transform: translateY(-3px); min-width: 12px; }
.ledger dd { margin: 0; font-size: 1.02rem; color: var(--ink); text-align: right; }
.dot { display:inline-block; width: .55rem; height: .55rem; border-radius: 50%; margin-right: .45rem; vertical-align: middle; }

/* verdict */
.verdict { border-top: 3px solid var(--ink); padding: .9rem 0 .4rem; }
.verdict .lab { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.verdict .word { font-family:'Newsreader',Georgia,serif; font-size: 3.4rem; line-height: 1; font-weight: 500; margin: .2rem 0 .1rem; }
.verdict.normal .word { color: var(--normal); }
.verdict.abnormal .word { color: var(--abnormal); }
.verdict .note { color: var(--muted); font-size: .88rem; }
.scale { position: relative; height: 12px; margin: 2rem 0 1.9rem; }
.scale .zone-n, .scale .zone-a { position:absolute; top: 3px; height: 6px; }
.scale .zone-n { left:0; background: rgba(8,144,138,.30); border-radius: 3px 0 0 3px; }
.scale .zone-a { right:0; background: rgba(217,83,30,.30); border-radius: 0 3px 3px 0; }
.scale .tick { position:absolute; top: -6px; width: 2px; height: 24px; background: var(--ink); }
.scale .tick span { position:absolute; top: 26px; left: 50%; transform: translateX(-50%); font-size: .7rem; color: var(--muted); white-space: nowrap; }
.scale .mark { position:absolute; top: -2px; width: 16px; height: 16px; border-radius: 50%; border: 3px solid var(--paper);
               box-shadow: 0 0 0 1.5px var(--ink); transform: translateX(-50%); }
.scale .end { position:absolute; top: -22px; font-size: .7rem; color: var(--muted); }
.clips { display:flex; align-items:flex-end; gap: 4px; height: 52px; border-bottom: 1px solid var(--ink); position: relative; margin-top: .4rem; }
.clips .bar { flex: 1; max-width: 34px; border-radius: 2px 2px 0 0; min-height: 2px; }
.clips .thr { position:absolute; left:0; right:0; border-top: 1px dashed var(--ink); }
.cap { font-size: .78rem; color: var(--muted); margin-top: .35rem; }
.callout { border-left: 3px solid var(--ink); padding: .2rem 0 .2rem .85rem; color: var(--muted); font-size: .9rem; line-height: 1.5; margin: .8rem 0; }
.callout.warn { border-left-color: var(--abnormal); }
.callout b { color: var(--ink); }

/* controls */
.stButton > button { border-radius: 2px; border: 1.5px solid var(--ink); background: var(--ink); color: var(--paper); font-weight: 500;
                     padding: .45rem 1.2rem; }
.stButton > button:hover { background: var(--paper); color: var(--ink); border-color: var(--ink); }
.stButton > button:disabled { background: var(--paper2); color: var(--faint); border-color: var(--rule); }
[data-testid="stFileUploaderDropzone"] { background: transparent; border: 1.5px dashed var(--faint); border-radius: 2px; }
button[data-baseweb="tab"] { font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--ink); }
[data-baseweb="tab-highlight"] { background: var(--ink) !important; height: 2px !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: 2px; }
div[data-testid="stAlert"] { border-radius: 2px; }
hr { border-color: var(--rule); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

TRACE = ('<svg viewBox="0 0 600 30" preserveAspectRatio="none" aria-hidden="true"><path d="M0 20 H110 l6 -3 l8 3 H168 l5 4 '
         'l7 -22 l7 26 l5 -8 H236 l9 -7 l11 7 H600" fill="none" stroke="#14213D" stroke-width="1.6" '
         'vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>')


def esc(x):
    return html.escape(str(x))


def page_header(kicker, title, lede=""):
    st.markdown(f'<div class="ph"><div class="kicker">{esc(kicker)}</div><h1>{esc(title)}</h1>'
                f'{f"<p class=lede>{esc(lede)}</p>" if lede else ""}{TRACE}</div>', unsafe_allow_html=True)


def section(num, title, sub=""):
    st.markdown(f'<div class="sec"><span class="n">{esc(num)}</span><h3>{esc(title)}</h3></div>'
                + (f'<div class="sub">{esc(sub)}</div>' if sub else ""), unsafe_allow_html=True)


def callout(text_html, warn=False):
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{text_html}</div>', unsafe_allow_html=True)


def ledger(rows):
    """rows: list of (label, value_html)."""
    body = "".join(f'<div><dt>{esc(k)}</dt><span class="lead"></span><dd>{v}</dd></div>' for k, v in rows)
    st.markdown(f'<dl class="ledger">{body}</dl>', unsafe_allow_html=True)


def show_fig(fig):
    st.pyplot(fig, clear_figure=True)


def style_ax(ax, grid_axis=None):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=RULE, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


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
        row = {"feature": name, "this one": round(feats[key], 3)}
        if ref is not None:
            n, a = float(ref.loc[0, key]), float(ref.loc[1, key])
            row["normal"] = round(n, 3)
            row["abnormal"] = round(a, 3)
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
st.sidebar.markdown('<div class="wordmark">Heart Sound<br>Classifier</div>'
                    '<div class="wordmark-sub">Phonocardiogram screening prototype</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Page", ["Status", "Data insights", "Predict", "Retrain"], label_visibility="collapsed")
st.sidebar.markdown(f'<div class="side-note">{esc(DISCLAIMER)}</div>'
                    f'<div class="side-api">API: {esc(API_URL)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------- Status
if page == "Status":
    page_header("Model status", "Is the service up, and how good is the model?",
                "Live state of the API, then the evaluation saved when the model was trained.")
    health = api_get("/health")
    if "_error" in health:
        callout(f"<b>The API is not reachable.</b> {esc(health['_error'])}", warn=True)
    else:
        left, right = st.columns([5, 7], gap="large")
        with left:
            section("01", "Service")
            ledger([
                ("API status", f'<span class="dot" style="background:{NORMAL}"></span>{esc(health["status"].upper())}'),
                ("Uptime", esc(human(health["uptime_seconds"]))),
                ("Model version", esc(health["model_version"] or "initial")),
                ("Model loaded", "yes" if health["model_loaded"] else "no"),
                ("Predictions served", esc(health["predictions_served"])),
                ("Average prediction time", f'{health["avg_prediction_ms"]} ms' if health["avg_prediction_ms"] else "n/a"),
                ("Retraining running", "yes" if health["retraining_running"] else "no"),
            ])
            st.markdown(f'<div class="cap">Started {esc(health["started_at"])} (UTC). The page does not refresh by itself; '
                        'reload to update.</div>', unsafe_allow_html=True)

        info = api_get("/model")
        m = info.get("metrics") if "_error" not in info else None
        with right:
            section("02", "Saved evaluation of the current model")
            if not m:
                st.write("No saved evaluation found.")
            else:
                thr = (info.get("threshold") or {}).get("threshold")
                if thr:
                    st.markdown(f'<div class="sub">Decision threshold, chosen on the validation data: '
                                f'<span class="num"><b>{thr:.3f}</b></span>. A recording is called abnormal when its average '
                                f'probability reaches this value.</div>', unsafe_allow_html=True)
                keys = ["roc_auc", "recall", "specificity", "precision", "f1", "accuracy"]
                names = ["ROC-AUC", "Recall", "Specificity", "Precision", "F1", "Accuracy"]
                val = m.get("validation") or {}
                test = m.get("test") or {}
                fig, ax = plt.subplots(figsize=(6.4, 3.0))
                ypos = np.arange(len(keys))[::-1]
                for y, k in zip(ypos, keys):
                    v, t = val.get(k), test.get(k)
                    if v is not None and t is not None:
                        ax.plot([v, t], [y, y], color=RULE, linewidth=2.2, zorder=1)
                    if v is not None:
                        ax.scatter(v, y, s=52, facecolor=PAPER, edgecolor=FAINT, linewidth=1.8, zorder=2)
                    if t is not None:
                        ax.scatter(t, y, s=52, color=INK, zorder=3)
                        ax.text(max(t, v or 0) + 0.018, y, f"{t:.3f}", va="center", fontsize=8.5, color=INK, family="monospace")
                ax.set_yticks(ypos); ax.set_yticklabels(names)
                ax.set_xlim(0.4, 1.0); ax.set_xlabel("score")
                ax.scatter([], [], s=52, facecolor=PAPER, edgecolor=FAINT, linewidth=1.8, label="validation")
                ax.scatter([], [], s=52, color=INK, label="test (computed once)")
                ax.legend(frameon=False, loc="lower left", fontsize=8, ncols=2, bbox_to_anchor=(0, -0.38))
                style_ax(ax, "x")
                fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.30)
                show_fig(fig)
                st.markdown('<div class="cap">Recall = share of abnormal recordings caught. Specificity = share of normal '
                            'recordings correctly called normal. Test numbers were computed once, at the end of the notebook.'
                            '</div>', unsafe_allow_html=True)

        if m and m.get("test_per_source"):
            section("03", "The same test set, split by recording source",
                    "One overall score can hide big differences. Hollow markers mark sources with fewer than 30 test "
                    "recordings, which are too few to judge. A score of 0.5 is chance.")
            ps = pd.DataFrame(m["test_per_source"]).sort_values("recordings", ascending=False).reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(8.4, 0.5 + 0.46 * len(ps)))
            ypos = np.arange(len(ps))[::-1]
            ax.axvline(0.5, color=FAINT, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            for y, (_, r) in zip(ypos, ps.iterrows()):
                auc = r.get("roc_auc")
                big = r["recordings"] >= 30
                if auc is not None and not pd.isna(auc):
                    ax.scatter(auc, y, s=70, facecolor=INK if big else PAPER, edgecolor=INK, linewidth=1.8, zorder=3)
                    if auc > 0.9:
                        ax.text(auc - 0.03, y, f"{auc:.2f}", va="center", ha="right", fontsize=8.5, family="monospace")
                    else:
                        ax.text(auc + 0.025, y, f"{auc:.2f}", va="center", fontsize=8.5, family="monospace")
                rec = r.get("recall"); spec = r.get("specificity")
                fmt = lambda x: "n/a" if x is None or pd.isna(x) else f"{x:.2f}"
                ax.text(1.06, y, f"n={int(r['recordings'])}   recall {fmt(rec)}   spec. {fmt(spec)}", va="center",
                        fontsize=8.2, color=MUTED, transform=ax.get_yaxis_transform(), clip_on=False)
            ax.set_yticks(ypos); ax.set_yticklabels([f"source {s}" for s in ps["source"]])
            ax.set_xlim(0, 1.04); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel("ROC-AUC on the test recordings of that source")
            ax.text(0.5, ypos[0] + 0.62, "chance", ha="center", fontsize=7.5, color=MUTED)
            style_ax(ax, "x")
            fig.subplots_adjust(left=0.1, right=0.62, top=0.93, bottom=0.2)
            show_fig(fig)
            callout("<b>How to read this.</b> The big source (e) is scored almost perfectly and carries the overall result. "
                    "In sources a and b the model is close to or below chance, and the abnormal rate differs a lot between "
                    "sources, so part of the overall score may come from recognising the recording setup rather than the heart. "
                    "That is why this prototype should not be trusted on recordings from a new device without re-checking it.",
                    warn=True)
            q = m.get("test_by_quality")
            if q:
                qdf = pd.DataFrame(q).rename(columns={"group": "recordings by quality", "roc_auc": "ROC-AUC"})
                st.dataframe(qdf.round(3), hide_index=True)


# ------------------------------------------------------------- Data insights
elif page == "Data insights":
    page_header("Data insights", "What the training recordings look like",
                "Three things worth knowing before trusting any model on this data. Only the training recordings are used here; "
                "the validation and test recordings stay untouched.")
    if not FEATURES_CSV.exists():
        callout("<b>data/features.csv is missing.</b> Run the notebook (Step 10) and copy the file into data/.", warn=True)
        st.stop()

    feats = pd.read_csv(FEATURES_CSV)
    train = feats[feats["split"] == "train"].copy()
    train["class"] = np.where(train["abnormal"] == 1, "abnormal", "normal")
    st.markdown(f'<div class="cap" style="margin-top:-.8rem">{len(train):,} training recordings.</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Where recordings come from", "Recording length", "Frequency features"])

    with t1:
        section("01", "The abnormal rate depends on where the recording came from")
        by_src = (train.groupby("source").agg(recordings=("abnormal", "size"), abnormal_rate=("abnormal", "mean"))
                  .reset_index().sort_values("source", ascending=False).reset_index(drop=True))
        overall = train["abnormal"].mean() * 100
        fig, ax = plt.subplots(figsize=(8.4, 0.55 + 0.42 * len(by_src)))
        ypos = np.arange(len(by_src))
        ax.barh(ypos, by_src["abnormal_rate"] * 100, color=ABNORMAL, height=0.58)
        ax.axvline(overall, color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
        for y, (_, r) in zip(ypos, by_src.iterrows()):
            ax.text(84, y, f"{r['abnormal_rate'] * 100:.0f}%  ·  n={int(r['recordings'])}", va="center", fontsize=8.5, color=INK)
        ax.text(overall + 0.8, len(by_src) - 0.35, f"all sources {overall:.0f}%", fontsize=7.8, color=INK)
        ax.set_yticks(ypos); ax.set_yticklabels([f"source {s}" for s in by_src["source"]])
        ax.set_xlim(0, 112); ax.set_xticks([0, 20, 40, 60, 80]); ax.set_xlabel("% of recordings that are abnormal")
        style_ax(ax)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.2)
        show_fig(fig)
        hi, lo = by_src.loc[by_src["abnormal_rate"].idxmax()], by_src.loc[by_src["abnormal_rate"].idxmin()]
        callout(f"Source <b>{esc(hi['source'])}</b> is {hi['abnormal_rate'] * 100:.0f}% abnormal but source <b>{esc(lo['source'])}</b> "
                f"only {lo['abnormal_rate'] * 100:.0f}%. A model could cheat by recognising the recording setup instead of the heart. "
                "That is why the data is split by source and class, and why results are also reported per source.")

    with t2:
        section("02", "Recording length")
        fig, ax = plt.subplots(figsize=(8.4, 2.4))
        data = [train[train["class"] == c]["duration"] for c in ["abnormal", "normal"]]
        bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=FAINT), capprops=dict(color=FAINT))
        for patch, col in zip(bp["boxes"], [ABNORMAL, NORMAL]):
            patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
        med = train.groupby("class")["duration"].median()
        for i, c in enumerate(["abnormal", "normal"], start=1):
            ax.text(med[c] + 0.6, i + 0.34, f"median {med[c]:.1f} s", fontsize=8, color=INK)
        ax.set_yticks([1, 2]); ax.set_yticklabels(["abnormal", "normal"])
        ax.set_xlabel("duration (seconds)")
        style_ax(ax, "x")
        fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.25)
        show_fig(fig)
        callout(f"Median length: normal {med['normal']:.1f} s vs abnormal {med['abnormal']:.1f} s. Lengths differ between classes "
                "mostly because of the source mix, so length could act as a shortcut. The model therefore always sees fixed 5 s clips.")

    with t3:
        section("03", "Frequency features: what is different between normal and abnormal?")
        from sklearn.metrics import roc_auc_score
        choices = ["all sources"] + sorted(train["source"].unique())
        src = st.selectbox("Compare within source", choices, help="Comparing inside one source avoids source differences.")
        sub = train if src == "all sources" else train[train["source"] == src]
        shown = [("low_share", "Low-frequency share", "20-150 Hz"), ("high_share", "High-frequency share", "400-1000 Hz"),
                 ("spectral_centroid_hz", "Spectral centroid", "Hz")]
        feature_names = ["low_share", "mid_share", "high_share", "spectral_centroid_hz", "spectral_flatness"]
        for f, title, unit in shown:
            c_chart, c_text = st.columns([6, 5], gap="large")
            fig, ax = plt.subplots(figsize=(5.2, 1.9))
            groups = [sub[sub["class"] == c][f] for c in ["abnormal", "normal"]]
            bp = ax.boxplot(groups, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=FAINT), capprops=dict(color=FAINT))
            for patch, col in zip(bp["boxes"], [ABNORMAL, NORMAL]):
                patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
            ax.set_yticks([1, 2]); ax.set_yticklabels(["abnormal", "normal"])
            ax.set_xlabel(f"{title} ({unit})")
            style_ax(ax, "x")
            fig.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.3)
            with c_chart:
                show_fig(fig)
            with c_text:
                st.markdown(f"<h4 style='margin:.2rem 0 .3rem'>{esc(title)}</h4>", unsafe_allow_html=True)
                if sub["abnormal"].nunique() == 2:
                    mn = sub[sub["class"] == "normal"][f].median(); ma = sub[sub["class"] == "abnormal"][f].median()
                    auc = roc_auc_score(sub["abnormal"], sub[f])
                    direction = "higher" if ma > mn else "lower"
                    strength = ("barely separates" if abs(auc - 0.5) < 0.05
                                else ("weakly separates" if abs(auc - 0.5) < 0.15 else "clearly separates"))
                    st.markdown(f'<div class="sub">Median <span class="num">{mn:.3f}</span> (normal) vs '
                                f'<span class="num">{ma:.3f}</span> (abnormal): abnormal is <b>{direction}</b>. On its own this '
                                f'feature has AUC <span class="num">{auc:.2f}</span>, so it {strength} the classes.</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sub">This source has only one class, so there is nothing to compare.</div>',
                                unsafe_allow_html=True)

        if sub["abnormal"].nunique() == 2:
            st.markdown("<h4 style='margin-top:1.4rem'>Each feature on its own against the label</h4>"
                        "<div class='sub'>0.5 means no information; 1.0 or 0.0 means a perfect (or perfectly reversed) split.</div>",
                        unsafe_allow_html=True)
            auc_tab = pd.DataFrame({"feature": feature_names,
                                    "AUC vs target": [roc_auc_score(sub["abnormal"], sub[f]) for f in feature_names]})
            auc_tab["strength (distance from 0.5)"] = (auc_tab["AUC vs target"] - 0.5).abs().round(3)
            st.dataframe(auc_tab.round(3), hide_index=True)
        callout("Differences between classes can come from noise or the recording device, not only from heart murmurs. "
                "No single feature is enough; the CNN looks at the whole spectrogram and is tested on unseen recordings.", warn=True)


# ------------------------------------------------------------------- Predict
elif page == "Predict":
    page_header("Predict", "Classify one heart-sound recording",
                "Upload a stethoscope recording. Best results: 5 seconds or longer, recorded on the chest, in a quiet room.")
    left, right = st.columns([4, 8], gap="large")
    with left:
        up = st.file_uploader("Heart-sound recording (.wav, .flac, .ogg)", type=["wav", "flac", "ogg"])
        go = st.button("Classify", disabled=up is None)
        if up is None:
            st.markdown('<div class="cap">Sample recordings are in <span class="mono">data/test/normal</span> and '
                        '<span class="mono">data/test/abnormal</span>.</div>', unsafe_allow_html=True)

    with right:
        if up is None:
            callout("Choose a file on the left to see its waveform, its spectrogram, a description of its sound, and the model's result.")
        else:
            key = f"{up.name}:{up.size}"
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
                tmp.write(up.getvalue())
                tmp_path = tmp.name
            result_box = st.container()
            try:
                y = load_audio(tmp_path)
                clip = middle_clip(y)
                feats, table = feature_panel(y)
                ok = True
            except Exception as exc:
                ok = False
                callout(f"<b>Could not read this file.</b> {esc(exc)}", warn=True)

            if go:
                with st.spinner("Analysing..."):
                    res = api_post("/predict", files={"file": (up.name, up.getvalue())})
                st.session_state["last_result"] = (key, res)

            last = st.session_state.get("last_result")
            with result_box:
                if last and last[0] == key:
                    res = last[1]
                    if "_error" in res:
                        callout(f"<b>The API returned an error.</b> {esc(res['_error'])}", warn=True)
                    else:
                        label = res["label"]
                        p, thr = float(res["probability_abnormal"]), float(res["threshold"])
                        st.markdown(f"""
<div class="verdict {label}">
  <div class="lab">Result for {esc(up.name)}</div>
  <div class="word">{esc(label.capitalize())}</div>
  <div class="note"><span class="num">{p:.2f}</span> probability of abnormal  ·  threshold <span class="num">{thr:.2f}</span>
   ·  {int(res['n_clips'])} clip(s) of 5 s  ·  {res['latency_ms']} ms</div>
  <div class="scale">
    <div class="zone-n" style="width:{thr * 100:.1f}%"></div><div class="zone-a" style="width:{(1 - thr) * 100:.1f}%"></div>
    <span class="end" style="left:0">0 normal</span><span class="end" style="right:0">1 abnormal</span>
    <div class="tick" style="left:{thr * 100:.1f}%"><span>threshold {thr:.2f}</span></div>
    <div class="mark" style="left:{p * 100:.1f}%; background:{ABNORMAL if label == 'abnormal' else NORMAL}"></div>
  </div>
</div>""", unsafe_allow_html=True)
                        if res["n_clips"] > 1:
                            bars = "".join(
                                f'<div class="bar" title="clip {i + 1}: {cp:.2f}" style="height:{max(cp, 0.02) * 100:.0f}%;'
                                f'background:{ABNORMAL if cp >= thr else NORMAL}"></div>'
                                for i, cp in enumerate(res["clip_probabilities"]))
                            st.markdown(f'<div class="clips">{bars}<div class="thr" style="bottom:{thr * 100:.0f}%"></div></div>'
                                        '<div class="cap">One bar per 5 s clip (height = probability of abnormal). The result is the '
                                        'average of these. Dashed line = threshold.</div>', unsafe_allow_html=True)
                        for w in res["warnings"]:
                            callout(f"<b>Quality warning.</b> {esc(w)}", warn=True)
                        st.markdown(f'<div class="cap" style="margin-top:.9rem">{esc(res["disclaimer"])}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="cap">Press <b>Classify</b> to get the model\'s result for this file. '
                                'The signal and the sound description below are available straight away.</div>',
                                unsafe_allow_html=True)

            if ok:
                tab_sig, tab_desc = st.tabs(["Signal", "Sound description"])
                with tab_sig:
                    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.7), gridspec_kw={"width_ratios": [1, 1]})
                    axes[0].plot(np.arange(len(clip)) / SR, clip, linewidth=0.7, color=INK)
                    axes[0].set_title("Waveform (middle 5 s)", fontsize=9.5, loc="left", color=INK)
                    axes[0].set_xlabel("seconds"); axes[0].set_yticks([])
                    axes[0].spines["left"].set_visible(False)
                    axes[1].imshow(to_melspec(clip), origin="lower", aspect="auto", cmap="cividis")
                    axes[1].set_title("Spectrogram (what the CNN sees)", fontsize=9.5, loc="left", color=INK)
                    axes[1].set_xlabel("time"); axes[1].set_ylabel("frequency band")
                    for a in axes:
                        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
                    fig.tight_layout()
                    show_fig(fig)
                with tab_desc:
                    parts = {"low (20-150 Hz)": feats["low_share"], "mid (150-400 Hz)": feats["mid_share"],
                             "high (400-1000 Hz)": feats["high_share"]}
                    top = max(parts, key=parts.get)
                    st.markdown(f'<div class="sub">Most of the energy in the middle 5 s is in the <b>{esc(top)}</b> band '
                                f'({parts[top] * 100:.0f}% of the energy between 20 and 1000 Hz).</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(6.2, 1.5))
                    names = list(parts)[::-1]
                    vals = [parts[k] * 100 for k in names]
                    ax.barh(names, vals, color=INK, height=0.5)
                    for i, v in enumerate(vals):
                        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=8.5, family="monospace")
                    ax.set_xlim(0, 108); ax.set_xlabel("share of energy")
                    style_ax(ax)
                    fig.subplots_adjust(left=0.24, right=0.97, top=0.95, bottom=0.3)
                    show_fig(fig)
                    st.dataframe(table, hide_index=True)
                    st.markdown('<div class="cap">The "normal" and "abnormal" columns are the typical values (medians) of the training recordings in each class. '
                                + esc("What each number means: " + "; ".join(f"{v[0]} = {v[1]}" for v in FEATURE_TEXT.values()) + ".")
                                + '</div>', unsafe_allow_html=True)
                    callout("<b>This describes the sound. It is not a diagnosis</b> and it does not name a murmur or a condition. "
                            "Each of these numbers separates normal from abnormal only weakly on its own (see Data insights), and the "
                            "recording device and noise change them too. The decision comes from the CNN, not from this table.",
                            warn=True)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ------------------------------------------------------------------- Retrain
elif page == "Retrain":
    page_header("Retrain", "Teach the model with new recordings",
                "Upload labelled recordings, then retrain. The saved model is fine-tuned on the new data plus a sample of the old, "
                "compared with the old model on the same validation set, and kept only if it is not worse.")
    up_info = api_get("/uploads")
    if "_error" in up_info:
        callout(f"<b>The API is not reachable.</b> {esc(up_info['_error'])}", warn=True)
        st.stop()
    c = up_info["counts"]["pending"]
    total_wait = c.get("normal", 0) + c.get("abnormal", 0)
    thr_n = up_info["auto_retrain_threshold"]

    left, right = st.columns([5, 7], gap="large")
    with left:
        section("·", "Waiting for retraining")
        ledger([
            ("Normal recordings waiting", esc(c.get("normal", 0))),
            ("Abnormal recordings waiting", esc(c.get("abnormal", 0))),
            ("Until automatic retraining", esc(up_info["recordings_until_auto_retrain"])),
        ])
        pct = min(100, total_wait / max(thr_n, 1) * 100)
        st.markdown(f'<div style="height:6px;background:{RULE};margin:.9rem 0 .35rem"><div style="height:6px;width:{pct:.0f}%;'
                    f'background:{INK}"></div></div><div class="cap">{total_wait} of {thr_n} recordings needed to start '
                    'retraining by itself.</div>', unsafe_allow_html=True)

    with right:
        section("01", "Add labelled recordings")
        mode = st.radio("How do you want to upload?", ["Many files with one label", "A .zip with normal/ and abnormal/ folders"],
                        horizontal=True)
        if mode.startswith("Many"):
            label = st.selectbox("Label of these recordings", ["normal", "abnormal"])
            files = st.file_uploader("Recordings", type=["wav", "flac", "ogg"], accept_multiple_files=True)
            if files and st.button("Upload recordings"):
                res = api_post("/upload", data={"label": label}, files=[("files", (f.name, f.getvalue())) for f in files])
                if "_error" in res:
                    callout(f"<b>Upload failed.</b> {esc(res['_error'])}", warn=True)
                else:
                    callout(f"Saved <b>{res['saved']}</b>, failed <b>{res['failed']}</b>. "
                            f"Automatic retraining started: {res['auto_retrain_started']}.")
                    for r in res["results"]:
                        if not r["saved"]:
                            callout(esc(r["message"]), warn=True)
        else:
            z = st.file_uploader("Zip file", type=["zip"])
            if z and st.button("Upload zip"):
                res = api_post("/upload_zip", files={"file": (z.name, z.getvalue())}, timeout=600)
                if "_error" in res:
                    callout(f"<b>Upload failed.</b> {esc(res['_error'])}", warn=True)
                else:
                    callout(f"Saved <b>{res['saved']}</b>, failed <b>{res['failed']}</b>. "
                            f"Automatic retraining started: {res['auto_retrain_started']}.")
                    for r in res["results"]:
                        if not r["saved"]:
                            callout(esc(r["message"]), warn=True)

        section("02", "Retrain")
        if st.button("Retrain now"):
            res = api_post("/retrain")
            callout(f"<b>Could not start.</b> {esc(res['_error'])}" if "_error" in res else esc(res["message"]),
                    warn="_error" in res)

        status = api_get("/retrain/status")
        if "_error" not in status:
            dotcol = ABNORMAL if status["error"] else (INK if status["running"] else FAINT)
            st.markdown(f'<div class="sub"><span class="dot" style="background:{dotcol}"></span><b>Status:</b> '
                        f'{esc(status["message"])}</div>', unsafe_allow_html=True)
            if status["error"]:
                callout(esc(status["error"]), warn=True)
            if status["result"]:
                r = status["result"]
                verdict = "The new model is now live." if r["promoted"] else "The new model was worse, so the old model was kept."
                callout(f"<b>Version {esc(r['version'])}:</b> validation AUC "
                        f"<span class='num'>{r['old']['roc_auc']:.3f} → {r['new']['roc_auc']:.3f}</span>, recall "
                        f"<span class='num'>{r['old']['recall']:.2f} → {r['new']['recall']:.2f}</span>. {verdict}",
                        warn=not r["promoted"])
            if status["running"]:
                time.sleep(3)
                st.rerun()

    section("03", "Model history")
    vers = api_get("/versions")
    if isinstance(vers, list) and vers:
        df = pd.DataFrame(vers)[["id", "created_at", "trigger", "n_new_recordings", "n_new_clips",
                                 "old_auc", "new_auc", "old_recall", "new_recall", "promoted", "note"]]
        st.dataframe(df.round(3), hide_index=True)
        d = df.iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.4, 0.6 + 0.45 * len(d)))
        ypos = np.arange(len(d))[::-1]
        for y, (_, r) in zip(ypos, d.iterrows()):
            ax.plot([r["old_auc"], r["new_auc"]], [y, y], color=RULE, linewidth=2.2, zorder=1)
            ax.scatter(r["old_auc"], y, s=55, facecolor=PAPER, edgecolor=FAINT, linewidth=1.8, zorder=2)
            ax.scatter(r["new_auc"], y, s=55, color=INK if r["promoted"] else ABNORMAL, zorder=3)
            ax.text(max(r["old_auc"], r["new_auc"]) + 0.004, y, "kept" if r["promoted"] else "rejected", va="center", fontsize=8,
                    color=MUTED)
        ax.set_yticks(ypos); ax.set_yticklabels([f"version {int(i)}" for i in d["id"]])
        lo = float(min(d["old_auc"].min(), d["new_auc"].min())) - 0.02
        ax.set_xlim(lo, float(max(d["old_auc"].max(), d["new_auc"].max())) + 0.03)
        ax.set_xlabel("validation ROC-AUC: before (hollow) and after (filled) each retraining")
        style_ax(ax, "x")
        fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.3)
        show_fig(fig)
    else:
        st.markdown('<div class="sub">No retraining has happened yet.</div>', unsafe_allow_html=True)
