# Heart Sound Classifier: an end-to-end ML pipeline (normal vs abnormal)

An end-to-end machine learning pipeline on **audio data**: a stethoscope recording (phonocardiogram) goes in, **normal or abnormal** comes out.
It covers data acquisition, preprocessing, model training and evaluation, a prediction API, a web UI, **retraining on newly uploaded data**, Docker deployment and a **load test**.

> **Educational prototype, not a medical device.** A "normal" result does not rule out heart disease. The model detects *abnormal vs normal*; it does not name a murmur type or a diagnosis.

| | |
|---|---|
| **Video demo (YouTube)** | `https://youtu.be/YThzkhz27IA` |
| **Live UI** | <https://heart-sound-ui.onrender.com> |
| **Live API docs** | <https://heart-sound-api.onrender.com/docs> |
| **GitHub repository** | <https://github.com/selenai9/summative-machine-learning-cycle> |

## 1. Problem and data

- **Problem:** screening for abnormal heart sounds (for example murmurs) from a short chest recording, where cardiologists are scarce.
- **Data:** PhysioNet/CinC Challenge 2016 open-access heart-sound database: 3,240 recordings (5 to 120 s, 2000 Hz) from six sources (a to f), labelled normal/abnormal, about 20.5% abnormal. Liu et al. (2016), *An open access database for the evaluation of heart sound algorithms*, Physiological Measurement 37(12). <https://physionet.org/content/challenge-2016/1.0.0/>
- **Why it is not tabular:** the model input is a **mel spectrogram image** (40 frequency bands x 251 time steps) made from each 5 s audio clip.
- **Main risks handled:** class imbalance (class weights, recall/precision/F1/AUC instead of accuracy alone), a *source shortcut* (the abnormal rate differs a lot between sources, so the split is stratified by source and label and results are reported per source), leakage (clips of one recording never end up in different splits), and noisy recordings (band-pass filter, quality warnings, results for good vs poor quality).

## 2. How it works

```
.wav  ->  band-pass 25-900 Hz  ->  5 s clips  ->  mel spectrogram (40 x 251)  ->  CNN  ->  P(abnormal) per clip
                                                                                          -> average over clips -> compare with threshold -> normal / abnormal
```

```
 Streamlit UI  --HTTP-->  FastAPI  --> src/prediction.py --> models/heart_cnn.keras
 (status, insights,        |  \-> SQLite (uploads, model versions)
  predict, retrain)        \----> app/retrain.py (fine-tune the saved model on new uploads + replay data)
```

| Part | File(s) |
|---|---|
| Notebook (all steps, explained) | `notebook/heart_sound_pipeline.ipynb` |
| Preprocessing | `src/preprocessing.py` |
| Model, training, metrics, fine-tuning | `src/model.py` |
| Prediction | `src/prediction.py` |
| API (FastAPI) | `app/api.py`, `app/db.py`, `app/retrain.py`, `app/config.py` |
| UI (Streamlit) | `ui/app.py` |
| Docker / scaling / load test | `Dockerfile`, `Dockerfile.ui`, `docker-compose.yml`, `docker-compose.scale.yml`, `nginx.conf`, `locustfile.py`, `scripts/run_flood_tests.py` |
| Deployment | `render.yaml` |
| Saved model + files the app needs | `models/` (`heart_cnn.keras`, `threshold.json`, `metrics.json`, `replay_set.npz`), `data/features.csv` |
| Sample recordings | `data/test/normal`, `data/test/abnormal` (10 + 10 held-out test recordings), `data/retrain_demo.zip` (60 recordings for the bulk-upload demo) |
| Reset after a retraining demo | `scripts/reset_to_v0.py` (restores the original model) |

## 3. Results

Evaluated on **486 held-out test recordings** (99 abnormal, 387 normal; split by recording, stratified by source and label). The test set was used once, after the model and the decision threshold (0.643, chosen on the validation set so that recall is at least 0.85) were fixed. The same numbers are in `models/metrics.json` and on the UI "Status" page.

| Metric (test, recording level) | Test | Validation |
|---|---|---|
| ROC-AUC | 0.936 | 0.947 |
| PR-AUC | 0.776 | |
| Recall (abnormal recordings caught) | 0.848 | 0.850 |
| Specificity (normal recordings kept normal) | 0.845 | 0.860 |
| Precision | 0.583 | 0.612 |
| F1 | 0.691 | 0.711 |
| Accuracy | 0.846 | 0.858 |

Confusion matrix (test): 327 true negatives, 60 false positives, 15 false negatives, 84 true positives.

**Baselines on validation (Step 13, threshold 0.5):**

| Model | ROC-AUC | Recall | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| Majority class | 0.500 | 0.00 | 1.000 | 0.000 | 0.000 |
| Logistic regression (80 band features) | 0.900 | 0.87 | 0.790 | 0.518 | 0.649 |
| Random forest (80 band features) | 0.961 | 0.85 | 0.915 | 0.720 | 0.780 |
| CNN, optimized (chosen threshold) | 0.947 | 0.85 | 0.860 | 0.612 | 0.711 |

The random forest on simple band statistics matches or beats the CNN on validation. The CNN was kept because it works directly on the spectrogram and can be fine-tuned from the saved model during retraining; the forest is the benchmark it should be judged against.

Optimization techniques used: batch normalisation, dropout, L2 penalty, class weights, SpecAugment-style augmentation, early stopping, learning-rate reduction, band-pass filtering, and decision-threshold selection on validation data (see Steps 14 to 16).

**Results differ strongly by recording source (test set):**

| Source | Recordings | AUC | Recall | Specificity |
|---|---|---|---|---|
| e | 321 | 1.000 | 1.000 | 0.993 |
| a | 62 | 0.614 | 0.955 | 0.167 |
| b | 73 | 0.359 | 0.467 | 0.310 |

Sources c, d and f have only 5 to 17 test recordings, so their results are not meaningful. The overall AUC of 0.936 is carried by source e (66% of the test set). Inside sources a and b the model is close to or below chance, and the abnormal rate differs a lot between sources (a is about 71% abnormal, e about 8%), so part of the overall score is probably the model recognising the recording setup rather than the heart. By recording quality: good-quality recordings AUC 0.941, poor-quality 0.902 with more false alarms (specificity 0.70 vs 0.86).

**Limitations:** the dataset has no patient IDs (the same patient may appear in several recordings and splits); results differ strongly by source and the model should not be trusted on recordings from a new device or clinic without re-checking it there; the small sources cannot be judged; poor-quality recordings are harder; labels come from the PhysioNet database; this is not a diagnostic tool.

## 4. Run it on your computer

Requirements: **Python 3.12** (TensorFlow 2.21 does not install on Python 3.14), and Docker Desktop if you want to use Docker.

```bash
# 1. create an environment and install
conda create -n heart python=3.12 -y        # or: python -m venv .venv
conda activate heart                         # venv on Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-ui.txt
# if tensorflow-cpu cannot be installed on your system, use:  pip install tensorflow==2.21.0

# 2. start the API  (docs at http://localhost:8000/docs)
uvicorn app.api:app --port 8000

# 3. in a second terminal (also with the environment active), start the UI  (http://localhost:8501)
streamlit run ui/app.py
```

**Docker (API + UI):**
```bash
docker compose up --build
# UI: http://localhost:8501     API: http://localhost:8000/docs
```

**Retrain from scratch yourself:** open `notebook/heart_sound_pipeline.ipynb` in VS Code, Jupyter or Colab and run all cells. Step 2 downloads the PhysioNet data (about 350 MB, needs internet; `python -m awscli s3 sync` is used) and the notebook writes the model and the other files the app needs into `models/` and `data/`. It runs on CPU.

## 5. Using the app

1. **Status:** API uptime, model version, predictions served, average prediction time, saved evaluation metrics.
2. **Data insights:** abnormal rate per source, recording length by class, three frequency features compared between normal and abnormal (with how well each feature alone predicts the label), all from the training recordings only. Main finding: no single feature is strong across all sources (the best consistent ones reach AUC 0.6 to 0.66), and the spectral centroid is strong in source e only, so it depends on the source.
3. **Predict:** upload one `.wav`; the app shows its waveform and spectrogram, a **sound description** (which frequency bands hold the energy, five features compared with the typical normal and abnormal training recordings), and returns normal/abnormal with the probability and warnings. The description only describes the sound; it is not a diagnosis and does not name a murmur.
4. **Retrain:** upload labelled recordings (many files with one label, or `data/retrain_demo.zip` with `normal/` and `abnormal/` folders). Retraining starts when you press **Retrain now**, or **automatically** when 20 new recordings are waiting (`RETRAIN_THRESHOLD`).

### Retraining process (trigger)

1. Uploaded files are validated, saved in `data/uploads/<label>/` and recorded in the SQLite database (`data/app.db`) as `pending`.
2. A **trigger** starts retraining: the button, or automatically once `RETRAIN_THRESHOLD` (default 20) new recordings are waiting.
3. The recordings are preprocessed (`src/preprocessing.py`) into spectrogram clips.
4. The **saved model is loaded as the starting point** and fine-tuned (`src/model.py: fine_tune`) on the new clips mixed with a replay sample of old training clips, with a smaller learning rate.
5. The old and new model are compared on the same fixed validation set. The new model is kept **only if its validation AUC is not worse by more than 0.01**; the threshold is chosen again for it. The previous model is archived in `models/archive/`, the uploads move to `data/train/<label>/`, and the history is stored in the database and shown in the UI. If the new model is worse, the old one stays live and the uploads stay pending.
6. Every running API container notices the new model file and loads it on the next request, without a restart.

**Demo run:** uploading `data/retrain_demo.zip` (60 recordings, 152 clips) triggered retraining automatically. Validation AUC went from 0.948 to 0.938 and recall from 0.87 to 0.85, which is within the 0.01 margin, so the new model was kept as version 1. These recordings come from the training split, so this shows that the mechanics work (upload, trigger, fine-tune, compare, promote), not that retraining improves the model. `python scripts/reset_to_v0.py` restores the original model after such a demo.

## 6. Load test (Locust) with different numbers of containers

Setup: nginx in front of N identical API containers (`docker-compose.scale.yml`); Locust users mostly upload a recording to `/predict` (the heavy request) and sometimes call `/health`. Locust ran from the same Windows laptop as the containers: 40 simulated users, spawn rate 10 per second, 60 seconds per run.

```bash
pip install -r requirements-dev.txt
python scripts/run_flood_tests.py                       # runs 1, 2 and 3 containers, writes results/flood_results.md
# options: python scripts/run_flood_tests.py --counts 1 2 4 --users 80 --time 90s
```

Manual alternative:
```bash
docker compose -f docker-compose.scale.yml up -d --build --scale api=3
locust -f locustfile.py --host http://localhost:8080          # open http://localhost:8089
docker compose -f docker-compose.scale.yml down
```

**Results** (machine: HP Envy x360 laptop, Intel Core 5 120U (10 cores, 12 logical processors), 8 GB RAM, Windows 11 Pro; Docker Desktop with 12 CPUs and 3.75 GiB memory). Raw files are in `results/`.

| Containers | Requests | Failures | Requests/s | Median (ms) | Average (ms) | 95th percentile (ms) |
|---|---|---|---|---|---|---|
| 1 | 776 | 0 | 13.5 | 1300 | 1627 | 5600 |
| 2 | 1217 | 0 | 20.4 | 550 | 677 | 1500 |
| 3 | 1510 | 0 | 25.9 | 210 | 267 | 610 |

More containers raised throughput (13.5 to 25.9 requests per second) and cut response times (median 1300 ms to 210 ms, 95th percentile 5600 ms to 610 ms). Throughput roughly doubles with 3 containers instead of tripling because all containers share the same laptop CPU, so one machine cannot scale linearly. Each row is a single 60-second run, so small differences should not be over-read.

**Second set of runs, with Locust screenshots.** I repeated the test later the same day through the Locust web page (40 users, spawn rate 10 per second, about 2.5 minutes per run, 0 failures after a warm-up). The laptop was busier this time (Chrome, VS Code and Docker open, little free RAM), so the numbers are slower than the table above. Screenshots and the downloaded Locust CSV files are in `results/locust_ui/`.

| Containers | Requests | Failures | Requests/s | Median (ms) | Average (ms) | 95th percentile (ms) |
|---|---|---|---|---|---|---|
| 1 | 1192 | 0 | 7.7 | 3000 | 3748 | 8200 |
| 2 | 2960 | 0 | 17.7 | 890 | 987 | 1900 |
| 3 | 2914 | 0 | 17.2 | 750 | 1039 | 3300 |

Going from 1 to 2 containers more than doubled throughput (7.7 to 17.7 requests per second) and cut the median from 3000 ms to 890 ms. A third container did not help in this set: throughput stayed about the same (17.2 requests per second) and the 95th percentile was worse (3300 ms against 1900 ms). The first table showed a gain from 3 containers, so the two sets disagree on that point. Both sets ran on one laptop, which also runs Locust and nginx, so CPU and memory are shared and the result depends on how busy the machine is. The honest conclusion is that 2 containers clearly help on this machine and a third gives no reliable extra gain.

**Warm-up failures:** the first 3-container attempt in this second set had 662 failed requests out of 3201 (21%), all `502 Bad Gateway`, all within the first 22 seconds. nginx was sending requests before the three API containers had finished loading TensorFlow. After they were ready there were no failures, and the rerun above (started after all three containers were serving requests) had 0 failures. Wait for all API containers to be ready before starting a load test.

**Note on an earlier run:** the first 3-container run had 22 failed requests out of 679 (3.2%), all "connection forcibly closed" errors reported by Locust on Windows, with a worst response time of 30 s (`results/run1_flood_3_containers_stats.csv`). nginx logged no errors during that test, and the rerun with identical settings had 0 failures. The likely cause is the connection between Windows and Docker Desktop, but this was not proven.

## 7. Deployment (Render)

1. Push this repository to GitHub (the model and `data/features.csv` must be committed).
2. On Render: **New, Blueprint**, select the repository (`render.yaml` creates two Docker web services: `heart-sound-api` and `heart-sound-ui`).
3. After the API is live, set the UI service's environment variable `API_URL` to the API's public URL and redeploy the UI.
4. Free-tier notes: services sleep after inactivity (the first request is slow), memory is limited to 512 MB (TensorFlow is heavy; if the API is killed, use a paid instance or another host), and uploaded data and retrained models on the free tier are lost when the service restarts.

## 8. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, uptime, model version, predictions served |
| GET | `/model` | threshold, saved metrics, version history |
| POST | `/predict` | one recording (`file`) to label + probability |
| POST | `/upload` | several recordings (`files`) with one `label` (`normal` or `abnormal`) |
| POST | `/upload_zip` | a zip with `normal/` and `abnormal/` folders |
| GET | `/uploads` | pending uploads and distance to the automatic trigger |
| POST | `/retrain` | start retraining |
| GET | `/retrain/status` | progress and result of the last run |
| GET | `/versions` | model versions with before/after metrics |
