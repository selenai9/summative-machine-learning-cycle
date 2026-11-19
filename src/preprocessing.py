"""Audio preprocessing for heart-sound (phonocardiogram) classification.

Pipeline for ONE recording:
    .wav file -> load (mono, 2000 Hz) -> band-pass filter (25-900 Hz)
              -> cut into non-overlapping 5 s clips -> mel spectrogram (dB)
              -> scale to 0..1 -> array of shape (n_clips, 40, 251, 1)

The same functions are used by the notebook, the API and the retraining code,
so training and prediction always see identical inputs.
"""
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt

# ---- fixed settings (changing any of these means the model must be retrained) ----
SR = 2000                      # sample rate used by the PhysioNet 2016 recordings (Hz)
CLIP_SECONDS = 5               # every model input is a 5 second clip
CLIP_SAMPLES = SR * CLIP_SECONDS
N_FFT = 256                    # spectrogram window (256 samples = 0.128 s)
HOP_LENGTH = 40                # window step (40 samples = 0.02 s)
N_MELS = 40                    # number of frequency bands
FMIN, FMAX = 20, 1000          # frequency range of the spectrogram (Hz)
BAND_LOW, BAND_HIGH = 25, 900  # band-pass filter range (Hz)
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH      # 251 time steps
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)            # (40, 251, 1)
DB_RANGE = 80.0                # dB values are kept between -80 and 0, then scaled to 0..1


def bandpass_filter(y, low=BAND_LOW, high=BAND_HIGH, sr=SR, order=4):
    """Remove very low drift (hand movement, breathing) and near-Nyquist hiss."""
    if len(y) < 200:
        return y
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype("float32")


def load_audio(path, bandpass=True):
    """Read an audio file as mono float32 at SR Hz. Raises ValueError if unreadable."""
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"Could not read audio file: {exc}")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if bandpass:
        y = bandpass_filter(y)
    return y


def make_clips(y, max_clips=None):
    """Cut a recording into non-overlapping 5 s clips.

    * shorter than 5 s  -> zero-padded to one clip
    * more clips than max_clips -> keep max_clips clips spread evenly over the recording
    """
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    n_clips = len(y) // CLIP_SAMPLES
    starts = np.arange(n_clips) * CLIP_SAMPLES
    if max_clips and n_clips > max_clips:
        keep = np.linspace(0, n_clips - 1, max_clips).round().astype(int)
        starts = starts[keep]
    return [y[s:s + CLIP_SAMPLES] for s in starts]


def middle_clip(y):
    """The 5 s clip from the middle of a recording (used for exploration)."""
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    start = (len(y) - CLIP_SAMPLES) // 2
    return y[start:start + CLIP_SAMPLES]


def to_melspec(clip):
    """5 s clip -> mel spectrogram in dB, shape (40, 251), values from -80 to 0."""
    clip = clip / (np.max(np.abs(clip)) + 1e-9)          # volume scaling: removes loudness as a clue
    mel = librosa.feature.melspectrogram(
        y=clip, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return librosa.power_to_db(mel, ref=np.max)


def scale_spec(spec_db):
    """dB values (-80..0) -> 0..1 float32, the range the CNN expects."""
    return np.clip((spec_db + DB_RANGE) / DB_RANGE, 0.0, 1.0).astype("float32")


def recording_to_array(source, max_clips=None, bandpass=True):
    """Path (or already-loaded signal) -> model input array (n_clips, 40, 251, 1)."""
    y = load_audio(source, bandpass=bandpass) if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__") else source
    clips = make_clips(y, max_clips=max_clips)
    X = np.stack([scale_spec(to_melspec(c)) for c in clips])
    return X[..., np.newaxis]


def extract_features(y):
    """A few interpretable numbers describing one recording (for the data-insight plots).

    Computed on the middle 5 s, volume-scaled, so loudness is not a feature.
    """
    clip = middle_clip(y)
    clip = clip / (np.max(np.abs(clip)) + 1e-9)
    power = np.abs(librosa.stft(clip, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    p = power.mean(axis=1)                                  # average power at each frequency
    band = (freqs >= FMIN) & (freqs <= FMAX)
    total = p[band].sum() + 1e-12

    def share(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(p[m].sum() / total)

    return {
        "low_share": share(FMIN, 150),                      # share of energy at 20-150 Hz
        "mid_share": share(150, 400),                       # 150-400 Hz
        "high_share": share(400, FMAX + 1),                 # 400-1000 Hz
        "spectral_centroid_hz": float((freqs[band] * p[band]).sum() / total),   # "centre of mass" of the spectrum
        "spectral_flatness": float(np.exp(np.mean(np.log(p[band] + 1e-12))) / (np.mean(p[band]) + 1e-12)),
    }


def quality_report(y_raw):
    """Simple warnings about a raw (unfiltered) signal, shown to the user next to a prediction."""
    warnings = []
    duration = len(y_raw) / SR
    peak = float(np.max(np.abs(y_raw))) if len(y_raw) else 0.0
    if duration < 1.0:
        warnings.append("Recording is shorter than 1 second; the result is unreliable.")
    elif duration < CLIP_SECONDS:
        warnings.append(f"Recording is shorter than {CLIP_SECONDS} s; it was padded with silence.")
    if peak < 1e-4:
        warnings.append("Recording is almost silent.")
    elif np.mean(np.abs(y_raw) >= 0.99 * peak) > 0.01 and peak >= 0.5:
        warnings.append("Recording looks clipped (too loud); the result may be unreliable.")
    return {"duration_s": round(duration, 1), "peak": round(peak, 4), "warnings": warnings}
