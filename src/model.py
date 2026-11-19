"""CNN model, training, evaluation and fine-tuning for heart-sound classification.

Labels: 0 = normal, 1 = abnormal. The model outputs the probability of "abnormal".
"""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from .preprocessing import INPUT_SHAPE, N_FRAMES, N_MELS

SEED = 42


# --------------------------------------------------------------------- model
def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.Precision(name="precision")],
    )
    return model


def build_cnn(input_shape=INPUT_SHAPE, filters=(16, 32, 64), dropout=0.3, l2=1e-4,
              use_bn=True, lr=1e-3):
    """Small CNN: 3 x (Conv -> [BatchNorm] -> ReLU -> MaxPool) -> pooling -> dense -> sigmoid.

    use_bn=False, dropout=0, l2=0 gives the plain version; the defaults give the optimized one.
    """
    reg = regularizers.l2(l2) if l2 else None
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", kernel_regularizer=reg)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(keras.Model(inputs, outputs), lr)


def save_model(model, path):
    """Save atomically (write a temp file, then rename) so a running API never reads half a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path.replace(".keras", ".tmp.keras")
    model.save(tmp)
    os.replace(tmp, path)


def load_model(path, compile_for_training=False, lr=1e-3):
    model = keras.models.load_model(path, compile=False)
    if compile_for_training:
        compile_model(model, lr)
    return model


# ------------------------------------------------------------- augmentation
def _mask(x, length, max_width, axis):
    """Zero out a random band along one axis (SpecAugment-style)."""
    width = tf.random.uniform([], 0, max_width + 1, dtype=tf.int32)
    start = tf.random.uniform([], 0, tf.maximum(length - width, 1), dtype=tf.int32)
    idx = tf.range(length)
    keep = tf.cast(tf.logical_not(tf.logical_and(idx >= start, idx < start + width)), x.dtype)
    shape = [1, 1, 1]
    shape[axis] = length
    return x * tf.reshape(keep, shape)


def spec_augment(x, y):
    """Random time shift + one time mask + one frequency mask on a (40, 251, 1) spectrogram."""
    shift = tf.random.uniform([], -25, 26, dtype=tf.int32)
    x = tf.roll(x, shift, axis=1)
    x = _mask(x, N_FRAMES, 20, axis=1)
    x = _mask(x, N_MELS, 4, axis=0)
    return x, y


def make_dataset(X, y, batch_size=32, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), np.asarray(y, dtype="float32")))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(spec_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------- training
def compute_class_weights(y):
    """Give the rare class (abnormal) a larger weight so the model does not ignore it."""
    y = np.asarray(y)
    n, n1 = len(y), int(y.sum())
    n0 = n - n1
    return {0: n / (2.0 * max(n0, 1)), 1: n / (2.0 * max(n1, 1))}


def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32,
                augment=True, use_class_weights=True, patience=6, verbose=2):
    """Train with early stopping on validation AUC and learning-rate reduction on plateau."""
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                                          patience=max(2, patience // 2), min_lr=1e-6),
    ]
    history = model.fit(
        make_dataset(X_train, y_train, batch_size, augment=augment, shuffle=True),
        validation_data=make_dataset(X_val, y_val, batch_size),
        epochs=epochs, shuffle=False,   # shuffling is done inside make_dataset
        class_weight=compute_class_weights(y_train) if use_class_weights else None,
        callbacks=callbacks, verbose=verbose,
    )
    return history


def predict_clips(model, X, batch_size=64):
    """Probability of 'abnormal' for each clip."""
    return model.predict(X.astype("float32"), batch_size=batch_size, verbose=0).ravel()


def fine_tune(model, X_new, y_new, X_replay, y_replay, X_val, y_val,
              epochs=8, lr=1e-4, batch_size=32, replay_multiplier=3, seed=SEED, verbose=2):
    """Continue training a pretrained model on NEW data mixed with a sample of OLD data.

    Mixing in old ("replay") data stops the model forgetting what it learned before.
    A smaller learning rate keeps the update gentle.
    """
    rng = np.random.RandomState(seed)
    n_replay = min(len(X_replay), max(200, replay_multiplier * len(X_new)))
    pick = rng.choice(len(X_replay), n_replay, replace=False)
    X_mix = np.concatenate([X_new, X_replay[pick]]).astype("float32")
    y_mix = np.concatenate([y_new, y_replay[pick]]).astype("float32")
    compile_model(model, lr)
    history = train_model(model, X_mix, y_mix, X_val, y_val, epochs=epochs,
                          batch_size=batch_size, patience=3, verbose=verbose)
    return model, history


# --------------------------------------------------------------- evaluation
def aggregate_by_recording(rec_ids, probs, labels):
    """Average the clip probabilities of each recording -> one probability per recording."""
    frame = pd.DataFrame({"rec": rec_ids, "p": probs, "y": labels})
    grouped = frame.groupby("rec").agg(p=("p", "mean"), y=("y", "first"))
    return grouped.index.to_numpy(), grouped["p"].to_numpy(), grouped["y"].to_numpy().astype(int)


def evaluate_binary(y_true, y_prob, threshold=0.5):
    """Accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, PR-AUC and the confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_true, y_prob, min_recall=0.85):
    """Pick the decision threshold on VALIDATION data.

    Among thresholds that catch at least `min_recall` of the abnormal recordings,
    take the one with the fewest false alarms. If none reaches it, use the best-F1 threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= min_recall)[0]
    if len(ok):
        t = float(thr[ok[0]])
        return min(t, 1.0)
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_prob >= c).astype(int), zero_division=0) for c in candidates]
    return float(candidates[int(np.argmax(scores))])


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
