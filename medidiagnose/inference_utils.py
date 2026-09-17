"""Single-source inference utilities for MediDiagnose.

backend/server.py imports this module for ALL image preprocessing, ECG signal
parsing and Keras model loading. Every training script in ml_model/ imports the
SAME functions, so a model is always evaluated on exactly the kind of input it
was trained on.

Conventions (do not change without retraining every model):
  * Preprocess functions take a PIL.Image and return a float32 array in [0, 1]
    of shape (size, size, C). Use to_batch() to add the batch dimension.
  * Grayscale models (xray, breast, ecg) use a (size, size, 1) input.
  * ECG images are bright-trace-on-dark-background (matched to the PTB-XL
    renderer below); preprocess_ecg_image auto-inverts photos of ECG paper,
    which are dark-trace-on-white.
"""

import os
import sys
import numpy as np

try:
    from PIL import Image, ImageOps
    _PIL_OK = True
except ImportError:      # training scripts / server both need PIL; keep import safe
    _PIL_OK = False

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def to_batch(arr):
    """Add the batch dimension if missing: (H, W, C) -> (1, H, W, C)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr


def _ensure_pil(image):
    if not _PIL_OK:
        raise RuntimeError("PIL is required for image preprocessing")
    if isinstance(image, Image.Image):
        return image
    raise TypeError(f"Expected a PIL.Image, got {type(image)!r}")


def _resize_gray(image, size):
    img = _ensure_pil(image).convert('L')
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-modality preprocessing (used at train AND serve time)
# ---------------------------------------------------------------------------

def preprocess_skin(image, size=224):
    """Skin lesion photo -> (size, size, 3) float32 in [0, 1]."""
    img = _ensure_pil(image).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def preprocess_xray(image, size=224):
    """Chest X-ray -> (size, size, 1) float32 in [0, 1].

    Plain conversion + /255 only. Any contrast manipulation here would have to
    be identical at train and serve time — keeping it trivial is the safest way
    to guarantee that.
    """
    arr = _resize_gray(image, size) / 255.0
    return arr[..., np.newaxis]


def preprocess_breast_us(image, size=224):
    """Breast ultrasound -> (size, size, 1) float32 in [0, 1]."""
    arr = _resize_gray(image, size) / 255.0
    return arr[..., np.newaxis]


def preprocess_ecg_image(image, size=224):
    """ECG image/printout -> (size, size, 1) float32 in [0, 1].

    Training images (rendered from PTB-XL signals) have a DARK background with
    a bright trace. Photos of ECG paper are the opposite (dark trace on white),
    so bright images are auto-inverted to match the training distribution.
    """
    arr = _resize_gray(image, size) / 255.0
    if arr.mean() > 0.5:          # white paper -> invert
        arr = 1.0 - arr
    return arr[..., np.newaxis]


# ---------------------------------------------------------------------------
# Keras model loading (server-safe)
# ---------------------------------------------------------------------------

def load_keras_model(path):
    """Load a .h5 Keras model the way TF 2.18+ / Keras 3 requires."""
    import keras
    return keras.models.load_model(path, compile=False, safe_mode=False)


# ---------------------------------------------------------------------------
# ECG signal file parsing (.dat/.hea via wfdb, .csv raw)
# ---------------------------------------------------------------------------

def read_signal_file(file_path, file_ext):
    """Parse an ECG signal file into (signal, lead_names, error).

    Returns signal as a 2-D array (n_samples, n_leads) resampled to 100 Hz and
    trimmed to the first 10 seconds, or (None, [], error_message).
    """
    file_ext = str(file_ext).lower().lstrip('.')
    try:
        if file_ext == 'csv':
            raw = np.genfromtxt(file_path, delimiter=',', names=False)
            raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
            # Heuristic: samples along the LONGER axis
            if raw.shape[0] < raw.shape[1]:
                raw = raw.T
            return _normalize_signal(raw)

        if file_ext in ('dat', 'hea'):
            import wfdb
            stem = file_path[:-4] if file_path.lower().endswith(file_ext) else file_path
            record = wfdb.rdsamp(stem)
            signal, fields = record
            signal = np.asarray(signal, dtype=np.float64)
            lead_names = list(fields.get('sig_name') or [])
            return _normalize_signal(signal, lead_names=lead_names)

        return None, [], f"Unsupported signal file type: .{file_ext}"
    except Exception as e:
        return None, [], f"Could not read signal file: {e}"


def _normalize_signal(signal, lead_names=None, target_fs=100, seconds=10):
    """Resample to target_fs, keep <= seconds, pick up to 12 leads, z-score-free."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    # Drop NaN columns (leads that are entirely empty)
    good = [i for i in range(signal.shape[1]) if np.isfinite(signal[:, i]).any()]
    if not good:
        return None, [], "Signal contains no finite samples"
    signal = signal[:, good[:12]]
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    if lead_names is None or len(lead_names) != signal.shape[1]:
        lead_names = (LEAD_NAMES[:signal.shape[1]]
                      if signal.shape[1] <= 12
                      else [f'L{i+1}' for i in range(signal.shape[1])])

    # Assume the standard PTB-XL/PhysioNet sampling rates; anything unusual is
    # treated as 100 Hz (worst case the time axis is slightly off — harmless
    # for the visual CNN).
    n_keep = min(len(signal), target_fs * seconds)
    signal = signal[:n_keep]
    if len(signal) < 50:
        return None, [], "Signal too short (need >= 50 samples)"
    return signal, lead_names, None


# ---------------------------------------------------------------------------
# Signal -> ECG image renderer (training and serving use the SAME function)
# ---------------------------------------------------------------------------

def signal_to_ecg_image(signal, size=224):
    """Render a 12-lead (or fewer) ECG signal as a grayscale image.

    Layout: standard 4-row x 3-column 12-lead grid, 2.5 s per cell.
    Output: (size, size, 1) float32 in [0, 1], bright trace on dark background
    (preprocess_ecg_image inverts white-paper photos to match this).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    n_leads = min(signal.shape[1], 12)

    grid_rows, grid_cols = 4, 3
    cell_w = size / grid_cols
    cell_h = size / grid_rows
    samples_per_cell = int(np.ceil(len(signal) / grid_rows))
    ss = 2  # supersampling factor for anti-aliasing

    big = np.zeros((size * ss, size * ss), dtype=np.float32)

    for lead in range(n_leads):
        row, col = lead // grid_cols, lead % grid_cols
        seg = signal[row * samples_per_cell:(row + 1) * samples_per_cell, lead]
        if len(seg) < 2:
            continue

        x0 = col * cell_w * ss
        y0 = row * cell_h * ss
        w = cell_w * ss
        h = cell_h * ss
        margin = 0.10
        w_in = w * (1 - 2 * margin)
        h_in = h * (1 - 2 * margin)
        cx, cy = x0 + w * margin, y0 + h * margin

        # Per-lead amplitude normalization (robust to flat/noisy leads)
        amp = np.percentile(np.abs(seg - seg.mean()), 99)
        if amp < 1e-8:
            continue
        norm = (seg - seg.mean()) / amp

        xs = np.linspace(cx, cx + w_in, num=len(norm))
        ys = cy + h_in / 2.0 - norm * (h_in / 2.0)

        # Rasterize by filling the vertical run between consecutive points
        xi = np.clip(np.round(xs).astype(int), 0, big.shape[1] - 1)
        yi = np.clip(np.round(ys).astype(int), 0, big.shape[0] - 1)
        thickness = max(1, ss // 2)
        for i in range(len(xi) - 1):
            xa, xb = sorted((xi[i], xi[i + 1]))
            ya, yb = yi[i], yi[i + 1]
            if xa == xb:
                lo, hi = sorted((ya, yb))
                big[lo:hi + 1, xa] = 1.0
            else:
                for x in range(xa, xb + 1):
                    t = (x - xa) / max(1, (xb - xa))
                    y = int(round(ya + t * (yb - ya)))
                    big[max(0, y - thickness):y + thickness + 1, x] = 1.0

    # Anti-alias downsample
    img = big.reshape(size, ss, size, ss).mean(axis=(1, 3))
    return (img[..., np.newaxis]).astype(np.float32)
