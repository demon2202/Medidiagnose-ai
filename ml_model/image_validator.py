import os
import numpy as np

# TensorFlow is NOT required for validation. Kept optional so importing this
# module can never break the server when TF is missing/mismatched.
try:  # pragma: no cover
    import tensorflow as _tf  # noqa: F401
    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATOR_MODEL_PATH = os.path.join(SCRIPT_DIR, "image_validator_model.h5")
VALIDATOR_CONFIG_PATH = os.path.join(SCRIPT_DIR, "image_validator_config.json")

IMG_SIZE = 224

# Canonical image-type catalogue (kept for backwards compatibility / configs).
IMAGE_TYPES = {
    0: {"code": "skin_lesion", "name": "Skin Lesion/Dermoscopy", "valid_for": ["skin"]},
    1: {"code": "xray_chest", "name": "Chest X-Ray", "valid_for": ["xray", "pneumonia"]},
    2: {"code": "mammogram", "name": "Breast Ultrasound", "valid_for": ["breast"]},
    3: {"code": "ecg", "name": "ECG/Heart Scan", "valid_for": ["heart"]},
    4: {"code": "other", "name": "Non-Medical/Unrecognized", "valid_for": []},
}

# ---------------------------------------------------------------------------
# Thresholds - every number below is justified by _featstats.json
# ---------------------------------------------------------------------------
COLOUR_RGB_DIFF = 0.05   # skin p5 = 0.142 ; gray = 0.0
COLOUR_SAT      = 0.05   # skin p5 = 0.115 ; gray = 0.0

SKIN_GRAY_RGB   = 0.04   # reject if rgb_diff < this AND sat < SKIN_GRAY_SAT
SKIN_GRAY_SAT   = 0.05
SKIN_DOC_BRIGHT = 0.88   # skin p95 brightness = 0.751
SKIN_DOC_BRIGHTR= 0.60   # skin p95 bright_ratio = 0.702 (kept lenient)
SKIN_DOC_EDGE   = 0.010  # HAM p50 edge_density = 0.0015

XRAY_DARK_MIN   = 0.08   # chest p5 brightness = 0.374 ; mammo/black frames lower
ECG_HISTPEAK    = 0.70   # rendered ECG peak ~0.93 ; xray peak p95 = 0.153
ECG_ENTROPY     = 1.00   # rendered ECG entropy ~0.34

BREAST_DOC_BRIGHT = 0.92
BREAST_DOC_EDGE   = 0.006

HEART_ECG_BRIGHT  = 0.80
HEART_DARK_BRIGHT = 0.05
HEART_DARK_RATIO  = 0.90


def _to_single(img_array):
    """Accept (H,W,C), (1,H,W,C), (H,W) and return a single 2-D or 3-D image."""
    arr = np.asarray(img_array, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def analyze_image_statistics(img_array):
    """Return the full metric dict used by the gate (and by the debug page)."""
    arr = _to_single(img_array)
    stats = {}

    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Some callers pass non-normalised 0..255 arrays - normalise defensively.
        scale = 255.0 if float(np.max(arr)) > 1.5 else 1.0
        a = arr[:, :, :3] / scale
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        rgb_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)))
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sat = np.where(mx > 0, (mx - mn) / (mx + 1e-7), 0.0)
        gray = a.mean(axis=2)
        stats["rgb_diff"] = rgb_diff
        stats["mean_saturation"] = float(np.mean(sat))
        stats["skin_tone_ratio"] = float(np.mean(
            (r > 0.3) & (r < 0.9) & (g > 0.2) & (g < 0.8) & (b > 0.1) & (b < 0.7) & (r > g) & (g > b)
        ))
    else:
        gray = arr[:, :, 0] if arr.ndim == 3 else arr
        if float(np.max(gray)) > 1.5:
            gray = gray / 255.0
        stats["rgb_diff"] = 0.0
        stats["mean_saturation"] = 0.0
        stats["skin_tone_ratio"] = 0.0

    stats["overall_brightness"] = float(np.mean(gray))
    stats["dark_region_ratio"] = float(np.mean(gray < 0.15))
    stats["bright_region_ratio"] = float(np.mean(gray > 0.75))

    gx = np.abs(gray[1:, :] - gray[:-1, :])
    gy = np.abs(gray[:, 1:] - gray[:, :-1])
    stats["edge_intensity"] = float(np.mean(gx) + np.mean(gy))
    gxf = np.zeros_like(gray); gyf = np.zeros_like(gray)
    gxf[1:, :] = gx; gyf[:, 1:] = gy
    stats["edge_density"] = float(np.mean(np.maximum(gxf, gyf) > 0.1))

    hist, _ = np.histogram(gray.flatten(), bins=50, range=(0, 1))
    hn = hist / (hist.sum() + 1e-7)
    stats["histogram_entropy"] = float(-np.sum(hn * np.log(hn + 1e-7)))
    stats["histogram_peak"] = float(hn.max())
    stats["has_grid_pattern"] = float(
        np.var(np.mean(gray, axis=0)) + np.var(np.mean(gray, axis=1))
    )
    stats["is_grayscale"] = bool(
        stats["rgb_diff"] < 0.045 and stats["mean_saturation"] < 0.045
    )
    return stats


def _result(is_valid, expected_type, stats, detected, message, suggestion=None,
            confidence=0.8, reasons=None):
    return {
        "is_valid": bool(is_valid),
        "predicted_type": detected,
        "predicted_code": detected.lower().replace(" ", "_").replace("/", "_"),
        "expected_type": expected_type,
        "confidence": float(confidence),
        "message": message,
        "suggestion": suggestion or "",
        "reasons": reasons or [],
        "image_stats": stats,
    }


def validate_image_type(img_array, expected_type):
    """Decide whether ``img_array`` matches ``expected_type``.

    expected_type: one of 'skin', 'xray'/'pneumonia', 'breast', 'heart'/'ecg'.
    Returns a dict with ``is_valid`` plus a human-readable ``message`` and
    ``suggestion`` and the raw ``image_stats``.
    """
    st = analyze_image_statistics(img_array)
    et = str(expected_type or "").lower().strip()

    rgb = st["rgb_diff"]; sat = st["mean_saturation"]; b = st["overall_brightness"]
    dark = st["dark_region_ratio"]; brightr = st["bright_region_ratio"]
    edens = st["edge_density"]; ent = st["histogram_entropy"]; peak = st["histogram_peak"]

    # ---------------------------------------------------------------- SKIN
    if et == "skin":
        reasons = []
        if rgb < SKIN_GRAY_RGB and sat < SKIN_GRAY_SAT:
            reasons.append(
                "rgb_diff=%.3f and saturation=%.3f are both near zero (a colour photo "
                "must have rgb_diff>=%.2f or saturation>=%.2f)" % (rgb, sat, SKIN_GRAY_RGB, SKIN_GRAY_SAT)
            )
            return _result(False, et, st, "Grayscale",
                           "This looks like a grayscale image, but skin lesion photos must be in colour.",
                           "Please upload a colour (RGB/JPEG/PNG) close-up photo of the lesion or mole.",
                           0.95, reasons)
        if b > SKIN_DOC_BRIGHT and brightr > SKIN_DOC_BRIGHTR and edens < SKIN_DOC_EDGE and sat < COLOUR_SAT:
            reasons.append(
                "brightness=%.2f, bright_ratio=%.2f, edge_density=%.3f - a flat, near-white page"
                % (b, brightr, edens)
            )
            return _result(False, et, st, "Document/ECG",
                           "This looks like a document, screenshot or ECG printout rather than a skin photo.",
                           "Please upload a close-up colour photo of the skin lesion or mole.",
                           0.85, reasons)
        return _result(True, et, st, "Skin Lesion",
                       "Valid colour skin photo (rgb_diff=%.3f, saturation=%.3f)." % (rgb, sat),
                       confidence=0.85)

    # --------------------------------------------------------- XRAY / PNEUMONIA
    if et in ("xray", "pneumonia"):
        reasons = []
        if rgb > COLOUR_RGB_DIFF or sat > COLOUR_SAT:
            reasons.append(
                "rgb_diff=%.3f, saturation=%.3f -> the image carries colour (grayscale scans have both ~0)"
                % (rgb, sat)
            )
            return _result(False, et, st, "Colour photo",
                           "This is a colour image, but a chest X-ray must be a grayscale scan.",
                           "Colour skin photos belong on the Skin tool. Please upload a grayscale chest X-ray.",
                           0.95, reasons)
        if peak > ECG_HISTPEAK or ent < ECG_ENTROPY:
            reasons.append("histogram_peak=%.2f, entropy=%.2f - flat line-art, not a radiograph"
                           % (peak, ent))
            return _result(False, et, st, "ECG/Line-art",
                           "This looks like an ECG printout or line-art, not a chest X-ray.",
                           "Please upload a grayscale chest X-ray image.",
                           0.85, reasons)
        if b < XRAY_DARK_MIN:
            reasons.append("brightness=%.3f is below the %.2f floor (chest X-rays are %.2f-%.2f)"
                           % (b, XRAY_DARK_MIN, 0.37, 0.59))
            return _result(False, et, st, "Too dark",
                           "This image is far too dark to be a chest X-ray (it may be a mammogram).",
                           "Please upload a properly exposed chest X-ray. Use the Breast tool for mammograms.",
                           0.80, reasons)
        return _result(True, et, st, "Chest X-Ray",
                       "Valid chest X-ray (brightness=%.2f, contrast ok)." % b,
                       confidence=0.82)

    # ------------------------------------------------------------- BREAST
    if et == "breast":
        reasons = []
        if rgb > COLOUR_RGB_DIFF or sat > COLOUR_SAT:
            reasons.append("rgb_diff=%.3f, saturation=%.3f -> colour image" % (rgb, sat))
            return _result(False, et, st, "Colour photo",
                           "This is a colour image, but a breast ultrasound/mammogram must be grayscale.",
                           "Please upload a grayscale breast ultrasound or mammogram.",
                           0.95, reasons)
        if b > BREAST_DOC_BRIGHT and edens < BREAST_DOC_EDGE and peak > 0.30:
            reasons.append("brightness=%.2f, edge_density=%.3f, hist_peak=%.2f - smooth white page"
                           % (b, edens, peak))
            return _result(False, et, st, "Document/ECG",
                           "This looks like a document or ECG printout, not a breast scan.",
                           "Please upload a grayscale breast ultrasound or mammogram.",
                           0.85, reasons)
        return _result(True, et, st, "Mammogram/Breast Ultrasound",
                       "Valid grayscale breast scan.", confidence=0.82)

    # --------------------------------------------------------- HEART / ECG
    if et in ("heart", "ecg"):
        reasons = []
        if rgb > COLOUR_RGB_DIFF and st["skin_tone_ratio"] > 0.30 and sat > 0.30:
            reasons.append("skin_tone_ratio=%.2f, saturation=%.2f -> looks like a skin photo"
                           % (st["skin_tone_ratio"], sat))
            return _result(False, et, st, "Skin photo",
                           "This looks like a colour skin photo, not an ECG or heart scan.",
                           "Please upload an ECG printout or echocardiogram image.",
                           0.85, reasons)
        is_ecg_paper = b > HEART_ECG_BRIGHT and ent < 1.5
        is_dark_trace = edens > 0.05 and (b < 0.20 or dark > HEART_DARK_RATIO)
        if not (is_ecg_paper or is_dark_trace) and b < 0.15 and dark > 0.80 and ent < 1.0:
            reasons.append("brightness=%.2f, dark_ratio=%.2f, entropy=%.2f - blank/near-black frame"
                           % (b, dark, ent))
            return _result(False, et, st, "Blank/Dark",
                           "This image is too dark and featureless to be an ECG or heart scan.",
                           "Please upload an ECG printout (bright paper) or an echocardiogram.",
                           0.80, reasons)
        return _result(True, et, st, "ECG/Heart Scan",
                       "Accepted as an ECG/heart scan.", confidence=0.78)

    # Unknown requested modality -> pass through (server validates elsewhere).
    return _result(True, et, st, "Unknown", "Image validation passed.", confidence=0.5)


# ---------------------------------------------------------------------------
# Backwards-compatible entry points
# ---------------------------------------------------------------------------
class ImageValidator:
    """Thin wrapper kept so older imports keep working."""

    def __init__(self, model_path=None):  # noqa: D401
        self.model = None
        self.use_ml = False

    def analyze_image_statistics(self, img_array):
        return analyze_image_statistics(img_array)

    def validate_image(self, img_array, expected_type):
        return validate_image_type(img_array, expected_type)


_validator_instance = None


def get_validator():
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ImageValidator()
    return _validator_instance


def validate_medical_image(img_array, expected_type):
    return validate_image_type(img_array, expected_type)


def train_validator_model():
    """No-op: the gate is rule/threshold based, no synthetic training needed."""
    print("Image validator: rule-based mode (no training required).")
    return None


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = {
        "skin": np.clip(rng.random((224, 224, 3)) * 0.4 + np.array([0.45, 0.30, 0.22]), 0, 1),
        "xray": np.repeat((rng.random((224, 224)) * 0.3 + 0.35)[..., None], 3, axis=-1),
    }
    for expected, img in demo.items():
        res = validate_image_type(img, expected)
        print(expected, "->", res["is_valid"], "|", res["message"])
