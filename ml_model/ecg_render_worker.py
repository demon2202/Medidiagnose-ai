"""Lightweight ECG render worker for multiprocessing.

Imported by spawned Pool workers, so it must NOT import TensorFlow (each
worker re-imports this module at spawn time on Windows, and importing TF in
16 processes at once deadlocks/slow-boots the pool to a crawl).
"""

import os
import sys
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from medidiagnose import inference_utils as MDI

IMG_SIZE = 224
_ERRORS_SHOWN = 0


def render_record(task):
    """Load one PTB-XL record and render it to a uint8 image.

    Args: (ecg_id, filename, class_idx) with filename relative to the
    PTB-XL directory. Returns (ecg_id, uint8 image or None, class_idx).
    """
    global _ERRORS_SHOWN
    import wfdb
    ecg_id, filename, cls = task
    try:
        path = filename.rsplit('.', 1)[0] if '.' in filename else filename
        signal, _ = wfdb.rdsamp(path)
        signal = np.asarray(signal, dtype=np.float64)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        if signal.shape[0] > 1000:
            signal = signal[:1000]
        img = MDI.signal_to_ecg_image(signal, size=IMG_SIZE)
        return ecg_id, np.clip(img * 255.0, 0, 255).astype(np.uint8), cls
    except Exception as e:
        if _ERRORS_SHOWN < 5:
            print(f'    [render error] {path}: {e}', flush=True)
            _ERRORS_SHOWN += 1
        return ecg_id, None, cls
