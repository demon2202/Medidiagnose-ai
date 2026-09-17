"""
train_heart_image_model.py — v3 (full PTB-XL + shared renderer)
===============================================================

WHY v2 FAILED (20% accuracy, everything predicted as Arrhythmia):

1. THE DATASET WAS EFFECTIVELY EMPTY. The local ptbxl_database.csv had been
   truncated to 150 rows while all 21,837 signal files were present — the
   model trained on ~120 images and was evaluated on 30. It has been replaced
   with the official full PhysioNet CSV (21,799 records).
2. TRAIN/SERVE RENDERER MISMATCH. v2 trained on matplotlib plots (white
   paper, lead labels, global y-scaling), but server.py renders signals with
   medidiagnose.inference_utils.signal_to_ecg_image (dark background, 4x3
   grid, per-lead amplitude normalization). The model never saw what it is
   asked to classify. Training below uses the inference renderer directly.
3. The split is now PATIENT-DISJOINT using PTB-XL's strat_fold
   (train=folds 1-8, val=fold 9, test=fold 10) — the official protocol.

Expected test accuracy: ~80-88% on 5-class with the full dataset.

Interfaces preserved (server.py compatibility):
  - HEART_IMAGE_MODEL_PATH, HEART_CONFIG_PATH unchanged
  - HEART_CLASSES, CLASS_NAMES, SCP_TO_CLASS, LEAD_NAMES unchanged
  - signal_to_grayscale_image() kept as a thin wrapper (deprecated — use
    medidiagnose.inference_utils.signal_to_ecg_image)
  - train_heart_image_model() signature unchanged
"""

import os
import sys
import ast
import json
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# ── Shared single-source helpers (train == serve) ───────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from medidiagnose import inference_utils as MDI
from medidiagnose import train_utils as TU

# ── wfdb ────────────────────────────────────────────────────────────────────
WFDB_AVAILABLE = False
try:
    import wfdb
    WFDB_AVAILABLE = True
    print("[OK] wfdb library available")
except ImportError:
    print("[ERR] wfdb not installed. Run: pip install wfdb")

# ── TensorFlow ──────────────────────────────────────────────────────────────
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print(f"[OK] TensorFlow {tf.__version__} available")
except ImportError:
    print("[ERR] TensorFlow not available")

import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ── Paths (UNCHANGED) ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
PTBXL_DIR = os.path.join(DATASET_DIR, 'ptb-xl')

HEART_IMAGE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'heart_image_model.h5')
HEART_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'heart_image_config.json')

IMG_SIZE = 224
SEED = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#                         CLASS DEFINITIONS (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

HEART_CLASSES = {
    0: {'code': 'normal',      'name': 'Normal',                  'severity': 'healthy'},
    1: {'code': 'mi',          'name': 'Myocardial Infarction',   'severity': 'critical'},
    2: {'code': 'arrhythmia',  'name': 'Arrhythmia',              'severity': 'moderate'},
    3: {'code': 'hf',          'name': 'Heart Failure Signs',     'severity': 'high'},
    4: {'code': 'hypertrophy', 'name': 'Ventricular Hypertrophy', 'severity': 'moderate'}
}

CLASS_NAMES = ['Normal', 'MI', 'Arrhythmia', 'Heart Failure', 'Hypertrophy']

SCP_TO_CLASS = {
    'NORM': 0, 'SR': 0,
    'IMI': 1, 'AMI': 1, 'LMI': 1, 'PMI': 1, 'ASMI': 1,
    'ILMI': 1, 'IPLMI': 1, 'IPMI': 1, 'MI': 1,
    'INJAL': 1, 'INJIL': 1, 'INJLA': 1, 'INJIN': 1, 'INJAS': 1,
    'ISC_': 1, 'ISCA': 1, 'ISCI': 1, 'STD_': 1, 'STE_': 1,
    'AFIB': 2, 'AFLT': 2, 'SVTAC': 2, 'PSVT': 2,
    'STACH': 2, 'SBRAD': 2, 'SARRH': 2,
    'BIGU': 2, 'TRIGU': 2, 'PAC': 2, 'PVC': 2,
    'VPRE': 2, 'WPW': 2, 'STTC': 2, 'NST_': 2,
    'LAFB': 3, 'LPFB': 3, 'IRBBB': 3, 'CRBBB': 3,
    'CLBBB': 3, 'ILBBB': 3, '1AVB': 3, '2AVB': 3, '3AVB': 3,
    'CD': 3, 'IVCB': 3,
    'LVH': 4, 'RVH': 4, 'LAO': 4, 'LAE': 4,
    'RAO': 4, 'RAE': 4, 'SEHYP': 4, 'HYP': 4,
}

LEAD_NAMES = MDI.LEAD_NAMES


def signal_to_grayscale_image(signal, img_size=(224, 224)):
    """Deprecated wrapper — kept for backward compatibility.

    Training and serving both use medidiagnose.inference_utils
    .signal_to_ecg_image (dark background, 4x3 grid, per-lead normalization).
    """
    return MDI.signal_to_ecg_image(signal, size=img_size[0])


# ══════════════════════════════════════════════════════════════════════════════
#                       PTB-XL DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def assign_primary_class(scp_codes):
    """Map a record's SCP codes to one of the 5 superclasses.

    Picks the disease class (1-4) whose code has the highest likelihood >= 50;
    falls back to Normal only if NORM/SR is present with likelihood >= 50.
    Returns None when no code qualifies (record skipped).
    """
    best_disease_class, best_disease_likelihood = None, -1.0
    best_norm_likelihood = -1.0
    if not isinstance(scp_codes, dict):
        return None
    for code, likelihood in scp_codes.items():
        cu = str(code).upper()
        if cu not in SCP_TO_CLASS:
            continue
        try:
            l_val = float(likelihood)
        except (ValueError, TypeError):
            l_val = 0.0
        if l_val < 50.0:
            continue
        cls = SCP_TO_CLASS[cu]
        if cls > 0:
            if l_val > best_disease_likelihood:
                best_disease_class, best_disease_likelihood = cls, l_val
        elif l_val > best_norm_likelihood:
            best_norm_likelihood = l_val
    return best_disease_class if best_disease_class is not None else \
        (0 if best_norm_likelihood >= 0 else None)


def load_ptbxl_dataset(sampling_rate=100, norm_train_cap=3000, img_size=224,
                       weight_mode='balanced'):
    """Load the FULL PTB-XL dataset, rendered with the serving renderer.

    Split (patient-disjoint, official strat_fold):
      train = folds 1-8 (Normal capped at norm_train_cap)
      val   = fold 9
      test  = fold 10

    Returns (X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
    as uint8 images, or None if the dataset is unavailable.
    """
    if not WFDB_AVAILABLE:
        print("❌ wfdb required. Run: pip install wfdb")
        return None

    metadata_path = os.path.join(PTBXL_DIR, 'ptbxl_database.csv')
    if not os.path.exists(metadata_path):
        print(f"❌ PTB-XL metadata not found: {metadata_path}")
        print("📥 Download: https://physionet.org/content/ptb-xl/1.0.3/")
        return None

    records_dir = os.path.join(PTBXL_DIR, 'records100' if sampling_rate == 100
                               else 'records500')
    if not os.path.exists(records_dir):
        print(f"❌ Records folder not found: {records_dir}")
        return None

    # Workers import ecg_render_worker (numpy + wfdb + PIL only — importing
    # TensorFlow in every spawned worker deadlocks the pool on Windows).
    sys.path.insert(0, SCRIPT_DIR)
    from ecg_render_worker import render_record

    print(f"📂 Loading PTB-XL ({sampling_rate} Hz)...")

    # Render cache — rendering 14k images takes minutes; reuse it across runs
    cache_path = os.path.join(SCRIPT_DIR,
                              f'ptbxl_render_cache_{sampling_rate}hz_{norm_train_cap}.npz')
    if os.path.exists(cache_path):
        print(f"  Loading rendered images from cache: {cache_path}")
        blob = np.load(cache_path, allow_pickle=True)
        return (blob['X_train'], blob['y_train'], blob['X_val'], blob['y_val'],
                blob['X_test'], blob['y_test'],
                {int(k): v for k, v in blob['class_weight'].item().items()})

    metadata = pd.read_csv(metadata_path, index_col='ecg_id')
    print(f"  Total records: {len(metadata)}")

    try:
        scp_parsed = metadata['scp_codes'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
    except Exception:
        scp_parsed = metadata['scp_codes'].apply(lambda x: {})

    filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'

    tasks = {'train': [], 'val': [], 'test': []}
    skipped = 0
    for ecg_id, row in metadata.iterrows():
        cls = assign_primary_class(scp_parsed.loc[ecg_id])
        fn = row.get(filename_col)
        if cls is None or not isinstance(fn, str):
            skipped += 1
            continue
        fold = int(row['strat_fold'])
        split = 'train' if fold <= 8 else ('val' if fold == 9 else 'test')
        tasks[split].append((ecg_id, os.path.join(PTBXL_DIR, fn), cls))
    print(f"  Classifiable records: {sum(len(v) for v in tasks.values())} "
          f"({skipped} skipped — no qualifying SCP code)")

    # Cap the huge Normal majority in TRAIN only (val/test stay untouched)
    rng = np.random.RandomState(SEED)
    train_norm = [t for t in tasks['train'] if t[2] == 0]
    if len(train_norm) > norm_train_cap:
        idx = rng.choice(len(train_norm), norm_train_cap, replace=False)
        train_norm = [train_norm[i] for i in idx]
    tasks['train'] = train_norm + [t for t in tasks['train'] if t[2] != 0]

    def render_split(split):
        items = tasks[split]
        print(f"  Rendering {len(items)} {split} images "
              f"({max(1, min(8, os.cpu_count() or 1))} workers)...")
        from multiprocessing import Pool
        X, y = [], []
        n_workers = max(1, min(8, os.cpu_count() or 1))
        with Pool(n_workers) as pool:
            for i, (_, img, cls) in enumerate(
                    pool.imap_unordered(render_record, items, chunksize=32)):
                if img is not None:
                    X.append(img)
                    y.append(cls)
                if (i + 1) % 2000 == 0:
                    print(f"    {i + 1}/{len(items)} rendered...")
        return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32)

    X_train, y_train = render_split('train')
    X_val, y_val = render_split('val')
    X_test, y_test = render_split('test')

    # Shuffle train (Normal-capped records were appended last)
    idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    for name, yy in [('train', y_train), ('val', y_val), ('test', y_test)]:
        print(f"  {name}: {len(yy)}  dist: {dict(sorted(Counter(yy).items()))}")

    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    if weight_mode == 'sqrt':
        # Milder compensation: chasing the 6%/5% minority classes with full
        # 'balanced' weights costs more majority accuracy than it returns.
        cw = np.sqrt(cw)
    class_weight = {int(i): float(w) for i, w in zip(np.unique(y_train), cw)}
    print(f"  Class weights: { {CLASS_NAMES[k]: round(v, 2) for k, v in class_weight.items()} }")

    try:
        np.savez_compressed(
            cache_path, X_train=X_train, y_train=y_train, X_val=X_val,
            y_val=y_val, X_test=X_test, y_test=y_test,
            class_weight=np.array([class_weight], dtype=object))
        print(f"  [OK] Render cache saved: {cache_path}")
    except Exception as e:
        print(f"  [WARN] Could not save render cache: {e}")

    return X_train, y_train, X_val, y_val, X_test, y_test, class_weight


# ══════════════════════════════════════════════════════════════════════════════
#                          TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_heart_image_model():
    """Train the ECG image classifier on the full PTB-XL dataset."""
    if not TF_AVAILABLE:
        print("❌ TensorFlow required"); return None

    print("\n" + "=" * 70)
    print("  HEART ECG MODEL — MobileNetV2 two-phase (v3, full PTB-XL)")
    print("=" * 70)

    num_classes = 5
    data = load_ptbxl_dataset(sampling_rate=100, norm_train_cap=4500,
                              weight_mode='sqrt', img_size=IMG_SIZE)
    if data is None:
        print("\n❌ Cannot train without PTB-XL — aborting (no synthetic "
              "fallback: a model trained on synthetic ECGs is useless).")
        return None
    X_train, y_train, X_val, y_val, X_test, y_test, class_weight = data

    print(f"\n🔧 Creating MobileNetV2 model (5-class ECG, "
          f"{IMG_SIZE}x{IMG_SIZE} dark-background renders)...")
    model, base_model = TU.build_model(
        num_classes=num_classes, channels=1, size=IMG_SIZE, dropout=0.4,
        augment='ecg', name='heart_ecg_mobilenetv2')
    TU.compile_model(model, 1e-3)
    model.summary()

    model = TU.train_two_phase(
        model, base_model, X_train, y_train, X_val, y_val,
        class_weight=class_weight, batch_size=32, model_path=HEART_IMAGE_MODEL_PATH,
        phase1_epochs=20, phase2_epochs=25, unfreeze='all', phase2_lr=3e-5,
        phase2_schedule='cosine', tag='ecg')

    metrics = TU.evaluate_model(model, X_test, y_test, CLASS_NAMES, tag='ecg')

    model.save(HEART_IMAGE_MODEL_PATH)
    print(f"\n[OK] Model saved: {HEART_IMAGE_MODEL_PATH}")

    config = {
        'model_path': HEART_IMAGE_MODEL_PATH,
        'model_type': 'image',
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'preprocessing': 'ECG rendered dark-background via '
                         'medidiagnose.inference_utils.signal_to_ecg_image; '
                         'photo uploads auto-inverted at serve time',
        'note': 'Model internally replicates 1ch to 3ch for MobileNetV2',
        'num_classes': num_classes,
        'classes': {str(k): v for k, v in HEART_CLASSES.items()},
        'class_names': CLASS_NAMES,
        'architecture': 'MobileNetV2_transfer_learning_v3',
        'training_notes': ('Full PTB-XL (records100), patient-disjoint '
                           'strat_fold split 1-8/9/10, class_weight balanced'),
        'accuracy': metrics['accuracy'],
        'using_real_data': True,
        'confusion_matrix': metrics['confusion_matrix']
    }
    with open(HEART_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Config saved: {HEART_CONFIG_PATH}")

    print("\n" + "=" * 70)
    print("  [WARN] REMINDERS:")
    print(f"  • Grayscale {IMG_SIZE}×{IMG_SIZE} input")
    print(f"  • 5 classes: {CLASS_NAMES}")
    print("  • Restart server.py to load the new model!")
    print("=" * 70)

    return model


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if not TF_AVAILABLE:
        print("[ERR] TensorFlow required. Install: pip install tensorflow")
    else:
        print("\n" + "=" * 70)
        print("  MediDiagnose-AI: Heart ECG Model Training (v3)")
        print("  Method: MobileNetV2 Transfer Learning, full PTB-XL")
        print("=" * 70)

        train_heart_image_model()

        print("\n✅ Done!")
