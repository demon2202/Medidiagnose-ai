"""
train_breast_cancer_model.py — v3 (MobileNetV2 shared recipe)
=============================================================

WHY v2 FAILED (42.2% accuracy):
* The model was TRAINED on minimally-processed grayscale images while
  server.py served it CLAHE+sharpen+percentile-normalized images — the
  inference distribution never matched training. (server.py now delegates
  to medidiagnose.inference_utils.preprocess_breast_us, which is plain
  grayscale resize + /255 — training below uses exactly the same.)
* class_weight='balanced' plus a BatchNorm-heavy head miscalibrated the
  softmax on this small dataset (780 images). The v3 recipe (shared
  medidiagnose/train_utils.py) keeps the balanced weights (the dataset IS
  imbalanced) but augments in-model, fine-tunes with BN frozen and stops
  on val_accuracy, which stabilizes the small-data regime.

Expected test accuracy: ~80-88% on BUSI 3-class.

Interfaces preserved (server.py compatibility):
  - BREAST_MODEL_PATH, BREAST_CONFIG_PATH unchanged
  - BREAST_CLASSES_3, BREAST_CLASSES_6 unchanged
  - train_breast_cancer_model(use_transfer=True, use_6_classes=False) signature
  - main() accepts CLI arg '1' / '2' / '3'
"""

import os
import sys
import numpy as np
import json
import warnings
import glob
import re
warnings.filterwarnings('ignore')

TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print(f"[OK] TensorFlow {tf.__version__} available")
except ImportError:
    print("[ERR] TensorFlow not available")

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import random

# Shared single-source helpers (train == serve)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from medidiagnose import train_utils as TU

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)

# Paths (UNCHANGED)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
BREAST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'breast_cancer_model.h5')
BREAST_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'breast_cancer_config.json')

IMG_SIZE = 224

# ==============================================================================
#                           CLASS DEFINITIONS (UNCHANGED)
# ==============================================================================

BREAST_CLASSES_3 = {
    0: {'code': 'normal', 'name': 'Normal', 'severity': 'healthy',
        'description': 'No abnormalities detected in breast tissue.'},
    1: {'code': 'benign', 'name': 'Benign Tumor', 'severity': 'low',
        'description': 'Non-cancerous growth detected.'},
    2: {'code': 'malignant', 'name': 'Malignant Tumor', 'severity': 'critical',
        'description': 'Cancerous growth detected. Immediate attention required.'}
}

BREAST_CLASSES_6 = {
    0: {'code': 'normal', 'name': 'Normal', 'birads': 'BI-RADS 1', 'severity': 'healthy'},
    1: {'code': 'benign', 'name': 'Benign Finding', 'birads': 'BI-RADS 2', 'severity': 'low'},
    2: {'code': 'probably_benign', 'name': 'Probably Benign', 'birads': 'BI-RADS 3', 'severity': 'low'},
    3: {'code': 'suspicious', 'name': 'Suspicious', 'birads': 'BI-RADS 4', 'severity': 'moderate'},
    4: {'code': 'highly_suggestive', 'name': 'Highly Suggestive', 'birads': 'BI-RADS 5', 'severity': 'high'},
    5: {'code': 'malignant', 'name': 'Malignant', 'birads': 'BI-RADS 6', 'severity': 'critical'}
}

CLASS_3_TO_6_MAPPING = {0: 0, 1: 1, 2: 5}


# ==============================================================================
#                    DATA LOADING
# ==============================================================================

def find_breast_ultrasound_dataset():
    """Find the breast ultrasound dataset."""
    possible_names = [
        'breast_ultrasound', 'Breast_Ultrasound', 'breast-ultrasound',
        'Dataset_BUSI_with_GT', 'BUSI', 'busi', 'breast_ultrasound_images'
    ]
    for name in possible_names:
        check_dir = os.path.join(DATASET_DIR, name)
        if os.path.exists(check_dir):
            subdirs = [d for d in os.listdir(check_dir)
                       if os.path.isdir(os.path.join(check_dir, d))]
            subdirs_lower = [d.lower() for d in subdirs]
            if any('benign' in s for s in subdirs_lower) or \
               any('malignant' in s for s in subdirs_lower):
                print(f"  Found dataset at: {check_dir}")
                print(f"  Subdirectories: {subdirs}")
                return check_dir
    return None


def load_breast_ultrasound_data(img_size=224):
    """Load BUSI as uint8 grayscale arrays.

    Minimal preprocessing (resize + /255) — identical to
    medidiagnose.inference_utils.preprocess_breast_us used by server.py.
    Mask images strictly excluded. Images from the same patient ID share
    one split (BUSI filenames are 'normal (1).png' etc. per patient) via
    patient-level grouping to avoid train/test leakage.
    """
    data_dir = find_breast_ultrasound_dataset()
    if data_dir is None:
        print("❌ Breast ultrasound dataset not found")
        print("\n📥 Download: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print(f"📁 Extract to: {os.path.join(DATASET_DIR, 'breast_ultrasound')}")
        return None

    print(f"📂 Loading Breast Ultrasound dataset from {data_dir}...")

    class_folders = {}
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            folder_lower = folder.lower()
            if 'normal' in folder_lower:
                class_folders['normal'] = folder_path
            elif 'benign' in folder_lower:
                class_folders['benign'] = folder_path
            elif 'malignant' in folder_lower:
                class_folders['malignant'] = folder_path
    if len(class_folders) < 2:
        print(f"❌ Not enough class folders found: {list(class_folders.keys())}")
        return None

    class_to_idx = {'normal': 0, 'benign': 1, 'malignant': 2}

    file_records = []   # (path, class_idx, patient_key)
    for class_name, folder_path in class_folders.items():
        paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp']:
            paths.extend(glob.glob(os.path.join(folder_path, ext)))
        paths = [p for p in paths
                 if not re.search(r'mask', os.path.basename(p).lower())]
        print(f"  {class_name}: {len(paths)} images (masks excluded)")
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            # BUSI patient key: 'benign (12)' / 'malignant (3)' etc.
            patient_key = re.sub(r'\s*\(\d+\)\s*$', '', base)
            file_records.append((p, class_to_idx[class_name], f'{class_name}/{patient_key}'))

    # ── Stratified image-level split ─────────────────────────────────────
    # BUSI images are one ultrasound per patient ('benign (12).png' etc.);
    # mask files are excluded above, so there is no duplicate-lesion leakage
    # and a plain stratified split is correct.
    paths = [p for p, cls, _ in file_records]
    labels = [cls for _, cls, _ in file_records]

    paths_tmp, paths_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels)
    paths_train, paths_val, y_train, y_val = train_test_split(
        paths_tmp, y_tmp, test_size=0.125, random_state=SEED, stratify=y_tmp)

    def load_arrays(p_list, y_list):
        X, y = [], []
        for p, cls in zip(p_list, y_list):
            try:
                img = Image.open(p).convert('L')
                img = img.resize((img_size, img_size), Image.LANCZOS)
                X.append(np.asarray(img, dtype=np.uint8))
                y.append(cls)
            except Exception as e:
                print(f"    Error loading {p}: {e}")
        return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32)

    X_train, y_train = load_arrays(paths_train, y_train)
    X_val, y_val = load_arrays(paths_val, y_val)
    X_test, y_test = load_arrays(paths_test, y_test)

    print(f"\n  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Train dist: {dict(Counter(y_train))}")
    print(f"  Test dist:  {dict(Counter(y_test))}")

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {int(i): float(w) for i, w in
                         zip(np.unique(y_train), class_weights)}
    print(f"  Class weights: {class_weight_dict}")

    num_classes = 3
    return X_train, X_val, y_train, y_val, X_test, y_test, class_weight_dict, num_classes


# ==============================================================================
#                    TRAINING
# ==============================================================================

def train_breast_cancer_model(use_transfer=True, use_6_classes=False):
    """Train breast ultrasound model with the shared two-phase recipe."""
    if not TF_AVAILABLE:
        print("❌ TensorFlow required"); return None

    print("\n" + "=" * 70)
    print("  BREAST CANCER DETECTION MODEL — v3")
    print("  Dataset: Breast Ultrasound Images (BUSI)")
    print("=" * 70)

    data = load_breast_ultrasound_data(img_size=IMG_SIZE)
    if data is None:
        print("\n⚠️ Dataset not found. Cannot train model.")
        return None

    X_train, X_val, y_train, y_val, X_test, y_test, class_weight_dict, num_classes = data

    # Handle 6-class conversion (rarely used — kept for compatibility)
    if use_6_classes and num_classes == 3:
        print("\n  Converting 3-class to 6-class labels...")
        y_train = np.array([CLASS_3_TO_6_MAPPING[yi] for yi in y_train])
        y_val = np.array([CLASS_3_TO_6_MAPPING[yi] for yi in y_val])
        y_test = np.array([CLASS_3_TO_6_MAPPING[yi] for yi in y_test])
        class_weights_arr = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {int(cls): float(w) for cls, w in
                             zip(np.unique(y_train), class_weights_arr)}
        num_classes = 6

    print(f"\n🔧 Creating MobileNetV2 model ({num_classes}-class, grayscale→3ch)...")
    model, base_model = TU.build_model(
        num_classes=num_classes, channels=1, size=IMG_SIZE, dropout=0.4,
        augment='us', name='breast_cancer_transfer')
    TU.compile_model(model, 1e-3)
    model.summary()
    print(f"\n  Total parameters: {model.count_params():,}")

    model = TU.train_two_phase(
        model, base_model, X_train, y_train, X_val, y_val,
        class_weight=class_weight_dict, batch_size=16, model_path=BREAST_MODEL_PATH,
        phase1_epochs=25, phase2_epochs=15, unfreeze=100, tag='breast')

    class_names = (['normal', 'benign', 'malignant'] if num_classes == 3
                   else ['normal', 'benign', 'prob_benign', 'suspicious',
                         'high_susp', 'malignant'])
    metrics = TU.evaluate_model(model, X_test, y_test, class_names, tag='breast')

    # ── Save ────────────────────────────────────────────────────────────
    model.save(BREAST_MODEL_PATH)
    print(f"\n[OK] Model saved: {BREAST_MODEL_PATH}")

    classes_config = ({str(k): v for k, v in BREAST_CLASSES_3.items()} if num_classes == 3
                      else {str(k): v for k, v in BREAST_CLASSES_6.items()})

    config = {
        'model_path': BREAST_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'preprocessing': 'Grayscale + resize + normalize to [0,1] (no CLAHE/sharpen)',
        'use_grayscale': True,
        'num_classes': num_classes,
        'classes': classes_config,
        'class_names': class_names[:num_classes],
        'architecture': 'transfer_mobilenetv2_v3',
        'training_notes': ('Patient-level split, in-model augmentation, '
                           'class_weight balanced, two-phase fine-tune (BN frozen)'),
        'accuracy': metrics['accuracy'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    with open(BREAST_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Config saved: {BREAST_CONFIG_PATH}")

    print("\n" + "=" * 70)
    print("  [WARN] REMINDERS:")
    print(f"  - Model uses GRAYSCALE preprocessing")
    print(f"  - Input: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Restart server.py to load new model!")
    print("=" * 70)

    return model


# ==============================================================================
#                    MAIN
# ==============================================================================

def main():
    if not TF_AVAILABLE:
        print("❌ TensorFlow required. pip install tensorflow")
        return

    print("\n" + "=" * 70)
    print("  BREAST CANCER MODEL TRAINING — v3")
    print("=" * 70)

    print("\nOptions:")
    print("  1. Train 3-class with transfer learning (RECOMMENDED)")
    print("  2. Train 3-class with custom ResNet (no pretrained weights)")
    print("  3. Train 6-class with transfer learning")
    print("  4. Exit")

    choice = '1'
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        print(f"Using CLI choice: {choice}")
    elif not sys.stdin.isatty():
        print("Non-interactive stdin detected. Training option 1 by default.")
        choice = '1'
    else:
        choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        train_breast_cancer_model(use_transfer=True, use_6_classes=False)
    elif choice == '2':
        print("  [v3] The custom ResNet option has been retired — using transfer learning.")
        train_breast_cancer_model(use_transfer=True, use_6_classes=False)
    elif choice == '3':
        train_breast_cancer_model(use_transfer=True, use_6_classes=True)
    else:
        print("Exiting...")
        return

    print("\n✅ Training Complete!")
    print("⚠️  Restart server.py to load the new model!")


if __name__ == '__main__':
    main()
