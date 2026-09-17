"""
image_classification.py — skin cancer + pneumonia training (v3)
================================================================

WHY v2 STILL FAILED (28.6% skin / 38.9% pneumonia):

* Pneumonia: class_weight='balanced' on a majority-POSITIVE dataset pushed
  every misclassification cost onto the minority Normal class, so the model
  collapsed to "always Normal" (0 TP). The class weights are REMOVED — the
  2-class softmax with a prior-matched output bias is enough.
* Skin: the pipeline fought itself (augmentation generator + class weights +
  a Lambda rescale layer that serializes badly in Keras 3). Training now uses
  the shared recipe in medidiagnose/train_utils.py: Keras Random*
  augmentation layers inside the model, Rescaling instead of Lambda,
  two-phase fine-tuning, EarlyStopping on val_accuracy.
* Both models now train on the SAME [0, 1] inputs that backend/server.py
  feeds at inference (see medidiagnose/inference_utils.py), so the
  train/serve gap that made accuracy look random is gone.

Expected test accuracy after retraining:
  - Skin cancer  (HAM10000, 7-class):  ~80%+
  - Pneumonia    (Chest X-ray, binary): ~90%+

Interfaces preserved (server.py compatibility):
  - SKIN_MODEL_PATH, PNEUMONIA_MODEL_PATH, *_CONFIG_PATH unchanged
  - HAM10000_CLASSES, CLASS_NAMES, CLASS_INFO, PNEUMONIA_CLASSES unchanged
  - preprocess_image_for_skin / preprocess_image_for_pneumonia unchanged
  - predict_skin_cancer / predict_pneumonia unchanged
  - get_demo_skin_result / get_demo_pneumonia_result unchanged
  - main() accepts CLI arg '1', '2', or '3' (skin / pneumonia / both)
"""

import os
import numpy as np
import json
import random
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── TensorFlow ──────────────────────────────────────────────────────────────
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow import keras
    TF_AVAILABLE = True
    print(f"[OK] TensorFlow {tf.__version__} available")
except ImportError:
    print("[ERR] TensorFlow not available — install with: pip install tensorflow")

from PIL import Image
import glob
from collections import Counter

# Shared single-source helpers (train == serve)
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from medidiagnose import train_utils as TU

# ── Paths (UNCHANGED — server.py compatibility) ─────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')

SKIN_MODEL_PATH = os.path.join(SCRIPT_DIR, 'skin_cancer_model.h5')
SKIN_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'skin_cancer_config.json')
PNEUMONIA_MODEL_PATH = os.path.join(SCRIPT_DIR, 'pneumonia_model.h5')
PNEUMONIA_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'pneumonia_config.json')

IMG_SIZE = 224

# ══════════════════════════════════════════════════════════════════════════════
#                         CLASS DEFINITIONS (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

HAM10000_CLASSES = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
}
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_INFO = {
    0: {'name': 'Actinic Keratoses',   'code': 'akiec', 'type': 'pre-cancerous', 'severity': 'moderate'},
    1: {'name': 'Basal Cell Carcinoma', 'code': 'bcc',   'type': 'malignant',     'severity': 'high'},
    2: {'name': 'Benign Keratosis',     'code': 'bkl',   'type': 'benign',        'severity': 'low'},
    3: {'name': 'Dermatofibroma',       'code': 'df',    'type': 'benign',        'severity': 'low'},
    4: {'name': 'Melanoma',             'code': 'mel',   'type': 'malignant',     'severity': 'critical'},
    5: {'name': 'Melanocytic Nevi',     'code': 'nv',    'type': 'benign',        'severity': 'low'},
    6: {'name': 'Vascular Lesions',     'code': 'vasc',  'type': 'benign',        'severity': 'low'}
}

PNEUMONIA_CLASSES = {
    0: {'name': 'Normal',    'code': 'normal',    'type': 'healthy', 'severity': 'healthy'},
    1: {'name': 'Pneumonia', 'code': 'pneumonia', 'type': 'disease', 'severity': 'high'}
}


# ══════════════════════════════════════════════════════════════════════════════
#                  SEVERITY / RECOMMENDATION DATA (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

SKIN_SEVERITY_DATA = {
    'critical': {
        'staging': {
            'stage': 'Potentially Advanced',
            'description': 'Melanoma detected — requires immediate professional evaluation for accurate staging',
            'prognosis': 'Early detection significantly improves outcomes. 5-year survival rate varies from 99% (Stage I) to 25% (Stage IV)'
        },
        'urgency': {'timeline': 'URGENT — Within 24-48 hours',
                    'action': 'Schedule an emergency dermatology appointment immediately.',
                    'color': 'red'},
        'treatment_options': [
            'Surgical excision with wide margins', 'Sentinel lymph node biopsy',
            'Immunotherapy (pembrolizumab, nivolumab)',
            'Targeted therapy (BRAF/MEK inhibitors if applicable)',
            'Radiation therapy for advanced cases', 'Clinical trial enrollment'
        ],
        'recommendations': {
            'level': 'critical',
            'title': '🚨 Critical — Immediate Medical Attention Required',
            'message': 'This lesion shows characteristics consistent with melanoma.',
            'actions': [
                'See a dermatologist within 24-48 hours',
                'Do NOT attempt to remove the lesion yourself',
                'Document the lesion with photos', 'Bring this analysis to your appointment'
            ],
            'next_steps': ['Dermoscopic examination', 'Skin biopsy',
                           'Staging workup if confirmed', 'Genetic testing if applicable'],
            'warning_signs': ['Asymmetry', 'Irregular borders', 'Multiple colors',
                              'Diameter > 6 mm', 'Evolution over time (ABCDE)'],
            'note': 'AI analysis is preliminary. Only a biopsy can confirm melanoma.'
        }
    },
    'high': {
        'staging': {
            'stage': 'Requires Evaluation',
            'description': 'Malignant characteristics detected — biopsy recommended',
            'prognosis': 'Generally good with early treatment.'
        },
        'urgency': {'timeline': 'Soon — Within 1-2 weeks',
                    'action': 'Schedule a dermatology appointment for biopsy.',
                    'color': 'orange'},
        'treatment_options': [
            'Surgical excision', 'Mohs micrographic surgery', 'Cryotherapy',
            'Topical treatments (imiquimod, 5-FU)', 'Photodynamic therapy'
        ],
        'recommendations': {
            'level': 'high',
            'title': '⚠️ High Priority — Professional Evaluation Needed',
            'message': 'This lesion warrants professional medical evaluation.',
            'actions': ['Schedule dermatologist within 1-2 weeks', 'Monitor for changes',
                        'Avoid sun exposure on the area', 'Take photos to track changes'],
            'next_steps': ['Dermoscopic examination', 'Possible biopsy', 'Treatment plan'],
            'note': 'Many concerning lesions turn out benign. Evaluation provides clarity.'
        }
    },
    'moderate': {
        'staging': {
            'stage': 'Pre-cancerous / Monitor',
            'description': 'Pre-cancerous changes detected — treatment recommended',
            'prognosis': 'Excellent with treatment.'
        },
        'urgency': {'timeline': 'Routine — Within 1 month',
                    'action': 'Schedule a routine dermatology check-up.',
                    'color': 'yellow'},
        'treatment_options': [
            'Cryotherapy', 'Topical medications (5-FU, imiquimod)',
            'Chemical peels', 'Photodynamic therapy', 'Regular checks every 6-12 months'
        ],
        'recommendations': {
            'level': 'moderate',
            'title': '📋 Moderate — Monitoring Recommended',
            'message': 'Pre-cancerous characteristics detected. Treatment prevents progression.',
            'actions': ['Dermatology visit within a month', 'Sun protection (SPF 30+)',
                        'Monitor for changes', 'Have all moles checked'],
            'next_steps': ['Evaluation', 'Treatment of pre-cancerous lesion',
                           'Regular screening schedule'],
            'risk_factors': ['Sun exposure', 'Fair skin', 'Multiple moles', 'Family history'],
            'note': 'Pre-cancerous lesions are very common and highly treatable.'
        }
    },
    'low': {
        'staging': {
            'stage': 'Benign',
            'description': 'Benign skin lesion — typically harmless',
            'prognosis': 'Excellent. Not dangerous.'
        },
        'urgency': {'timeline': 'Routine — Next regular check-up',
                    'action': 'Include in your next dermatology screening.',
                    'color': 'green'},
        'treatment_options': [
            'No treatment necessary', 'Cosmetic removal if desired',
            'Cryotherapy for removal', 'Regular monitoring'
        ],
        'recommendations': {
            'level': 'low',
            'title': '✅ Low Risk — Likely Benign',
            'message': 'This lesion appears benign. Monitor for changes.',
            'actions': ['Monthly skin self-examinations', 'Annual dermatology screening',
                        'Monitor with ABCDE criteria', 'Maintain sun protection'],
            'next_steps': ['Annual screening', 'Self-examination monthly',
                           'Photo documentation'],
            'note': 'Even benign lesions should be monitored. See a doctor if changes occur.'
        }
    }
}

PNEUMONIA_SEVERITY_DATA = {
    'healthy': {
        'staging': {
            'stage': 'Normal',
            'description': 'No pneumonia detected — lungs appear normal',
            'prognosis': 'No concerns identified.'
        },
        'urgency': {'timeline': 'No urgent action needed',
                    'action': 'Continue routine health maintenance.',
                    'color': 'green'},
        'treatment_options': [
            'No treatment needed', 'Annual flu vaccination',
            'Pneumonia vaccination if age-appropriate', 'Healthy lifestyle'
        ],
        'recommendations': {
            'level': 'healthy',
            'title': '✅ Normal — No Pneumonia Detected',
            'message': 'Chest X-ray analysis does not show signs of pneumonia.',
            'actions': ['Continue regular check-ups', 'Good respiratory hygiene',
                        'Stay up to date with vaccinations',
                        'Seek care if symptoms develop'],
            'note': 'If you have symptoms, consult a provider regardless of this result.'
        }
    },
    'high': {
        'staging': {
            'stage': 'Pneumonia Detected',
            'description': 'Signs consistent with pneumonia identified',
            'prognosis': 'Generally good with treatment. Most cases resolve in 1-3 weeks.'
        },
        'urgency': {'timeline': 'Urgent — Within 24 hours',
                    'action': 'See a doctor as soon as possible.',
                    'color': 'red'},
        'treatment_options': [
            'Antibiotics (bacterial)', 'Antivirals (viral)', 'Rest and hydration',
            'Fever management', 'Oxygen therapy if needed', 'Hospitalization if severe'
        ],
        'recommendations': {
            'level': 'high',
            'title': '⚠️ Pneumonia Signs Detected',
            'message': 'X-ray shows patterns consistent with pneumonia.',
            'actions': ['See a doctor within 24 hours', 'Describe all symptoms',
                        'Do not self-medicate', 'Stay hydrated and rest',
                        'Go to ER if severe breathing difficulty'],
            'next_steps': ['Physical examination', 'Blood tests / sputum culture',
                           'Professional X-ray reading', 'Prescription treatment'],
            'warning_signs': ['Severe breathing difficulty', 'Fever > 103°F',
                              'Confusion', 'Blue lips/fingertips', 'Severe chest pain'],
            'note': 'Pneumonia can be serious in elderly, children, or immunocompromised.'
        }
    }
}


# ══════════════════════════════════════════════════════════════════════════════
#                  PREPROCESSING (used by server.py at inference — UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image_for_skin(image_path_or_array, img_size=224):
    """Preprocess for skin model — RGB, [0,1], shape (1, H, W, 3).
    The model contains a built-in Rescaling layer that maps [0,1] -> [-1,1]
    for MobileNetV2. No external rescaling needed.
    """
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array)
    elif isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.max() > 1.0:
            image_path_or_array = image_path_or_array.astype(np.float32) / 255.0
        if len(image_path_or_array.shape) == 4:
            return image_path_or_array
        return np.expand_dims(image_path_or_array, axis=0)
    else:
        img = image_path_or_array
    img = img.convert('RGB')
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_image_for_pneumonia(image_path_or_array, img_size=224):
    """Preprocess for pneumonia model — Grayscale, [0,1], shape (1, H, W, 1).
    The model contains a built-in Rescaling layer that maps [0,1] -> [-1,1]
    for MobileNetV2. No external rescaling needed.
    """
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array)
    elif isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.max() > 1.0:
            image_path_or_array = image_path_or_array.astype(np.float32) / 255.0
        if len(image_path_or_array.shape) == 4:
            return image_path_or_array
        if len(image_path_or_array.shape) == 2:
            image_path_or_array = np.expand_dims(image_path_or_array, axis=-1)
        return np.expand_dims(image_path_or_array, axis=0)
    else:
        img = image_path_or_array
    img = img.convert('L')
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#                    PREDICTION FUNCTIONS (used by server.py — UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

def predict_skin_cancer(model, image_path_or_array, img_size=224):
    """Run skin cancer prediction → structured dict for frontend."""
    img_batch = preprocess_image_for_skin(image_path_or_array, img_size)
    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])
    info = CLASS_INFO[predicted_idx]
    severity = info['severity']

    all_predictions = []
    for idx in np.argsort(predictions)[::-1]:
        ci = CLASS_INFO[int(idx)]
        all_predictions.append({
            'name': ci['name'], 'code': ci['code'], 'type': ci['type'],
            'confidence': float(predictions[int(idx)]),
            'confidence_percent': f"{float(predictions[int(idx)]) * 100:.1f}%"
        })

    sev_data = SKIN_SEVERITY_DATA.get(severity, SKIN_SEVERITY_DATA['low'])
    return {
        'success': True, 'demo_mode': False,
        'prediction': {
            'name': info['name'], 'code': info['code'], 'type': info['type'],
            'confidence': confidence, 'confidence_percent': f"{confidence * 100:.1f}%",
            'class_index': predicted_idx
        },
        'severity': severity,
        'staging': sev_data['staging'], 'urgency': sev_data['urgency'],
        'treatment_options': sev_data['treatment_options'],
        'recommendations': sev_data['recommendations'],
        'all_predictions': all_predictions,
        'note': 'AI-assisted analysis. Always consult a qualified dermatologist.'
    }


def predict_pneumonia(model, image_path_or_array, img_size=224):
    """Run pneumonia prediction → structured dict for frontend.
    Handles both 2-class softmax output (this version) and legacy 1-unit sigmoid.
    """
    img_batch = preprocess_image_for_pneumonia(image_path_or_array, img_size)
    predictions = model.predict(img_batch, verbose=0)[0]

    if len(predictions) == 2:
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        normal_conf = float(predictions[0])
        pneumonia_conf = float(predictions[1])
    elif len(predictions) == 1:
        pneumonia_conf = float(predictions[0])
        normal_conf = 1.0 - pneumonia_conf
        predicted_idx = 1 if pneumonia_conf >= 0.5 else 0
        confidence = pneumonia_conf if predicted_idx == 1 else normal_conf
    else:
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        normal_conf = float(predictions[0]) if len(predictions) > 0 else 0.5
        pneumonia_conf = float(predictions[1]) if len(predictions) > 1 else 0.5

    info = PNEUMONIA_CLASSES[predicted_idx]
    severity = info['severity']

    all_predictions = [
        {'name': 'Normal', 'code': 'normal', 'type': 'healthy',
         'confidence': normal_conf, 'confidence_percent': f"{normal_conf * 100:.1f}%"},
        {'name': 'Pneumonia', 'code': 'pneumonia', 'type': 'disease',
         'confidence': pneumonia_conf, 'confidence_percent': f"{pneumonia_conf * 100:.1f}%"}
    ]
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

    sev_data = PNEUMONIA_SEVERITY_DATA.get(severity, PNEUMONIA_SEVERITY_DATA['healthy'])
    return {
        'success': True, 'demo_mode': False,
        'prediction': {
            'name': info['name'], 'code': info['code'], 'type': info['type'],
            'confidence': confidence, 'confidence_percent': f"{confidence * 100:.1f}%",
            'class_index': predicted_idx
        },
        'severity': severity,
        'staging': sev_data['staging'], 'urgency': sev_data['urgency'],
        'treatment_options': sev_data['treatment_options'],
        'recommendations': sev_data['recommendations'],
        'all_predictions': all_predictions,
        'note': 'AI-assisted analysis. Always consult a qualified physician.'
    }


# ══════════════════════════════════════════════════════════════════════════════
#                    DEMO FUNCTIONS (when no model is available — UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

def get_demo_skin_result(image_path_or_array=None):
    """Demo skin prediction when model is not trained yet."""
    demo_idx = 5
    demo_confidence = 0.65

    if image_path_or_array is not None:
        try:
            if isinstance(image_path_or_array, str):
                img = Image.open(image_path_or_array).convert('RGB')
            elif isinstance(image_path_or_array, np.ndarray):
                d = (image_path_or_array * 255).astype(np.uint8) if image_path_or_array.max() <= 1.0 else image_path_or_array.astype(np.uint8)
                if len(d.shape) == 4: d = d[0]
                img = Image.fromarray(d)
            else:
                img = image_path_or_array
            img = img.convert('RGB')
            a = np.array(img, dtype=np.float32) / 255.0
            brightness = np.mean(a)
            r_m, g_m, b_m = np.mean(a[:, :, 0]), np.mean(a[:, :, 1]), np.mean(a[:, :, 2])
            if brightness < 0.3:
                demo_idx, demo_confidence = 4, 0.45
            elif r_m > g_m * 1.3 and r_m > b_m * 1.3:
                demo_idx, demo_confidence = 6, 0.50
            elif r_m > 0.4 and g_m > 0.25 and b_m < 0.35:
                demo_idx, demo_confidence = 2, 0.55
            else:
                demo_idx, demo_confidence = 5, 0.60
        except Exception:
            pass

    info = CLASS_INFO[demo_idx]
    severity = info['severity']
    sev_data = SKIN_SEVERITY_DATA.get(severity, SKIN_SEVERITY_DATA['low'])
    fake = np.random.dirichlet(np.ones(7) * 0.3)
    fake[demo_idx] = demo_confidence
    rem = 1.0 - demo_confidence
    others = [i for i in range(7) if i != demo_idx]
    s = sum(fake[j] for j in others) + 1e-7
    for j in others:
        fake[j] = rem * (fake[j] / s)

    all_predictions = []
    for idx in np.argsort(fake)[::-1]:
        ci = CLASS_INFO[int(idx)]
        all_predictions.append({
            'name': ci['name'], 'code': ci['code'], 'type': ci['type'],
            'confidence': float(fake[int(idx)]),
            'confidence_percent': f"{float(fake[int(idx)]) * 100:.1f}%"
        })

    return {
        'success': True, 'demo_mode': True,
        'prediction': {'name': info['name'], 'code': info['code'], 'type': info['type'],
                        'confidence': demo_confidence,
                        'confidence_percent': f"{demo_confidence * 100:.1f}%",
                        'class_index': demo_idx},
        'severity': severity, 'staging': sev_data['staging'],
        'urgency': sev_data['urgency'],
        'treatment_options': sev_data['treatment_options'],
        'recommendations': sev_data['recommendations'],
        'all_predictions': all_predictions,
        'note': 'DEMO MODE: No trained model. Train for accurate predictions.'
    }


def get_demo_pneumonia_result(image_path_or_array=None):
    """Demo pneumonia prediction when model is not trained yet."""
    demo_idx, demo_confidence = 0, 0.70
    if image_path_or_array is not None:
        try:
            if isinstance(image_path_or_array, str):
                img = Image.open(image_path_or_array).convert('L')
            elif isinstance(image_path_or_array, np.ndarray):
                d = (image_path_or_array * 255).astype(np.uint8) if image_path_or_array.max() <= 1.0 else image_path_or_array.astype(np.uint8)
                if len(d.shape) == 4: d = d[0]
                if len(d.shape) == 3: d = d[:, :, 0]
                img = Image.fromarray(d, mode='L')
            else:
                img = image_path_or_array.convert('L')
            a = np.array(img, dtype=np.float32) / 255.0
            if np.mean(a > 0.7) > 0.35 or np.mean(a) > 0.55:
                demo_idx, demo_confidence = 1, 0.65
            else:
                demo_idx, demo_confidence = 0, 0.70
        except Exception:
            pass

    info = PNEUMONIA_CLASSES[demo_idx]
    severity = info['severity']
    sev_data = PNEUMONIA_SEVERITY_DATA.get(severity, PNEUMONIA_SEVERITY_DATA['healthy'])
    n_c = demo_confidence if demo_idx == 0 else 1.0 - demo_confidence
    p_c = demo_confidence if demo_idx == 1 else 1.0 - demo_confidence
    all_predictions = [
        {'name': 'Normal', 'code': 'normal', 'type': 'healthy',
         'confidence': n_c, 'confidence_percent': f"{n_c * 100:.1f}%"},
        {'name': 'Pneumonia', 'code': 'pneumonia', 'type': 'disease',
         'confidence': p_c, 'confidence_percent': f"{p_c * 100:.1f}%"}
    ]
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return {
        'success': True, 'demo_mode': True,
        'prediction': {'name': info['name'], 'code': info['code'], 'type': info['type'],
                        'confidence': demo_confidence,
                        'confidence_percent': f"{demo_confidence * 100:.1f}%",
                        'class_index': demo_idx},
        'severity': severity, 'staging': sev_data['staging'],
        'urgency': sev_data['urgency'],
        'treatment_options': sev_data['treatment_options'],
        'recommendations': sev_data['recommendations'],
        'all_predictions': all_predictions,
        'note': 'DEMO MODE: No trained model. Train for accurate predictions.'
    }


# ══════════════════════════════════════════════════════════════════════════════
#                         DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_ham10000_data(img_size=224, oversample_target=1100):
    """Load the FULL HAM10000 dataset as uint8 arrays (no class caps — the
    previous 1500/class cap threw away 2/3 of the 'nv' class for no benefit).

    The TRAIN split is oversampled (with replacement) so every class has at
    least `oversample_target` samples, and NO class_weight is used — extreme
    balanced weights (df: 12.5x) starve the majority classes. Val/test keep
    the true distribution.

    Returns (X_train, X_val, y_train, y_val, X_test, y_test, class_weight|None).
    Images stay uint8 in RAM and are converted to float32 per batch inside
    the tf.data pipeline.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    ham_dir = os.path.join(DATASET_DIR, 'HAM10000')
    metadata_path = os.path.join(ham_dir, 'HAM10000_metadata.csv')

    if not os.path.exists(metadata_path):
        print(f"❌ HAM10000 metadata not found at {metadata_path}")
        print("📥 Download: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"📁 Extract to: {ham_dir}")
        return None

    print("📂 Loading HAM10000 dataset (full, uint8)...")
    metadata = pd.read_csv(metadata_path)
    print(f"  Total entries: {len(metadata)}")

    image_dirs = []
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2',
                    'HAM10000_images', 'images', 'train', 'all_images']:
        path = os.path.join(ham_dir, folder)
        if os.path.exists(path):
            image_dirs.append(path)
    if not image_dirs:
        imgs = glob.glob(os.path.join(ham_dir, '*.jpg')) + glob.glob(os.path.join(ham_dir, '*.png'))
        if imgs:
            image_dirs.append(ham_dir)
        else:
            print("❌ No image directories found"); return None

    image_paths = {}
    for d in image_dirs:
        for p in glob.glob(os.path.join(d, '*.jpg')) + glob.glob(os.path.join(d, '*.jpeg')) + glob.glob(os.path.join(d, '*.png')):
            image_paths[os.path.splitext(os.path.basename(p))[0]] = p
    print(f"  Found {len(image_paths)} images")

    X, y = [], []
    skipped = 0
    for _, row in metadata.iterrows():
        p = image_paths.get(row['image_id'])
        if p is None or row['dx'] not in HAM10000_CLASSES:
            skipped += 1
            continue
        try:
            img = Image.open(p).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
            X.append(np.asarray(img, dtype=np.uint8))
            y.append(HAM10000_CLASSES[row['dx']])
        except Exception:
            skipped += 1
        if len(X) % 2000 == 0 and len(X) > 0:
            print(f"    loaded {len(X)}...")

    X = np.array(X); y = np.array(y, dtype=np.int32)
    print(f"  ✓ Loaded {len(X)} images ({skipped} skipped), shape {X.shape}")
    print(f"  Class dist: {dict(Counter(y))}")

    # 10% val for checkpoint selection, 20% test — both stratified.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.125, random_state=SEED, stratify=y_tmp)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Oversample minority classes in the TRAIN split only (val/test keep the
    # real distribution). Repeats are fine: in-model augmentation makes each
    # repeat a different variation.
    if oversample_target:
        rng = np.random.RandomState(SEED)
        balanced_idx = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            n = len(cls_idx)
            if n >= oversample_target:
                balanced_idx.extend(cls_idx.tolist())
            else:
                extra = rng.choice(cls_idx, oversample_target - n, replace=True)
                balanced_idx.extend(cls_idx.tolist())
                balanced_idx.extend(extra.tolist())
        rng.shuffle(balanced_idx)
        X_train = X_train[np.array(balanced_idx)]
        y_train = y_train[np.array(balanced_idx)]
        print(f"  After oversampling to >= {oversample_target}/class: "
              f"{dict(sorted(Counter(y_train).items()))}")

    class_weight = None   # oversampling replaces class weighting
    return X_train, X_val, y_train, y_val, X_test, y_test, class_weight


PAD20_CLASSES = {
    # PAD-UFES-20 diagnostic -> our 7-class scheme (server.py CLASS_INFO)
    'ACK': 'akiec',   # actinic keratosis
    'SCC': 'akiec',   # squamous cell carcinoma (~intraepithelial carcinoma)
    'BCC': 'bcc',
    'SEK': 'bkl',     # seborrheic keratosis (benign keratosis-like)
    'NEV': 'nv',
    'MEL': 'mel',
}


def load_skin_combined_data(img_size=224, nv_train_cap=2500,
                            oversample_target=1200):
    """Load HAM10000 + PAD-UFES-20 (Dataset/Skin_Cancer) as one training set.

    PAD-UFES-20 is smartphone (non-dermoscopy) photography — the same kind of
    images users upload to the web app — and it adds ~1.7k real samples to the
    weakest classes (akiec, bcc). Its diagnostics map onto the existing
    7-class scheme via PAD20_CLASSES, so server.py and the frontend need no
    changes.

    Split is GROUP-disjoint (HAM: lesion_id, PAD: patient_id) via
    StratifiedGroupKFold — no lesion/patient appears in two splits.
    Train-only rebalancing: cap 'nv' at nv_train_cap, oversample classes
    below oversample_target. Val/test keep the real distribution.

    Returns (X_train, X_val, y_train, y_val, X_test, y_test, None).
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    records = []   # (path, class_code, group_key)

    # ── HAM10000 ────────────────────────────────────────────────────────
    ham_dir = os.path.join(DATASET_DIR, 'HAM10000')
    meta_path = os.path.join(ham_dir, 'HAM10000_metadata.csv')
    if os.path.exists(meta_path):
        image_paths = {}
        for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2',
                        'HAM10000_images', 'images']:
            d = os.path.join(ham_dir, folder)
            if os.path.isdir(d):
                for p in glob.glob(os.path.join(d, '*.jpg')):
                    image_paths[os.path.splitext(os.path.basename(p))[0]] = p
        ham_meta = pd.read_csv(meta_path)
        n = 0
        for _, row in ham_meta.iterrows():
            p = image_paths.get(row['image_id'])
            if p is not None and row['dx'] in HAM10000_CLASSES:
                records.append((p, row['dx'], f'ham/{row["lesion_id"]}'))
                n += 1
        print(f"  HAM10000: {n} images")

    # ── PAD-UFES-20 ─────────────────────────────────────────────────────
    pad_dir = os.path.join(DATASET_DIR, 'Skin_Cancer')
    pad_meta_path = os.path.join(pad_dir, 'metadata.csv')
    if os.path.exists(pad_meta_path):
        pad_paths = {}
        for root, _dirs, files in os.walk(pad_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    pad_paths[os.path.splitext(f)[0]] = os.path.join(root, f)
        pad_meta = pd.read_csv(pad_meta_path)
        n = 0
        for _, row in pad_meta.iterrows():
            code = PAD20_CLASSES.get(str(row['diagnostic']).upper())
            # img_id includes the file extension ('PAT_..._530.png')
            p = pad_paths.get(str(row['img_id']).rsplit('.', 1)[0])
            if code and p is not None:
                records.append((p, code, f'pad/{row["patient_id"]}'))
                n += 1
        print(f"  PAD-UFES-20: {n} images")

    if not records:
        print("❌ No skin images found (HAM10000 and PAD-UFES-20 both missing)")
        return None

    paths = np.array([r[0] for r in records])
    codes = np.array([r[1] for r in records])
    groups = np.array([r[2] for r in records])
    labels = np.array([HAM10000_CLASSES[c] for c in codes])
    print(f"  Combined: {len(paths)} images, "
          f"{dict(sorted(Counter(labels).items()))}")

    # ── Group-disjoint stratified split (folds: 0=test, 1=val) ──────────
    sgkf = StratifiedGroupKFold(n_splits=12, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(paths, labels, groups))
    test_idx, val_idx, train_idx = folds[0][1], folds[1][1], None
    val_set, test_set = set(val_idx), set(test_idx)
    train_idx = np.array([i for i in range(len(paths))
                          if i not in val_set and i not in test_set])

    def load_split(idx):
        X, y = [], []
        for i in idx:
            try:
                img = Image.open(paths[i]).convert('RGB')
                img = img.resize((img_size, img_size), Image.LANCZOS)
                X.append(np.asarray(img, dtype=np.uint8))
                y.append(labels[i])
            except Exception:
                continue
        return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32)

    X_train, y_train = load_split(train_idx)
    X_val, y_val = load_split(val_idx)
    X_test, y_test = load_split(test_idx)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Test dist: {dict(sorted(Counter(y_test).items()))}")

    # ── Train-only rebalancing: cap nv, oversample rare classes ─────────
    rng = np.random.RandomState(SEED)
    keep = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        if cls == HAM10000_CLASSES['nv'] and len(cls_idx) > nv_train_cap:
            cls_idx = rng.choice(cls_idx, nv_train_cap, replace=False)
        keep.extend(cls_idx.tolist())
    y_train_r = y_train[np.array(keep)]
    X_train_r = X_train[np.array(keep)]

    balanced_idx = []
    for cls in np.unique(y_train_r):
        cls_idx = np.where(y_train_r == cls)[0]
        n = len(cls_idx)
        balanced_idx.extend(cls_idx.tolist())
        if n < oversample_target:
            extra = rng.choice(cls_idx, oversample_target - n, replace=True)
            balanced_idx.extend(extra.tolist())
    rng.shuffle(balanced_idx)
    X_train = X_train_r[np.array(balanced_idx)]
    y_train = y_train_r[np.array(balanced_idx)]
    print(f"  Train after rebalance: {dict(sorted(Counter(y_train).items()))}")

    return X_train, X_val, y_train, y_val, X_test, y_test, None


def load_chest_xray_data(img_size=224):
    """Load Chest X-Ray Pneumonia dataset as uint8 grayscale arrays.

    IMPORTANT: plain resize + /255 — NO CLAHE. server.py serves the model
    with medidiagnose.inference_utils.preprocess_xray (plain conversion), so
    training must see exactly the same pixel statistics.
    """
    from sklearn.utils.class_weight import compute_class_weight

    xray_dir = os.path.join(DATASET_DIR, 'chest_xray')
    train_dir = os.path.join(xray_dir, 'train')
    test_dir = os.path.join(xray_dir, 'test')
    val_dir = os.path.join(xray_dir, 'val')

    if not os.path.exists(train_dir):
        print(f"❌ Chest X-Ray dataset not found at {train_dir}")
        print("📥 Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        return None

    print("📂 Loading Chest X-Ray dataset (plain resize, NO CLAHE)...")
    np.random.seed(SEED)

    def load_folder(folder, label):
        images, labels = [], []
        files = [f for f in glob.glob(os.path.join(folder, '*'))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for path in files:
            try:
                img = Image.open(path).convert('L')
                img = img.resize((img_size, img_size), Image.LANCZOS)
                images.append(np.asarray(img, dtype=np.uint8))
                labels.append(label)
            except Exception:
                continue
        return images, labels

    print("  Loading NORMAL training images...")
    X_n, y_n = load_folder(os.path.join(train_dir, 'NORMAL'), 0)
    if os.path.exists(os.path.join(val_dir, 'NORMAL')):
        x2, y2 = load_folder(os.path.join(val_dir, 'NORMAL'), 0)
        X_n += x2; y_n += y2
    print(f"    NORMAL: {len(X_n)}")

    print("  Loading PNEUMONIA training images...")
    X_p, y_p = load_folder(os.path.join(train_dir, 'PNEUMONIA'), 1)
    if os.path.exists(os.path.join(val_dir, 'PNEUMONIA')):
        x2, y2 = load_folder(os.path.join(val_dir, 'PNEUMONIA'), 1)
        X_p += x2; y_p += y2
    print(f"    PNEUMONIA: {len(X_p)}")

    X_full = np.array(X_n + X_p, dtype=np.uint8)
    y_full = np.array(y_n + y_p, dtype=np.int32)
    idx = np.random.permutation(len(X_full))
    X_full, y_full = X_full[idx], y_full[idx]
    print(f"  Total: {len(X_full)}  Distribution: {dict(Counter(y_full))}")

    # NO class weights here. Pneumonia is the MAJORITY class; 'balanced'
    # weights would over-penalize Normal errors and collapse the model to
    # predicting Normal everywhere (the exact failure of the last version).
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_full)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    X_tn, y_tn = load_folder(os.path.join(test_dir, 'NORMAL'), 0)
    X_tp, y_tp = load_folder(os.path.join(test_dir, 'PNEUMONIA'), 1)
    X_test = np.array(X_tn + X_tp, dtype=np.uint8)
    y_test = np.array(y_tn + y_tp, dtype=np.int32)
    idx = np.random.permutation(len(X_test))
    X_test, y_test = X_test[idx], y_test[idx]
    print(f"  Test: {len(X_test)}  Distribution: {dict(Counter(y_test))}")

    # 10% stratified val split out of training data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=SEED, stratify=y_full)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    return X_train, X_val, y_train, y_val, X_test, y_test, class_weight


# ══════════════════════════════════════════════════════════════════════════════
#                         TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_skin_cancer_model():
    """
    Train skin cancer detection: MobileNetV2 transfer learning via the shared
    two-phase recipe (see medidiagnose/train_utils.py).

    Key changes vs v2:
      - HAM10000 + PAD-UFES-20 combined (group-disjoint split by
        lesion_id/patient_id), mapped onto the same 7 classes
      - Train-only rebalancing: nv capped, rare classes oversampled,
        NO class_weight
      - Augmentation as in-model Random* layers (flip/rotate/zoom/shift/contrast)
      - Rescaling layer instead of Lambda (clean .h5 serialization)
      - Full-backbone fine-tune (BN frozen) with cosine LR
    Expected test accuracy: ~80% (7-class, group-disjoint split).
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow required"); return None

    print("\n" + "=" * 70)
    print("  SKIN CANCER MODEL — MobileNetV2 two-phase (v4: HAM + PAD-UFES-20)")
    print("=" * 70)

    data = load_skin_combined_data(IMG_SIZE)
    if data is None:
        return None
    X_train, X_val, y_train, y_val, X_test, y_test, class_weight = data

    print("\n🔧 Creating MobileNetV2 model (7-class skin cancer)...")
    model, base_model = TU.build_model(
        num_classes=7, channels=3, size=IMG_SIZE, dropout=0.3,
        augment='skin', name='skin_mobilenetv2')
    TU.compile_model(model, 1e-3)
    model.summary()

    model = TU.train_two_phase(
        model, base_model, X_train, y_train, X_val, y_val,
        class_weight=class_weight, batch_size=32, model_path=SKIN_MODEL_PATH,
        phase1_epochs=20, phase2_epochs=20, unfreeze='all', phase2_lr=3e-5,
        phase2_schedule='cosine', tag='skin')

    metrics = TU.evaluate_model(model, X_test, y_test, CLASS_NAMES, tag='skin')

    model.save(SKIN_MODEL_PATH)
    print(f"\n[OK] Model saved: {SKIN_MODEL_PATH}")

    config = {
        'model_path': SKIN_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'preprocessing': 'RGB, normalize to [0,1]',
        'num_classes': 7,
        'class_names': CLASS_NAMES,
        'class_mapping': HAM10000_CLASSES,
        'architecture': 'MobileNetV2_transfer_learning_v4',
        'training_notes': ('HAM10000 + PAD-UFES-20 (group-disjoint split), '
                           'in-model Random* augmentation, train-only '
                           'rebalancing, full-backbone fine-tune, cosine LR'),
        'accuracy': metrics['accuracy'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    with open(SKIN_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Config saved: {SKIN_CONFIG_PATH}")
    print("\n  [WARN] Restart server.py to load the new model!")
    return model


def train_pneumonia_model():
    """
    Train pneumonia detection: MobileNetV2 transfer learning via the shared
    two-phase recipe.

    Key changes vs v2:
      - NO class_weight (majority-positive dataset — weights caused the
        all-Normal collapse with 0 TP)
      - NO CLAHE (server.py inference uses plain resize+/255 — train==serve)
      - Output bias initialized to the class prior (Pneumonia is majority)
      - 2-class softmax, standard cross-entropy
    Expected test accuracy: ~90%+ on the official chest_xray test split.
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow required"); return None

    print("\n" + "=" * 70)
    print("  PNEUMONIA MODEL — MobileNetV2 two-phase (v3)")
    print("  Fix: no class_weight, no CLAHE, prior-matched output bias")
    print("=" * 70)

    data = load_chest_xray_data(IMG_SIZE)
    if data is None:
        return None
    X_train, X_val, y_train, y_val, X_test, y_test, _class_weight = data

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(len(y_train) - n_pos)
    p_neg = max(n_neg / (n_pos + n_neg), 1e-7)
    p_pos = max(n_pos / (n_pos + n_neg), 1e-7)
    output_bias = [np.log(p_neg), np.log(p_pos)]
    print(f"  Output bias (class log-priors): neg={output_bias[0]:.3f}, "
          f"pos={output_bias[1]:.3f}  (pos={n_pos}, neg={n_neg})")

    print("\n🔧 Creating MobileNetV2 model (2-class softmax)...")
    model, base_model = TU.build_model(
        num_classes=2, channels=1, size=IMG_SIZE, dropout=0.3,
        augment='xray', name='pneumonia_mobilenetv2')

    # Set the output layer's bias to the class log-priors
    out_layer = model.get_layer('pneumonia_mobilenetv2_output')
    weights = out_layer.get_weights()
    out_layer.set_weights([weights[0], np.array(output_bias, dtype=np.float32)])

    TU.compile_model(model, 1e-3)
    model.summary()

    model = TU.train_two_phase(
        model, base_model, X_train, y_train, X_val, y_val,
        class_weight=None, batch_size=32, model_path=PNEUMONIA_MODEL_PATH,
        phase1_epochs=15, phase2_epochs=10, unfreeze=60, tag='pneumonia')

    metrics = TU.evaluate_model(model, X_test, y_test, ['Normal', 'Pneumonia'],
                                tag='pneumonia')

    model.save(PNEUMONIA_MODEL_PATH)
    print(f"\n[OK] Model saved: {PNEUMONIA_MODEL_PATH}")

    cm = np.array(metrics['confusion_matrix'])
    config = {
        'model_path': PNEUMONIA_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'preprocessing': 'Grayscale, plain resize, normalize to [0,1] (no CLAHE)',
        'note': 'Model internally replicates 1ch to 3ch for MobileNetV2',
        'num_classes': 2,
        'class_names': ['NORMAL', 'PNEUMONIA'],
        'output_type': 'softmax_2class',
        'architecture': 'MobileNetV2_transfer_learning_v3',
        'optimal_threshold': 0.5,   # 2-class softmax argmax — no tuning needed
        'accuracy': metrics['accuracy'],
        'confusion_matrix': {
            'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]), 'TP': int(cm[1, 1])
        }
    }
    with open(PNEUMONIA_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Config saved: {PNEUMONIA_CONFIG_PATH}")
    print("\n  [WARN] Restart server.py to load the new model!")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not TF_AVAILABLE:
        print("❌ TensorFlow is required. Install: pip install tensorflow")
        return

    print("\n" + "=" * 70)
    print("  MediDiagnose-AI: Image Classification Training (v3)")
    print("  Method: MobileNetV2 Transfer Learning, shared recipe")
    print("=" * 70)

    os.makedirs(DATASET_DIR, exist_ok=True)

    print("\nSelect model to train:")
    print("  1. Skin Cancer  (HAM10000 dataset)")
    print("  2. Pneumonia    (Chest X-Ray dataset)")
    print("  3. Both models")
    print("  4. Exit")

    import sys
    choice = '3'
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        print(f"Using CLI choice: {choice}")
    elif not sys.stdin.isatty():
        print("Non-interactive stdin detected. Training both models by default.")
        choice = '3'
    else:
        choice = input("\nChoice (1-4): ").strip()

    if choice == '1':
        train_skin_cancer_model()
    elif choice == '2':
        train_pneumonia_model()
    elif choice == '3':
        train_skin_cancer_model()
        train_pneumonia_model()
    else:
        print("Exiting...")
        return

    print("\n✅ Done!")
    print("⚠️  Restart server.py to load the new models!")


if __name__ == '__main__':
    main()
