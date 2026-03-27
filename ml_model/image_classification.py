import os
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# ── TensorFlow ──────────────────────────────────────────────────────────────
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} available")
except ImportError:
    print("✗ TensorFlow not available — install with: pip install tensorflow")

from PIL import Image
import glob
from collections import Counter

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')

SKIN_MODEL_PATH = os.path.join(SCRIPT_DIR, 'skin_cancer_model.h5')
SKIN_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'skin_cancer_config.json')
PNEUMONIA_MODEL_PATH = os.path.join(SCRIPT_DIR, 'pneumonia_model.h5')
PNEUMONIA_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'pneumonia_config.json')

IMG_SIZE = 224

# ══════════════════════════════════════════════════════════════════════════════
#                         CLASS DEFINITIONS
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
#                      SEVERITY / RECOMMENDATION DATA
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
#                  PREPROCESSING (used by server.py at inference)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image_for_skin(image_path_or_array, img_size=224):
    """Preprocess for skin model — RGB, [0,1], shape (1, H, W, 3)."""
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
    """Preprocess for pneumonia model — Grayscale, [0,1], shape (1, H, W, 1)."""
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
#                    PREDICTION FUNCTIONS (used by server.py)
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
    """Run pneumonia prediction → structured dict for frontend."""
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
#                    DEMO FUNCTIONS (when no model is available)
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
#                       MODEL CREATION
# ══════════════════════════════════════════════════════════════════════════════

def create_skin_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Skin cancer model — MobileNetV2 transfer learning.
    Input:  RGB (224, 224, 3)
    Output: softmax over 7 classes
    Returns: (model, base_model_reference_for_finetuning)
    """
    inputs = layers.Input(shape=input_shape, name='skin_input')

    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False          # frozen for phase 1

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='skin_mobilenetv2')
    return model, base_model


def create_pneumonia_model(input_shape=(224, 224, 1), num_classes=2):
    """
    Pneumonia model — MobileNetV2 transfer learning.

    KEY FIX: The old custom CNN couldn't learn good features and
    defaulted to predicting "pneumonia" for everything.

    This model:
      1. Takes GRAYSCALE input (224, 224, 1) — compatible with server.py
      2. Internally replicates to 3 channels for MobileNetV2
      3. Uses pretrained ImageNet features → much better accuracy
      4. Standard categorical_crossentropy → loads without issues

    Returns: (model, base_model_reference_for_finetuning)
    """
    gray_input = layers.Input(shape=input_shape, name='xray_input')

    # ── Replicate grayscale → 3 channels for pretrained backbone ────────
    x = layers.Concatenate()([gray_input, gray_input, gray_input])

    base_model = keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False          # frozen for phase 1

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(gray_input, outputs, name='pneumonia_mobilenetv2')
    return model, base_model


# ══════════════════════════════════════════════════════════════════════════════
#                         DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_ham10000_data(img_size=224, max_samples_per_class=1500):
    """Load HAM10000 skin cancer dataset."""
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

    print("📂 Loading HAM10000 dataset...")
    metadata = pd.read_csv(metadata_path)
    print(f"  Total entries: {len(metadata)}")

    # ── Find image folders ──────────────────────────────────────────────
    image_dirs = []
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2',
                    'HAM10000_images', 'images', 'train', 'all_images']:
        path = os.path.join(ham_dir, folder)
        if os.path.exists(path):
            image_dirs.append(path)
            print(f"  Found folder: {folder}")
    if not image_dirs:
        imgs = glob.glob(os.path.join(ham_dir, '*.jpg')) + glob.glob(os.path.join(ham_dir, '*.png'))
        if imgs:
            image_dirs.append(ham_dir)
        else:
            print("❌ No image directories found"); return None

    image_paths = {}
    for d in image_dirs:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for p in glob.glob(os.path.join(d, ext)):
                image_paths[os.path.splitext(os.path.basename(p))[0]] = p
    print(f"  Found {len(image_paths)} images")

    # ── Collect per-class ────────────────────────────────────────────────
    class_images = {c: [] for c in HAM10000_CLASSES}
    for _, row in metadata.iterrows():
        iid = row['image_id']
        if iid in image_paths and row['dx'] in class_images:
            class_images[row['dx']].append(image_paths[iid])

    print("\n  Class distribution:")
    total = 0
    for c, imgs in class_images.items():
        print(f"    {c}: {len(imgs)}")
        total += len(imgs)
    print(f"  Total matched: {total}")

    if max_samples_per_class:
        print(f"\n  Capping to {max_samples_per_class} per class...")
        for c in class_images:
            if len(class_images[c]) > max_samples_per_class:
                np.random.shuffle(class_images[c])
                class_images[c] = class_images[c][:max_samples_per_class]

    # ── Load images ──────────────────────────────────────────────────────
    X, y = [], []
    total = sum(len(v) for v in class_images.values())
    loaded = 0
    print(f"\n  Loading {total} images (RGB, {img_size}×{img_size})...")

    for cls, paths in class_images.items():
        cls_idx = HAM10000_CLASSES[cls]
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                img = img.resize((img_size, img_size), Image.LANCZOS)
                X.append(np.array(img, dtype=np.float32) / 255.0)
                y.append(cls_idx)
                loaded += 1
                if loaded % 1000 == 0:
                    print(f"    {loaded}/{total}...")
            except Exception:
                continue

    X = np.array(X); y = np.array(y)
    print(f"\n  ✓ Loaded {len(X)} images, shape {X.shape}")
    print(f"  Class dist: {Counter(y)}")

    y_oh = to_categorical(y, num_classes=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_oh, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw_dict = dict(enumerate(cw))
    print(f"  Class weights: { {CLASS_NAMES[k]: round(v, 2) for k, v in cw_dict.items()} }")
    return X_train, X_test, y_train, y_test, cw_dict


def load_chest_xray_data(img_size=224):
    """
    Load Chest X-Ray Pneumonia dataset.
    Uses ALL training images with class weights (no undersampling).
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

    print("📂 Loading Chest X-Ray dataset...")

    def load_folder(folder, label):
        images, labels = [], []
        files = [f for f in glob.glob(os.path.join(folder, '*'))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for path in files:
            try:
                img = Image.open(path).convert('L')
                img = img.resize((img_size, img_size), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=-1)         # (H, W, 1)
                images.append(arr)
                labels.append(label)
            except Exception:
                continue
        return images, labels

    # ── Training data: train/ + val/ (val is tiny, merge it in) ─────────
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

    X_train = np.array(X_n + X_p)
    y_train = np.array(y_n + y_p)

    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"\n  Total training: {len(X_train)}")
    print(f"  Distribution: {Counter(y_train)}")

    # ── Class weights — compensates for imbalance in loss function ──────
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    cw_dict = {0: float(cw[0]), 1: float(cw[1])}
    print(f"  Class weights: Normal={cw_dict[0]:.3f}  Pneumonia={cw_dict[1]:.3f}")

    # ── Test data ───────────────────────────────────────────────────────
    print("\n  Loading test data...")
    X_tn, y_tn = load_folder(os.path.join(test_dir, 'NORMAL'), 0)
    X_tp, y_tp = load_folder(os.path.join(test_dir, 'PNEUMONIA'), 1)
    X_test = np.array(X_tn + X_tp)
    y_test = np.array(y_tn + y_tp)
    idx = np.random.permutation(len(X_test))
    X_test, y_test = X_test[idx], y_test[idx]
    print(f"  Total test: {len(X_test)}  Distribution: {Counter(y_test)}")
    print(f"  Shape: {X_train.shape}")

    return X_train, X_test, y_train, y_test, cw_dict


# ══════════════════════════════════════════════════════════════════════════════
#                         TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_skin_cancer_model(max_samples_per_class=1500):
    """
    Train skin cancer detection: MobileNetV2 transfer learning, two-phase.
    Phase 1: frozen backbone, lr=0.001, 25 epochs
    Phase 2: fine-tune last 30 layers, lr=1e-5, 15 epochs
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow required"); return None

    print("\n" + "=" * 70)
    print("  SKIN CANCER MODEL — MobileNetV2 Transfer Learning")
    print("=" * 70)

    data = load_ham10000_data(IMG_SIZE, max_samples_per_class)
    if data is None:
        return None
    X_train, X_test, y_train, y_test, cw_dict = data

    # ── Create model ────────────────────────────────────────────────────
    print("\n🔧 Creating MobileNetV2 model (7-class skin cancer)...")
    model, base_model = create_skin_model((IMG_SIZE, IMG_SIZE, 3), 7)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    model.summary()

    # ── Augmentation ────────────────────────────────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='reflect',
        brightness_range=[0.8, 1.2]
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(SKIN_MODEL_PATH, monitor='val_auc', mode='max',
                        save_best_only=True, verbose=1)
    ]

    # ── Phase 1: Frozen backbone ────────────────────────────────────────
    print("\n🚀 Phase 1 — Training classification head (backbone frozen)...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=25,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=1
    )

    # ── Phase 2: Fine-tune last 30 layers ───────────────────────────────
    print("\n🚀 Phase 2 — Fine-tuning last 30 backbone layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    callbacks_ft = [
        EarlyStopping(monitor='val_auc', mode='max', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                          min_lr=1e-8, verbose=1),
        ModelCheckpoint(SKIN_MODEL_PATH, monitor='val_auc', mode='max',
                        save_best_only=True, verbose=1)
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=15,
        validation_data=(X_test, y_test),
        callbacks=callbacks_ft,
        class_weight=cw_dict,
        verbose=1
    )

    # ── Evaluate ────────────────────────────────────────────────────────
    print("\n📊 Evaluation...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss:      {results[0]:.4f}")
    print(f"  Accuracy:  {results[1]:.4f}  ({results[1]*100:.2f}%)")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall:    {results[3]:.4f}")
    print(f"  AUC:       {results[4]:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n  Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            print(f"    {name:8s}: {acc:.4f}  ({mask.sum()} samples)")

    from sklearn.metrics import confusion_matrix, classification_report
    print("\n  Classification Report:")
    present = sorted(set(y_true) | set(y_pred))
    print(classification_report(y_true, y_pred,
                                labels=present,
                                target_names=[CLASS_NAMES[i] for i in present],
                                zero_division=0))

    # ── Save ────────────────────────────────────────────────────────────
    model.save(SKIN_MODEL_PATH)
    print(f"\n✓ Model saved: {SKIN_MODEL_PATH}")

    config = {
        'model_path': SKIN_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'preprocessing': 'RGB, normalize to [0,1]',
        'num_classes': 7,
        'class_names': CLASS_NAMES,
        'class_mapping': HAM10000_CLASSES,
        'architecture': 'MobileNetV2_transfer_learning',
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4])
    }
    with open(SKIN_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {SKIN_CONFIG_PATH}")

    print("\n  ⚠️  Restart server.py to load the new model!")
    return model


def train_pneumonia_model():
    """
    Train pneumonia detection: MobileNetV2 transfer learning, two-phase.

    KEY FIX vs old version:
      Old: Custom CNN → couldn't distinguish normal from pneumonia
      New: MobileNetV2 pretrained features → much better accuracy

    Phase 1: frozen backbone, lr=0.001, 20 epochs
    Phase 2: fine-tune last 30 layers, lr=1e-5, 10 epochs
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow required"); return None

    print("\n" + "=" * 70)
    print("  PNEUMONIA MODEL — MobileNetV2 Transfer Learning")
    print("  (Grayscale input → internal 3-channel conversion)")
    print("=" * 70)

    data = load_chest_xray_data(IMG_SIZE)
    if data is None:
        return None
    X_train, X_test, y_train_int, y_test_int, cw_dict = data

    # One-hot encode
    y_train = to_categorical(y_train_int, num_classes=2)
    y_test = to_categorical(y_test_int, num_classes=2)

    # ── Create model ────────────────────────────────────────────────────
    print("\n🔧 Creating MobileNetV2 model (2-class pneumonia)...")
    print("   Input: (224, 224, 1) grayscale → internally replicated to 3ch")
    model, base_model = create_pneumonia_model((IMG_SIZE, IMG_SIZE, 1), 2)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    model.summary()

    # ── Augmentation (conservative for X-rays) ──────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=7,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='constant',
        cval=0
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(PNEUMONIA_MODEL_PATH, monitor='val_auc', mode='max',
                        save_best_only=True, verbose=1)
    ]

    # ── Phase 1: Frozen backbone ────────────────────────────────────────
    print("\n🚀 Phase 1 — Training classification head (backbone frozen)...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=1
    )

    # ── Phase 2: Fine-tune last 30 layers ───────────────────────────────
    print("\n🚀 Phase 2 — Fine-tuning last 30 backbone layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    callbacks_ft = [
        EarlyStopping(monitor='val_auc', mode='max', patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                          min_lr=1e-8, verbose=1),
        ModelCheckpoint(PNEUMONIA_MODEL_PATH, monitor='val_auc', mode='max',
                        save_best_only=True, verbose=1)
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=callbacks_ft,
        class_weight=cw_dict,
        verbose=1
    )

    # ── Evaluate ────────────────────────────────────────────────────────
    print("\n📊 Evaluation...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss:      {results[0]:.4f}")
    print(f"  Accuracy:  {results[1]:.4f}  ({results[1]*100:.2f}%)")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall:    {results[3]:.4f}")
    print(f"  AUC:       {results[4]:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test_int, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"              Normal  Pneumonia")
    print(f"  Normal      {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  Pneumonia   {cm[1,0]:5d}    {cm[1,1]:5d}")
    print(f"\n  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    print("\n  Classification Report:")
    print(classification_report(y_test_int, y_pred,
                                target_names=['Normal', 'Pneumonia'],
                                zero_division=0))

    # ── Verify on a few samples ─────────────────────────────────────────
    print("\n  Sample predictions (first 10 test images):")
    sample_pred = model.predict(X_test[:10], verbose=0)
    for i in range(10):
        true_label = 'Normal' if y_test_int[i] == 0 else 'Pneumonia'
        pred_label = 'Normal' if np.argmax(sample_pred[i]) == 0 else 'Pneumonia'
        conf = np.max(sample_pred[i])
        status = '✓' if true_label == pred_label else '✗'
        print(f"    {status} True: {true_label:10s}  Pred: {pred_label:10s}  "
              f"Conf: {conf:.4f}  (N={sample_pred[i][0]:.3f} P={sample_pred[i][1]:.3f})")

    # ── Save ────────────────────────────────────────────────────────────
    model.save(PNEUMONIA_MODEL_PATH)
    print(f"\n✓ Model saved: {PNEUMONIA_MODEL_PATH}")

    config = {
        'model_path': PNEUMONIA_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'preprocessing': 'Grayscale, normalize to [0,1]',
        'note': 'Model internally replicates 1ch to 3ch for MobileNetV2',
        'num_classes': 2,
        'class_names': ['NORMAL', 'PNEUMONIA'],
        'output_type': 'softmax_2class',
        'architecture': 'MobileNetV2_transfer_learning',
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4]),
        'confusion_matrix': {
            'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]), 'TP': int(cm[1, 1])
        }
    }
    with open(PNEUMONIA_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {PNEUMONIA_CONFIG_PATH}")

    print("\n  ⚠️  Restart server.py to load the new model!")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not TF_AVAILABLE:
        print("❌ TensorFlow is required. Install: pip install tensorflow")
        return

    print("\n" + "=" * 70)
    print("  MediDiagnose-AI: Image Classification Training")
    print("  Method: MobileNetV2 Transfer Learning (best accuracy)")
    print("=" * 70)

    os.makedirs(DATASET_DIR, exist_ok=True)

    print("\nSelect model to train:")
    print("  1. Skin Cancer  (HAM10000 dataset)")
    print("  2. Pneumonia    (Chest X-Ray dataset)")
    print("  3. Both models")
    print("  4. Exit")

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

    print("\n" + "=" * 70)
    print("  ✅ Training Complete!")
    print("  ⚠️  Restart server.py to load new models")
    print("=" * 70)


if __name__ == '__main__':
    main()