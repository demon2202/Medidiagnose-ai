import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*np.object.*')
warnings.filterwarnings('ignore', message='.*np.int.*')
warnings.filterwarnings('ignore', message='.*np.float.*')

import numpy as np

# Fix for np.object deprecation in NumPy 1.20+
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'str'):
    np.str = str

# ============== REGULAR IMPORTS ==============
import os
import sys
import json
import logging
from datetime import datetime
from functools import wraps
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import joblib
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== TENSORFLOW IMPORT ==============
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    logger.info(f"✅ TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ TensorFlow not available: {e}")

# ============== PIL IMPORT ==============
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("✅ PIL loaded successfully")
except ImportError:
    logger.warning("⚠️ PIL not available - image processing will not work")

# ============== FLASK APP INITIALIZATION ==============
app = Flask(__name__)

# Configure CORS for frontend communication
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
            "*"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"]
    }
})


@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    origin = request.headers.get('Origin', '*')
    response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# ============== CONFIGURATION ==============
class Config:
    """Application configuration"""
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
    # Path to ml_model folder (one level up from backend)
    ML_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml_model')


app.config.from_object(Config)

# Create uploads folder if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# ============== MODEL PATHS ==============
MODEL_PATHS = {
    # Image-based models (TensorFlow/Keras)
    'skin_cancer': os.path.join(Config.ML_MODEL_DIR, 'skin_cancer_model.h5'),
    'skin_config': os.path.join(Config.ML_MODEL_DIR, 'skin_cancer_config.json'),
    'heart_image': os.path.join(Config.ML_MODEL_DIR, 'heart_image_model.h5'),
    'heart_config': os.path.join(Config.ML_MODEL_DIR, 'heart_image_config.json'),
    'breast_cancer': os.path.join(Config.ML_MODEL_DIR, 'breast_cancer_model.h5'),
    'breast_config': os.path.join(Config.ML_MODEL_DIR, 'breast_cancer_config.json'),
    'pneumonia': os.path.join(Config.ML_MODEL_DIR, 'pneumonia_model.h5'),
    'pneumonia_config': os.path.join(Config.ML_MODEL_DIR, 'pneumonia_config.json'),
    
    # Scikit-learn models
    'disease': os.path.join(Config.ML_MODEL_DIR, 'disease_model.joblib'),
    'label_encoder': os.path.join(Config.ML_MODEL_DIR, 'label_encoder.joblib'),
    'symptom_list': os.path.join(Config.ML_MODEL_DIR, 'symptom_list.json'),
    'cancer': os.path.join(Config.ML_MODEL_DIR, 'cancer_model.joblib'),
    'cancer_scaler': os.path.join(Config.ML_MODEL_DIR, 'cancer_scaler.joblib'),
    'heart_disease': os.path.join(Config.ML_MODEL_DIR, 'heart_disease_model.joblib'),
    'heart_scaler': os.path.join(Config.ML_MODEL_DIR, 'heart_scaler.joblib'),
    'heart_signal_scaler': os.path.join(Config.ML_MODEL_DIR, 'heart_signal_scaler.joblib'),
}

# Global models dictionary
models = {}
configs = {}

# ============== DISEASE CLASS DEFINITIONS ==============

# Skin Cancer Classes - 7 classes (HAM10000)
SKIN_CANCER_CLASSES = {
    0: {'code': 'akiec', 'name': 'Actinic Keratoses', 'type': 'pre-cancerous', 'severity': 'moderate'},
    1: {'code': 'bcc', 'name': 'Basal Cell Carcinoma', 'type': 'malignant', 'severity': 'moderate'},
    2: {'code': 'bkl', 'name': 'Benign Keratosis', 'type': 'benign', 'severity': 'low'},
    3: {'code': 'df', 'name': 'Dermatofibroma', 'type': 'benign', 'severity': 'low'},
    4: {'code': 'mel', 'name': 'Melanoma', 'type': 'malignant', 'severity': 'critical'},
    5: {'code': 'nv', 'name': 'Melanocytic Nevi', 'type': 'benign', 'severity': 'low'},
    6: {'code': 'vasc', 'name': 'Vascular Lesions', 'type': 'benign', 'severity': 'low'}
}

# Heart Conditions - 5 classes
HEART_CONDITIONS = {
    0: {'code': 'normal', 'name': 'Normal', 'severity': 'healthy'},
    1: {'code': 'mi', 'name': 'Myocardial Infarction', 'severity': 'critical'},
    2: {'code': 'arrhythmia', 'name': 'Arrhythmia', 'severity': 'moderate'},
    3: {'code': 'hf', 'name': 'Heart Failure Signs', 'severity': 'high'},
    4: {'code': 'hypertrophy', 'name': 'Ventricular Hypertrophy', 'severity': 'moderate'}
}

# Breast Cancer - 3 classes (matching new model)
BREAST_CANCER_CLASSES_3 = {
    0: {'code': 'normal', 'name': 'Normal', 'birads': 'BI-RADS 1', 'severity': 'healthy'},
    1: {'code': 'benign', 'name': 'Benign Tumor', 'birads': 'BI-RADS 2', 'severity': 'low'},
    2: {'code': 'malignant', 'name': 'Malignant Tumor', 'birads': 'BI-RADS 5', 'severity': 'critical'}
}

# Breast Cancer - 6 classes (legacy compatibility)
BREAST_CANCER_CLASSES_6 = {
    0: {'code': 'normal', 'name': 'Normal', 'birads': 'BI-RADS 1', 'severity': 'healthy'},
    1: {'code': 'benign', 'name': 'Benign Finding', 'birads': 'BI-RADS 2', 'severity': 'low'},
    2: {'code': 'probably_benign', 'name': 'Probably Benign', 'birads': 'BI-RADS 3', 'severity': 'low'},
    3: {'code': 'suspicious', 'name': 'Suspicious Abnormality', 'birads': 'BI-RADS 4', 'severity': 'moderate'},
    4: {'code': 'highly_suggestive', 'name': 'Highly Suggestive of Malignancy', 'birads': 'BI-RADS 5', 'severity': 'high'},
    5: {'code': 'malignant', 'name': 'Known Malignancy', 'birads': 'BI-RADS 6', 'severity': 'critical'}
}

# Pneumonia Classes - 2 classes
PNEUMONIA_CLASSES = {
    0: {'code': 'normal', 'name': 'Normal', 'severity': 'healthy'},
    1: {'code': 'pneumonia', 'name': 'Pneumonia', 'severity': 'high'}
}

# Demo symptoms list
DEMO_SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain',
    'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
    'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
    'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
    'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
    'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine',
    'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
    'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite',
    'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
    'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections',
    'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
    'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking',
    'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
    'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]


# ==============================================================================
#                           IMAGE VALIDATION
# ==============================================================================

def validate_image_type(img_array, expected_type):
    """
    Validate if uploaded image matches expected medical image type.
    Uses simple heuristics.
    
    Args:
        img_array: numpy array of image normalized to [0, 1]
        expected_type: 'skin', 'xray', 'breast', 'heart'
    
    Returns:
        dict with 'is_valid', 'message', 'confidence'
    """
    # Get image statistics
    if len(img_array.shape) == 4:
        img = img_array[0]  # Remove batch dimension
    else:
        img = img_array
    
    # Handle different channel configurations
    if len(img.shape) == 3 and img.shape[2] == 3:
        # RGB image
        gray = np.mean(img, axis=2)
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        rgb_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
        is_grayscale = rgb_diff < 0.05
        
        # Check for skin tones
        skin_mask = (r > 0.3) & (r < 0.9) & (g > 0.2) & (g < 0.8) & (b > 0.1) & (b < 0.7) & (r > g)
        skin_ratio = np.mean(skin_mask)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        # Grayscale with channel
        gray = img[:,:,0]
        is_grayscale = True
        skin_ratio = 0
    else:
        # Already grayscale
        gray = img
        is_grayscale = True
        skin_ratio = 0
    
    # Calculate statistics
    brightness = np.mean(gray)
    dark_ratio = np.mean(gray < 0.2)
    
    # Edge detection
    gx = np.abs(gray[1:, :] - gray[:-1, :])
    gy = np.abs(gray[:, 1:] - gray[:, :-1])
    edge_intensity = np.mean(gx) + np.mean(gy)
    
    # Validation based on expected type
    if expected_type == 'skin':
        # Skin images should be colorful with skin tones
        if is_grayscale:
            return {
                'is_valid': False,
                'message': 'This appears to be a grayscale image (like an X-ray or scan). Skin lesion images should be in color.',
                'suggestion': 'Please upload a COLOR photo of the skin lesion taken with a camera or phone.',
                'confidence': 0.8
            }
        if skin_ratio < 0.1 and brightness < 0.3:
            return {
                'is_valid': False,
                'message': 'This image does not appear to contain skin. It may be an X-ray, scan, or other type of medical image.',
                'suggestion': 'Please upload a color photo showing the skin lesion or mole.',
                'confidence': 0.7
            }
        return {'is_valid': True, 'message': 'Image appears to be a valid skin photo.', 'confidence': 0.8}
    
    elif expected_type in ['xray', 'pneumonia']:
        # X-rays should be grayscale with dark regions
        if not is_grayscale and skin_ratio > 0.3:
            return {
                'is_valid': False,
                'message': 'This appears to be a color photo of skin, not a chest X-ray.',
                'suggestion': 'Please upload a chest X-ray image (should be grayscale/black-and-white).',
                'confidence': 0.8
            }
        if dark_ratio < 0.1 and brightness > 0.8:
            return {
                'is_valid': False,
                'message': 'This image appears too bright to be an X-ray. It may be an ECG printout or document.',
                'suggestion': 'Please upload a chest X-ray image showing the lungs.',
                'confidence': 0.6
            }
        return {'is_valid': True, 'message': 'Image appears to be a valid X-ray.', 'confidence': 0.75}
    
    elif expected_type == 'breast':
        # Mammograms/ultrasounds are grayscale
        if not is_grayscale and skin_ratio > 0.3:
            return {
                'is_valid': False,
                'message': 'This appears to be a color photo of skin, not a mammogram or ultrasound.',
                'suggestion': 'Please upload a mammogram or breast ultrasound image (should be grayscale).',
                'confidence': 0.8
            }
        return {'is_valid': True, 'message': 'Image appears to be a valid breast scan.', 'confidence': 0.7}
    
    elif expected_type in ['heart', 'ecg']:
        # ECG images typically have grid patterns and high brightness
        if not is_grayscale and skin_ratio > 0.3:
            return {
                'is_valid': False,
                'message': 'This appears to be a color photo of skin, not an ECG.',
                'suggestion': 'Please upload an ECG printout or heart scan image.',
                'confidence': 0.8
            }
        return {'is_valid': True, 'message': 'Image appears to be a valid ECG/heart scan.', 'confidence': 0.7}
    
    return {'is_valid': True, 'message': 'Image validation passed.', 'confidence': 0.5}


# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def preprocess_image_for_skin(image, target_size=(224, 224)):
    """
    Preprocess image for SKIN CANCER model.
    Uses RGB - matches trained model!
    """
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image_for_xray(image, target_size=(224, 224)):
    """
    Preprocess image for PNEUMONIA model.
    Uses Grayscale - matches trained model!
    """
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('L')  # Grayscale
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
    return np.expand_dims(img_array, axis=0)


def preprocess_image_for_breast(image, target_size=(224, 224)):
    """
    Preprocess image for BREAST CANCER model.
    Uses Grayscale - matches trained model!
    """
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('L')  # Grayscale
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
    return np.expand_dims(img_array, axis=0)


def preprocess_image_for_heart(image, target_size=(224, 224)):
    """
    Preprocess image for HEART/ECG model.
    Uses Grayscale - matches trained model!
    """
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('L')  # Grayscale
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
    return np.expand_dims(img_array, axis=0)


def get_stage_info(condition_type, class_info, confidence):
    """Generate staging information based on prediction"""
    severity = class_info.get('severity', 'low')
    
    if condition_type == 'skin':
        if class_info.get('type') == 'malignant':
            if class_info.get('code') == 'mel':
                # Melanoma staging
                if confidence > 0.85:
                    return {
                        'stage': 'Stage II-III',
                        'description': 'Advanced melanoma features detected. Tumor may have grown deeper or spread to nearby lymph nodes.',
                        'prognosis': 'Requires immediate oncological evaluation and staging workup.'
                    }
                elif confidence > 0.6:
                    return {
                        'stage': 'Stage I-II',
                        'description': 'Early to intermediate melanoma. Tumor likely confined to skin.',
                        'prognosis': 'Good prognosis with prompt surgical treatment.'
                    }
                else:
                    return {
                        'stage': 'Stage 0-I',
                        'description': 'Very early melanoma or melanoma in situ.',
                        'prognosis': 'Excellent prognosis with complete surgical excision.'
                    }
            else:
                # Basal cell carcinoma
                if confidence > 0.8:
                    return {
                        'stage': 'Locally Advanced',
                        'description': 'Larger or deeper basal cell carcinoma.',
                        'prognosis': 'Treatable with Mohs surgery or excision.'
                    }
                else:
                    return {
                        'stage': 'Early',
                        'description': 'Small, superficial basal cell carcinoma.',
                        'prognosis': 'Excellent prognosis with treatment.'
                    }
        elif class_info.get('type') == 'pre-cancerous':
            return {
                'stage': 'Pre-cancerous',
                'description': 'Actinic keratosis that may progress to squamous cell carcinoma if untreated.',
                'prognosis': 'Treatable with cryotherapy, topical medications, or photodynamic therapy.'
            }
        else:
            return {
                'stage': 'Benign',
                'description': 'Non-cancerous skin lesion.',
                'prognosis': 'No cancer treatment needed. May remove for cosmetic reasons.'
            }
    
    elif condition_type == 'breast':
        if severity == 'critical':
            return {
                'stage': 'Stage II-IV (Estimated)',
                'description': 'Confirmed malignancy. Exact staging requires full workup including biopsy, imaging, and pathology.',
                'prognosis': 'Depends on specific stage, tumor size, lymph node involvement, and molecular subtype.'
            }
        elif severity == 'high':
            if confidence > 0.8:
                return {
                    'stage': 'Stage I-III (Suspected)',
                    'description': 'Highly suspicious findings. Biopsy will confirm diagnosis and staging.',
                    'prognosis': 'Early detection improves outcomes significantly. 5-year survival rates are high with treatment.'
                }
            else:
                return {
                    'stage': 'Stage 0-II (Possible)',
                    'description': 'Suspicious findings but may be early stage if malignant.',
                    'prognosis': 'Likely treatable if confirmed. Biopsy needed for definitive diagnosis.'
                }
        elif severity == 'moderate':
            return {
                'stage': 'Indeterminate - Further Evaluation Needed',
                'description': 'Requires biopsy for definitive diagnosis. Cannot determine staging without tissue analysis.',
                'prognosis': 'Most biopsies return benign results. Follow-up is important.'
            }
        else:
            return {
                'stage': 'N/A - Benign or Normal',
                'description': 'No malignancy detected.',
                'prognosis': 'Continue regular screening mammography as recommended by age and risk factors.'
            }
    
    elif condition_type == 'heart':
        if severity == 'critical':
            return {
                'stage': 'Acute/Severe Cardiac Event',
                'description': 'Signs of heart attack (myocardial infarction) or severe cardiac emergency.',
                'prognosis': 'Requires immediate emergency intervention. Time is critical - every minute counts.'
            }
        elif severity == 'high':
            return {
                'stage': 'Moderate-Severe Cardiac Abnormality',
                'description': 'Significant cardiac abnormality detected that requires urgent evaluation.',
                'prognosis': 'Requires prompt cardiology evaluation and likely treatment.'
            }
        elif severity == 'moderate':
            return {
                'stage': 'Mild-Moderate Cardiac Irregularity',
                'description': 'Cardiac irregularity detected such as arrhythmia or conduction abnormality.',
                'prognosis': 'Usually manageable with medication and lifestyle changes.'
            }
        else:
            return {
                'stage': 'Normal Cardiac Function',
                'description': 'No significant abnormality detected in ECG.',
                'prognosis': 'Maintain heart-healthy lifestyle and regular check-ups.'
            }
    
    elif condition_type == 'pneumonia':
        if severity == 'high':
            return {
                'stage': 'Moderate to Severe Pneumonia',
                'description': 'Pneumonia detected requiring medical treatment.',
                'prognosis': 'Usually responds well to appropriate antibiotics and supportive care.'
            }
        elif severity == 'moderate':
            return {
                'stage': 'Mild Pneumonia',
                'description': 'Early or mild pneumonia detected.',
                'prognosis': 'Usually self-limiting with supportive care. May require antibiotics.'
            }
        else:
            return {
                'stage': 'Normal Chest X-ray',
                'description': 'No pneumonia detected.',
                'prognosis': 'Continue healthy habits and seek care if symptoms develop.'
            }
    
    return {
        'stage': 'Unknown',
        'description': 'Unable to determine staging from available information.',
        'prognosis': 'Consult a specialist for proper evaluation and staging.'
    }


def get_treatment_options(condition_type, class_info, stage_info):
    """Get treatment recommendations based on condition"""
    severity = class_info.get('severity', 'low')
    
    if condition_type == 'skin':
        if class_info.get('code') == 'mel':
            # Melanoma treatment options
            return [
                'Wide local excision surgery (primary treatment)',
                'Sentinel lymph node biopsy (to check spread)',
                'Immunotherapy (pembrolizumab, nivolumab for advanced stages)',
                'Targeted therapy (BRAF/MEK inhibitors if mutation present)',
                'Radiation therapy (selected cases)',
                'Adjuvant therapy (for high-risk melanoma)',
                'Regular surveillance with full-body skin exams'
            ]
        elif class_info.get('code') == 'bcc':
            # Basal cell carcinoma
            return [
                'Mohs micrographic surgery (highest cure rate)',
                'Surgical excision',
                'Curettage and electrodesiccation',
                'Topical treatments (imiquimod, 5-fluorouracil)',
                'Radiation therapy (for inoperable cases)',
                'Photodynamic therapy',
                'Cryotherapy (for superficial lesions)'
            ]
        elif class_info.get('type') == 'pre-cancerous':
            # Actinic keratosis
            return [
                'Cryotherapy (liquid nitrogen freezing)',
                'Topical 5-fluorouracil cream',
                'Topical imiquimod cream',
                'Photodynamic therapy',
                'Chemical peels',
                'Laser resurfacing',
                'Field treatment for multiple lesions'
            ]
        else:
            # Benign lesions
            return [
                'Usually no treatment needed',
                'Surgical removal if desired (cosmetic)',
                'Cryotherapy for removal',
                'Laser treatment (cosmetic)',
                'Regular monitoring',
                'Sun protection to prevent new lesions'
            ]
    
    elif condition_type == 'breast':
        if severity in ['critical', 'high']:
            return [
                'Surgical options: Lumpectomy (breast-conserving) or Mastectomy',
                'Sentinel lymph node biopsy or axillary dissection',
                'Chemotherapy (neoadjuvant or adjuvant based on stage)',
                'Radiation therapy (usually after lumpectomy)',
                'Hormone therapy (if estrogen/progesterone receptor positive)',
                'Targeted therapy: Trastuzumab (Herceptin) for HER2+ tumors',
                'Immunotherapy (for triple-negative or advanced cases)',
                'Breast reconstruction (if mastectomy performed)',
                'Genetic counseling and testing (BRCA1/2)'
            ]
        elif severity == 'moderate':
            return [
                'Image-guided core needle biopsy',
                'Additional imaging (MRI, ultrasound)',
                'Close surveillance with follow-up mammography',
                'Surgical excision if biopsy confirms malignancy',
                'Genetic counseling if family history present'
            ]
        else:
            return [
                'Routine mammography screening (annual after age 40)',
                'Clinical breast exams',
                'Monthly breast self-examination',
                'Healthy lifestyle: maintain healthy weight, exercise',
                'Limit alcohol consumption',
                'Know your breast density and family history'
            ]
    
    elif condition_type == 'heart':
        if severity == 'critical':
            return [
                'Emergency cardiac catheterization',
                'Percutaneous coronary intervention (PCI/angioplasty with stent)',
                'Thrombolytic therapy (clot-busting drugs)',
                'Coronary artery bypass graft surgery (CABG)',
                'Intensive cardiac care unit monitoring',
                'Antiplatelet therapy (aspirin, clopidogrel)',
                'Beta-blockers, ACE inhibitors, statins',
                'Cardiac rehabilitation program'
            ]
        elif severity in ['high', 'moderate']:
            return [
                'Medications: Beta-blockers, ACE inhibitors, ARBs',
                'Antiarrhythmic drugs (for rhythm disorders)',
                'Blood thinners (anticoagulation if needed)',
                'Diuretics (for heart failure)',
                'Lifestyle modifications: diet, exercise, stress reduction',
                'Cardiac monitoring (Holter monitor, event recorder)',
                'Possible pacemaker or ICD (implantable defibrillator)',
                'Cardiac electrophysiology study if needed'
            ]
        else:
            return [
                'Heart-healthy diet (Mediterranean, DASH diet)',
                'Regular aerobic exercise (150 minutes per week)',
                'Blood pressure monitoring and control',
                'Cholesterol management',
                'Diabetes control if applicable',
                'Smoking cessation',
                'Stress reduction techniques',
                'Annual cardiac risk assessment'
            ]
    
    elif condition_type == 'pneumonia':
        if severity == 'high':
            return [
                'Antibiotic therapy (for bacterial pneumonia)',
                'Hospitalization for severe cases',
                'Oxygen therapy',
                'IV fluids and supportive care',
                'Chest physiotherapy',
                'Follow-up chest X-ray to confirm resolution',
                'Pneumococcal and influenza vaccination after recovery',
                'Rest and adequate hydration'
            ]
        elif severity == 'moderate':
            return [
                'Rest and hydration',
                'Over-the-counter medications for fever/pain',
                'Antiviral medications (if viral and applicable)',
                'Monitor oxygen saturation',
                'Seek medical care if symptoms worsen',
                'Breathing exercises',
                'Avoid smoking and secondhand smoke'
            ]
        else:
            return [
                'Annual influenza vaccination',
                'Pneumococcal vaccination (for at-risk individuals)',
                'Good hand hygiene',
                'Healthy lifestyle to boost immunity',
                'Avoid smoking',
                'Stay up to date with vaccinations'
            ]
    
    return ['Consult with a specialist for personalized treatment options']


def get_urgency_timeline(severity):
    """Get recommended timeline based on severity"""
    timelines = {
        'critical': {
            'timeline': 'IMMEDIATELY - Within hours',
            'action': 'Seek emergency medical care or call 911',
            'color': 'red',
            'icon': '🚨'
        },
        'high': {
            'timeline': 'URGENT - Within 24-48 hours',
            'action': 'Contact specialist immediately',
            'color': 'orange',
            'icon': '⚠️'
        },
        'moderate': {
            'timeline': 'Soon - Within 1-2 weeks',
            'action': 'Schedule specialist appointment',
            'color': 'yellow',
            'icon': '📅'
        },
        'low': {
            'timeline': 'Routine - Within 1-3 months',
            'action': 'Follow-up at next regular appointment',
            'color': 'green',
            'icon': '✓'
        },
        'healthy': {
            'timeline': 'Annual screening',
            'action': 'Continue regular health maintenance',
            'color': 'blue',
            'icon': '💚'
        }
    }
    return timelines.get(severity, timelines['low'])


def get_skin_recommendations(class_info, confidence):
    """Generate skin cancer specific recommendations"""
    cancer_type = class_info.get('type', 'benign')
    severity = class_info.get('severity', 'low')
    
    if cancer_type == 'malignant' and class_info.get('code') == 'mel':
        return {
            'level': 'critical',
            'title': 'URGENT: Melanoma Detected',
            'message': 'Melanoma is the most serious type of skin cancer. Immediate specialist evaluation is essential.',
            'actions': [
                'Contact a dermatologist or surgical oncologist within 24-48 hours',
                'Do NOT attempt to remove or treat the lesion yourself',
                'Take clear photos to document the lesion',
                'Avoid sun exposure on the affected area',
                'Prepare a list of any changes you\'ve noticed (size, color, shape, symptoms)',
                'Bring any previous photos of the lesion if available'
            ],
            'next_steps': [
                'Skin biopsy for definitive diagnosis',
                'Staging workup if melanoma confirmed (CT, PET scan)',
                'Wide local excision surgery',
                'Possible sentinel lymph node biopsy',
                'Genetic testing if family history present'
            ]
        }
    elif cancer_type == 'malignant':
        return {
            'level': 'high',
            'title': 'Basal Cell Carcinoma Suspected',
            'message': 'BCC is the most common skin cancer. While rarely life-threatening, it requires treatment.',
            'actions': [
                'Schedule dermatologist appointment within 1-2 weeks',
                'Avoid further sun exposure to the area',
                'Do not pick at or irritate the lesion',
                'Document any changes in size or appearance',
                'Use broad-spectrum SPF 30+ sunscreen daily'
            ],
            'next_steps': [
                'Skin biopsy for confirmation',
                'Mohs surgery or surgical excision',
                'Regular skin checks afterward',
                'Full-body skin examination'
            ]
        }
    elif cancer_type == 'pre-cancerous':
        return {
            'level': 'moderate',
            'title': 'Pre-cancerous Lesion Detected',
            'message': 'Actinic keratoses can develop into squamous cell carcinoma if left untreated.',
            'actions': [
                'See a dermatologist within 2-4 weeks',
                'Use broad-spectrum SPF 30+ sunscreen daily',
                'Wear protective clothing outdoors',
                'Monitor for any changes in the lesion',
                'Check for other similar lesions'
            ],
            'next_steps': [
                'Treatment with cryotherapy, topical medications, or other methods',
                'Regular skin cancer screening',
                'Full-body skin examination'
            ]
        }
    else:
        return {
            'level': 'low',
            'title': 'Benign Lesion - Low Risk',
            'message': 'This appears to be a non-cancerous skin lesion.',
            'actions': [
                'Continue regular skin self-examinations',
                'Use the ABCDE rule to monitor moles (Asymmetry, Border, Color, Diameter, Evolving)',
                'Annual skin cancer screenings recommended',
                'Protect skin from excessive sun exposure',
                'Use broad-spectrum sunscreen SPF 30+ daily'
            ],
            'next_steps': [
                'No immediate treatment needed',
                'Can be removed for cosmetic reasons if desired',
                'Regular monitoring recommended'
            ]
        }


def get_heart_recommendations(class_info, confidence):
    """Generate heart condition specific recommendations"""
    severity = class_info.get('severity', 'healthy')
    
    if severity == 'critical':
        return {
            'level': 'critical',
            'title': 'EMERGENCY: Cardiac Event Detected',
            'message': 'Signs of heart attack or severe cardiac condition. Immediate emergency care needed.',
            'actions': [
                'CALL 911 IMMEDIATELY - Do not drive yourself',
                'Chew aspirin (325mg) if not allergic and not already taken',
                'Sit or lie down in comfortable position',
                'Loosen tight clothing',
                'Stay calm and await emergency services',
                'Have someone stay with you'
            ],
            'warning_signs': [
                'Chest pain or pressure (may radiate to arm, jaw, back)',
                'Shortness of breath',
                'Sweating, nausea, lightheadedness',
                'Rapid or irregular heartbeat',
                'Severe fatigue or weakness'
            ]
        }
    elif severity == 'high':
        return {
            'level': 'high',
            'title': 'Significant Cardiac Abnormality',
            'message': 'Urgent cardiology evaluation recommended.',
            'actions': [
                'Contact cardiologist or go to ER today',
                'Avoid strenuous activity',
                'Monitor for worsening symptoms',
                'Take prescribed heart medications as directed',
                'Have someone stay with you',
                'Keep list of medications handy'
            ]
        }
    elif severity == 'moderate':
        return {
            'level': 'moderate',
            'title': 'Cardiac Irregularity Detected',
            'message': 'Should be evaluated by a cardiologist.',
            'actions': [
                'Schedule cardiology appointment within 1-2 weeks',
                'Continue current medications',
                'Limit caffeine and alcohol',
                'Monitor heart rate and blood pressure',
                'Avoid excessive stress',
                'Keep a symptom diary'
            ]
        }
    else:
        return {
            'level': 'healthy',
            'title': 'Normal Heart Findings',
            'message': 'No significant cardiac abnormalities detected.',
            'actions': [
                'Continue heart-healthy lifestyle',
                'Regular aerobic exercise (150 minutes per week)',
                'Maintain healthy diet (low sodium, high fiber)',
                'Monitor blood pressure regularly',
                'Annual cardiac checkups',
                'Know your cholesterol and blood sugar levels'
            ]
        }


def get_breast_recommendations(class_info, confidence):
    """Generate breast cancer specific recommendations"""
    severity = class_info.get('severity', 'low')
    
    if severity == 'critical':
        return {
            'level': 'critical',
            'title': 'Confirmed Breast Malignancy',
            'message': 'Immediate oncology consultation and treatment planning needed.',
            'actions': [
                'Contact your oncologist or breast surgeon immediately',
                'Gather all imaging and pathology reports',
                'Prepare for treatment planning consultation',
                'Consider seeking second opinion at comprehensive cancer center',
                'Bring support person to appointments',
                'Ask about clinical trials if appropriate'
            ],
            'next_steps': [
                'Complete staging workup (CT, bone scan, PET if needed)',
                'Tumor marker testing and genetic profiling',
                'Discuss treatment options: surgery, chemotherapy, radiation',
                'Genetic counseling (BRCA testing)',
                'Fertility preservation discussion if premenopausal'
            ]
        }
    elif severity == 'high':
        return {
            'level': 'high',
            'title': 'Highly Suspicious Finding',
            'message': 'Findings are highly suggestive of malignancy. Urgent evaluation needed.',
            'actions': [
                'Contact breast surgeon/oncologist within 24-48 hours',
                'Image-guided biopsy will be scheduled',
                'Avoid panic - get definitive diagnosis first',
                'Prepare questions for your doctor',
                'Bring list of medications and medical history',
                'Consider bringing support person'
            ]
        }
    elif severity == 'moderate':
        return {
            'level': 'moderate',
            'title': 'Suspicious Finding - Biopsy Recommended',
            'message': 'Further evaluation needed to determine nature of finding.',
            'actions': [
                'Schedule appointment with breast specialist within 1-2 weeks',
                'Image-guided biopsy likely recommended',
                'Remember: Most biopsies turn out benign (80%)',
                'Don\'t delay follow-up appointments',
                'Continue monthly breast self-exams'
            ]
        }
    elif severity == 'low':
        return {
            'level': 'low',
            'title': 'Benign/Probably Benign Finding',
            'message': 'Finding appears non-cancerous. Follow-up may be recommended.',
            'actions': [
                'Follow-up imaging in 6 months may be recommended',
                'Continue regular screening mammography schedule',
                'Perform monthly breast self-examinations',
                'Report any new lumps or changes immediately',
                'Know your breast density'
            ]
        }
    else:
        return {
            'level': 'healthy',
            'title': 'Normal Breast Tissue',
            'message': 'No abnormalities detected.',
            'actions': [
                'Continue annual mammography screening (age 40+)',
                'Monthly breast self-examination',
                'Maintain healthy lifestyle and weight',
                'Limit alcohol consumption',
                'Know your family history and breast density',
                'Discuss screening schedule with doctor based on risk factors'
            ]
        }


def get_pneumonia_recommendations(class_info, confidence):
    """Generate pneumonia specific recommendations"""
    severity = class_info.get('severity', 'healthy')
    
    if severity == 'high':
        return {
            'level': 'high',
            'title': 'Pneumonia Detected',
            'message': 'Pneumonia requires medical treatment.',
            'actions': [
                'See a doctor within 24 hours',
                'Antibiotics will likely be prescribed (if bacterial)',
                'Rest and stay well hydrated',
                'Monitor temperature and symptoms',
                'Use humidifier to ease breathing',
                'Take all prescribed medications as directed'
            ],
            'warning_signs': [
                'High fever (over 102°F/39°C)',
                'Severe difficulty breathing or rapid breathing',
                'Confusion or altered mental state',
                'Blue-tinged lips or fingernails (cyanosis)',
                'Severe chest pain',
                'Persistent vomiting'
            ],
            'note': 'Seek emergency care if you experience any warning signs listed above.'
        }
    elif severity == 'moderate':
        return {
            'level': 'moderate',
            'title': 'Possible Viral Pneumonia',
            'message': 'Viral pneumonia usually resolves with supportive care.',
            'actions': [
                'Rest and stay hydrated (drink plenty of fluids)',
                'Take over-the-counter fever reducers (acetaminophen, ibuprofen)',
                'Use a humidifier',
                'Monitor symptoms closely',
                'Seek medical care if symptoms worsen',
                'Avoid smoking and secondhand smoke'
            ]
        }
    else:
        return {
            'level': 'healthy',
            'title': 'Normal Chest X-ray',
            'message': 'No signs of pneumonia detected.',
            'actions': [
                'Continue healthy habits',
                'Get annual flu vaccination',
                'Consider pneumococcal vaccine (if 65+ or at risk)',
                'Practice good hand hygiene',
                'Avoid close contact with sick individuals',
                'Don\'t smoke - quit if you do'
            ]
        }


def get_disease_recommendations(disease_name, confidence):
    """Generate recommendations based on predicted disease"""
    general_recommendations = [
        'Consult with a healthcare professional for proper diagnosis',
        'Keep track of your symptoms and their progression',
        'Maintain good hygiene and rest',
        'Stay hydrated and maintain a balanced diet',
        'Take over-the-counter medications as appropriate for symptom relief'
    ]
    
    if confidence > 0.8:
        general_recommendations.insert(0, f'High confidence in {disease_name} prediction - seek medical attention promptly')
    elif confidence > 0.5:
        general_recommendations.insert(0, f'Moderate confidence - consider scheduling a doctor visit soon')
    else:
        general_recommendations.insert(0, 'Low confidence - monitor symptoms and consult if they persist or worsen')
    
    return general_recommendations


# ==============================================================================
#                           ERROR HANDLER DECORATOR
# ==============================================================================

def handle_errors(f):
    """Decorator to handle exceptions in routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': str(e),
                'endpoint': f.__name__
            }), 500
    return decorated


# ==============================================================================
#                           MODEL LOADING
# ==============================================================================

def load_models():
    """Load all ML models from the ml_model directory"""
    global models, configs
    
    logger.info("=" * 50)
    logger.info("Loading ML Models...")
    logger.info(f"Model directory: {os.path.abspath(Config.ML_MODEL_DIR)}")
    logger.info("=" * 50)
    
    # Check if model directory exists
    if not os.path.exists(Config.ML_MODEL_DIR):
        logger.error(f"❌ Model directory not found: {Config.ML_MODEL_DIR}")
        logger.info("Creating ml_model directory...")
        os.makedirs(Config.ML_MODEL_DIR, exist_ok=True)
        return
    
    # Load TensorFlow/Keras models
    if TF_AVAILABLE:
        # Skin Cancer Model
        if os.path.exists(MODEL_PATHS['skin_cancer']):
            try:
                models['skin_cancer'] = keras.models.load_model(MODEL_PATHS['skin_cancer'])
                logger.info("✅ Skin cancer model loaded")
                if os.path.exists(MODEL_PATHS['skin_config']):
                    with open(MODEL_PATHS['skin_config']) as f:
                        configs['skin_cancer'] = json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load skin cancer model: {e}")
        else:
            logger.warning(f"⚠️ Skin cancer model not found: {MODEL_PATHS['skin_cancer']}")
        
        # Heart Image Model
        if os.path.exists(MODEL_PATHS['heart_image']):
            try:
                models['heart_image'] = keras.models.load_model(MODEL_PATHS['heart_image'])
                logger.info("✅ Heart image model loaded")
                if os.path.exists(MODEL_PATHS['heart_config']):
                    with open(MODEL_PATHS['heart_config']) as f:
                        configs['heart_image'] = json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load heart image model: {e}")
        else:
            logger.warning(f"⚠️ Heart image model not found: {MODEL_PATHS['heart_image']}")
        
        # Breast Cancer Model
        if os.path.exists(MODEL_PATHS['breast_cancer']):
            try:
                models['breast_cancer'] = keras.models.load_model(MODEL_PATHS['breast_cancer'])
                logger.info("✅ Breast cancer model loaded")
                if os.path.exists(MODEL_PATHS['breast_config']):
                    with open(MODEL_PATHS['breast_config']) as f:
                        configs['breast_cancer'] = json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load breast cancer model: {e}")
        else:
            logger.warning(f"⚠️ Breast cancer model not found: {MODEL_PATHS['breast_cancer']}")
        
        # Pneumonia Model
        if os.path.exists(MODEL_PATHS['pneumonia']):
            try:
                models['pneumonia'] = keras.models.load_model(MODEL_PATHS['pneumonia'])
                logger.info("✅ Pneumonia model loaded")
                if os.path.exists(MODEL_PATHS['pneumonia_config']):
                    with open(MODEL_PATHS['pneumonia_config']) as f:
                        configs['pneumonia'] = json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load pneumonia model: {e}")
        else:
            logger.warning(f"⚠️ Pneumonia model not found: {MODEL_PATHS['pneumonia']}")
    else:
        logger.warning("⚠️ TensorFlow not available - image models will not be loaded")
    
    # Load Scikit-learn models
    # Disease Prediction Model
    if os.path.exists(MODEL_PATHS['disease']):
        try:
            models['disease'] = joblib.load(MODEL_PATHS['disease'])
            logger.info("✅ Disease prediction model loaded")
            
            if os.path.exists(MODEL_PATHS['label_encoder']):
                models['label_encoder'] = joblib.load(MODEL_PATHS['label_encoder'])
                logger.info("✅ Label encoder loaded")
            
            if os.path.exists(MODEL_PATHS['symptom_list']):
                with open(MODEL_PATHS['symptom_list']) as f:
                    models['symptom_list'] = json.load(f)
                logger.info("✅ Symptom list loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load disease model: {e}")
    else:
        logger.warning(f"⚠️ Disease model not found: {MODEL_PATHS['disease']}")
    
    # Cancer Screening Model (tumor characteristics)
    if os.path.exists(MODEL_PATHS['cancer']):
        try:
            models['cancer'] = joblib.load(MODEL_PATHS['cancer'])
            logger.info("✅ Cancer screening model loaded")
            
            if os.path.exists(MODEL_PATHS['cancer_scaler']):
                models['cancer_scaler'] = joblib.load(MODEL_PATHS['cancer_scaler'])
                logger.info("✅ Cancer scaler loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load cancer model: {e}")
    else:
        logger.warning(f"⚠️ Cancer model not found: {MODEL_PATHS['cancer']}")
    
    # Heart Risk Model
    if os.path.exists(MODEL_PATHS['heart_disease']):
        try:
            
            models['heart_risk'] = joblib.load(MODEL_PATHS['heart_disease'])
            models['heart_scaler'] = joblib.load(MODEL_PATHS['heart_scaler'])

            logger.info("✅ Heart risk model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load heart risk model: {e}")
    else:
        logger.warning(f"⚠️ Heart risk model not found: {MODEL_PATHS['heart_risk']}")
    
    logger.info("=" * 50)
    logger.info(f"Model loading complete. Loaded: {list(models.keys())}")
    logger.info("=" * 50)


# ==============================================================================
#                           API ROUTES
# ==============================================================================

@app.route('/')
def home():
    """API home endpoint with information"""
    return jsonify({
        'status': 'online',
        'name': 'MediDiagnose-AI API',
        'version': '4.0.0 - FIXED',
        'description': 'AI-powered medical diagnosis system with image validation',
        'features': [
            'Skin Cancer Detection with Staging',
            'Heart Disease Detection from ECG/Scans',
            'Breast Cancer Detection from Mammograms',
            'Pneumonia Detection from X-rays',
            'Symptom-based Disease Prediction',
            'Breast Cancer Screening from Tumor Characteristics',
            'Heart Disease Risk Assessment',
            'Image Type Validation'
        ],
        'endpoints': {
            'GET /': 'API info',
            'GET /health': 'Health check and model status',
            'GET /symptoms': 'Get available symptoms list',
            'POST /analyze/skin': 'Skin cancer detection from image',
            'POST /analyze/heart': 'Heart condition from ECG/scan image',
            'POST /analyze/breast': 'Breast cancer from mammogram',
            'POST /analyze/xray': 'Pneumonia from chest X-ray',
            'POST /predict-disease': 'Disease prediction from symptoms',
            'POST /predict-heart': 'Heart disease risk from factors',
            'POST /predict-cancer': 'Breast cancer from tumor characteristics'
        },
        'models_loaded': list(models.keys()),
        'tensorflow_available': TF_AVAILABLE,
        'pil_available': PIL_AVAILABLE
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'skin_cancer': 'skin_cancer' in models,
            'heart_image': 'heart_image' in models,
            'breast_cancer': 'breast_cancer' in models,
            'pneumonia': 'pneumonia' in models,
            'disease': 'disease' in models,
            'cancer': 'cancer' in models,
            'heart_risk': 'heart_risk' in models
        },
        'dependencies': {
            'tensorflow': TF_AVAILABLE,
            'pil': PIL_AVAILABLE
        },
        'model_directory': os.path.abspath(Config.ML_MODEL_DIR),
        'model_directory_exists': os.path.exists(Config.ML_MODEL_DIR)
    })


@app.route('/symptoms', methods=['GET', 'OPTIONS'])
def get_symptoms():
    """Get list of available symptoms for disease prediction"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if 'symptom_list' in models:
        symptoms = [s.replace('_', ' ').title() for s in models['symptom_list']]
        return jsonify({
            'success': True,
            'symptoms': symptoms,
            'count': len(symptoms)
        })
    
    # Return demo symptoms if model not loaded
    symptoms = [s.replace('_', ' ').title() for s in DEMO_SYMPTOMS]
    return jsonify({
        'success': True,
        'symptoms': symptoms,
        'count': len(symptoms),
        'demo_mode': True,
        'note': 'Using demo symptoms list - load disease model for full list'
    })


# ==============================================================================
#                     SYMPTOM-BASED DISEASE PREDICTION
# ==============================================================================

@app.route('/predict-disease', methods=['POST', 'OPTIONS'])
@handle_errors
def predict_disease():
    """Predict disease from symptoms - FIXED for feature mismatch"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    symptoms = data.get('symptoms', [])
    if not symptoms:
        return jsonify({'success': False, 'error': 'No symptoms provided'}), 400
    
    # Normalize symptoms
    normalized_symptoms = []
    for symptom in symptoms:
        normalized = symptom.lower().strip().replace(' ', '_')
        normalized_symptoms.append(normalized)
    
    # Check if disease model is loaded
    if 'disease' not in models or 'label_encoder' not in models:
        # Demo mode response
        demo_diseases = [
            {'disease': 'Common Cold', 'probability': 0.65},
            {'disease': 'Influenza', 'probability': 0.20},
            {'disease': 'Allergic Rhinitis', 'probability': 0.10},
            {'disease': 'Sinusitis', 'probability': 0.03},
            {'disease': 'Bronchitis', 'probability': 0.02}
        ]
        
        return jsonify({
            'success': True,
            'demo_mode': True,
            'prediction': {
                'disease': demo_diseases[0]['disease'],
                'confidence': demo_diseases[0]['probability'],
                'confidence_percent': f"{demo_diseases[0]['probability'] * 100:.1f}%"
            },
            'top_predictions': demo_diseases,
            'symptoms_analyzed': symptoms,
            'symptoms_count': len(symptoms),
            'recommendations': get_disease_recommendations(demo_diseases[0]['disease'], demo_diseases[0]['probability']),
            'note': 'Demo mode - train and load disease model for real predictions'
        })
    
    try:
        # Get symptom list from saved file
        symptom_list = models.get('symptom_list', DEMO_SYMPTOMS)
        
        # IMPORTANT: Check how many features the model expects
        # Try to get the expected number of features from the model
        try:
            if hasattr(models['disease'], 'n_features_in_'):
                expected_features = models['disease'].n_features_in_
            elif hasattr(models['disease'], 'estimators_'):
                # For ensemble models like VotingClassifier
                first_estimator = models['disease'].estimators_[0]
                if hasattr(first_estimator, 'n_features_in_'):
                    expected_features = first_estimator.n_features_in_
                else:
                    expected_features = len(symptom_list)
            else:
                expected_features = len(symptom_list)
        except:
            expected_features = len(symptom_list)
        
        logger.info(f"Disease model expects {expected_features} features, symptom list has {len(symptom_list)} symptoms")
        
        # Use the smaller of the two to prevent mismatch
        num_features = min(expected_features, len(symptom_list))
        
        # Create feature vector with correct size
        feature_vector = np.zeros(expected_features)
        
        matched_symptoms = []
        unmatched_symptoms = []
        
        for symptom in normalized_symptoms:
            if symptom in symptom_list:
                idx = symptom_list.index(symptom)
                # Only add if index is within the expected feature range
                if idx < expected_features:
                    feature_vector[idx] = 1
                    matched_symptoms.append(symptom)
                else:
                    unmatched_symptoms.append(symptom)
            else:
                unmatched_symptoms.append(symptom)
        
        if len(matched_symptoms) == 0:
            return jsonify({
                'success': False,
                'error': 'None of the provided symptoms match our database. Please check symptom names.',
                'symptoms_provided': symptoms,
                'hint': 'Try symptoms like: fever, headache, cough, fatigue, etc.'
            }), 400
        
        # Make prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        if hasattr(models['disease'], 'predict_proba'):
            probabilities = models['disease'].predict_proba(feature_vector)[0]
            predicted_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_idx])
            
            # Get top 5 predictions
            top_indices = np.argsort(probabilities)[::-1][:5]
            top_predictions = []
            for idx in top_indices:
                disease_name = models['label_encoder'].inverse_transform([idx])[0]
                top_predictions.append({
                    'disease': disease_name,
                    'probability': float(probabilities[idx])
                })
        else:
            predicted_idx = int(models['disease'].predict(feature_vector)[0])
            confidence = 0.75
            disease_name = models['label_encoder'].inverse_transform([predicted_idx])[0]
            top_predictions = [{'disease': disease_name, 'probability': confidence}]
        
        # Decode prediction
        disease_name = models['label_encoder'].inverse_transform([predicted_idx])[0]
        
        return jsonify({
            'success': True,
            'prediction': {
                'disease': disease_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%"
            },
            'top_predictions': top_predictions,
            'symptoms_analyzed': symptoms,
            'matched_symptoms': [s.replace('_', ' ').title() for s in matched_symptoms],
            'unmatched_symptoms': unmatched_symptoms,
            'recommendations': get_disease_recommendations(disease_name, confidence)
        })
        
    except Exception as e:
        logger.error(f"Disease prediction error: {e}\n{traceback.format_exc()}")
        
        # Return a graceful error with demo prediction
        demo_diseases = [
            {'disease': 'Common Cold', 'probability': 0.60},
            {'disease': 'Viral Infection', 'probability': 0.25},
            {'disease': 'Fatigue Syndrome', 'probability': 0.10}
        ]
        
        return jsonify({
            'success': True,
            'demo_mode': True,
            'prediction': {
                'disease': demo_diseases[0]['disease'],
                'confidence': demo_diseases[0]['probability'],
                'confidence_percent': f"{demo_diseases[0]['probability'] * 100:.1f}%"
            },
            'top_predictions': demo_diseases,
            'symptoms_analyzed': symptoms,
            'recommendations': get_disease_recommendations(demo_diseases[0]['disease'], demo_diseases[0]['probability']),
            'note': f'Using demo mode due to model compatibility issue. Error: {str(e)[:100]}'
        })

# ==============================================================================
#                     TUMOR CHARACTERISTICS CANCER PREDICTION
# ==============================================================================

@app.route('/predict-cancer', methods=['POST', 'OPTIONS'])
@handle_errors
def predict_cancer():
    """Predict breast cancer from tumor characteristics"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Expected features (Wisconsin Breast Cancer Dataset features)
    required_features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]
    
    # Check if all features are present
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return jsonify({
            'success': False,
            'error': f'Missing features: {", ".join(missing_features)}',
            'required_features': required_features
        }), 400
    
    try:
        # Extract features in correct order
        features = np.array([[
            float(data['radius_mean']),
            float(data['texture_mean']),
            float(data['perimeter_mean']),
            float(data['area_mean']),
            float(data['smoothness_mean']),
            float(data['compactness_mean']),
            float(data['concavity_mean']),
            float(data['concave_points_mean']),
            float(data['symmetry_mean']),
            float(data['fractal_dimension_mean'])
        ]])
        
        # Check if model is loaded
        if 'cancer' not in models:
            # Demo mode - simple heuristic based on feature values
            risk_score = (
                (features[0][0] - 12) * 0.15 +  # radius
                (features[0][2] - 78) * 0.05 +  # perimeter
                (features[0][3] - 450) * 0.001 +  # area
                (features[0][6]) * 3.0 +  # concavity
                (features[0][7]) * 5.0  # concave_points
            )
            
            probability = 1 / (1 + np.exp(-risk_score))  # Sigmoid
            probability = float(np.clip(probability, 0.05, 0.95))
            prediction = 'Malignant' if probability > 0.5 else 'Benign'
            confidence = probability if probability > 0.5 else (1 - probability)
            
            if prediction == 'Malignant':
                recommendation = {
                    'level': 'critical' if probability > 0.7 else 'warning',
                    'message': 'The tumor characteristics suggest malignancy. Immediate medical consultation is strongly recommended.',
                    'actions': [
                        'Schedule an appointment with an oncologist immediately',
                        'Bring all test results and imaging to your appointment',
                        'Do not delay - early treatment significantly improves outcomes',
                        'Consider getting a second opinion from a cancer specialist',
                        'Discuss treatment options including surgery, chemotherapy, and radiation'
                    ]
                }
            else:
                recommendation = {
                    'level': 'info',
                    'message': 'The tumor characteristics appear benign, but follow-up is still important.',
                    'actions': [
                        'Continue regular breast cancer screening',
                        'Schedule follow-up mammograms as recommended',
                        'Perform monthly breast self-examinations',
                        'Report any changes to your healthcare provider immediately',
                        'Maintain a healthy lifestyle to reduce cancer risk'
                    ]
                }
            
            return jsonify({
                'success': True,
                'demo_mode': True,
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%",
                'recommendation': recommendation,
                'features_analyzed': required_features,
                'note': 'Demo mode - load trained model for real predictions'
            })
        
        # Apply scaler if available
        if 'cancer_scaler' in models:
            features = models['cancer_scaler'].transform(features)
        
        # Real model prediction
        if hasattr(models['cancer'], 'predict_proba'):
            probabilities = models['cancer'].predict_proba(features)[0]
            probability = float(probabilities[1])  # Probability of malignancy
            prediction = 'Malignant' if probability > 0.5 else 'Benign'
            confidence = probability if probability > 0.5 else (1 - probability)
        else:
            pred_class = int(models['cancer'].predict(features)[0])
            prediction = 'Malignant' if pred_class == 1 else 'Benign'
            probability = 0.85 if pred_class == 1 else 0.15
            confidence = 0.85
        
        if prediction == 'Malignant':
            recommendation = {
                'level': 'critical' if probability > 0.7 else 'warning',
                'message': 'The tumor characteristics suggest malignancy. Immediate medical consultation is strongly recommended.',
                'actions': [
                    'Schedule an appointment with an oncologist immediately',
                    'Bring all test results and imaging to your appointment',
                    'Do not delay - early treatment significantly improves outcomes',
                    'Consider getting a second opinion from a cancer specialist',
                    'Discuss treatment options including surgery, chemotherapy, and radiation'
                ]
            }
        else:
            recommendation = {
                'level': 'info',
                'message': 'The tumor characteristics appear benign, but follow-up is still important.',
                'actions': [
                    'Continue regular breast cancer screening',
                    'Schedule follow-up mammograms as recommended',
                    'Perform monthly breast self-examinations',
                    'Report any changes to your healthcare provider immediately',
                    'Maintain a healthy lifestyle to reduce cancer risk'
                ]
            }
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'recommendation': recommendation,
            'features_analyzed': required_features
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid feature values: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Cancer prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==============================================================================
#                     HEART DISEASE RISK PREDICTION
# ==============================================================================

@app.route('/predict-heart', methods=['POST', 'OPTIONS'])
@handle_errors
def predict_heart():
    """Predict heart disease risk from clinical factors"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Expected features (Cleveland Heart Disease Dataset)
    required_features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)'
    }
    
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return jsonify({
            'success': False,
            'error': f'Missing features: {", ".join(missing_features)}',
            'required_features': required_features,
            'feature_descriptions': feature_descriptions
        }), 400
    
    try:
        features = np.array([[
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]])
        
        if 'heart_risk' not in models:
            # Demo mode - simple risk calculation
            age_risk = (float(data['age']) - 30) * 0.02
            bp_risk = (float(data['trestbps']) - 120) * 0.01
            chol_risk = (float(data['chol']) - 200) * 0.005
            heart_rate_factor = (180 - float(data['thalach'])) * 0.01
            chest_pain_risk = float(data['cp']) * 0.15
            vessels_risk = float(data['ca']) * 0.2
            
            risk_score = age_risk + bp_risk + chol_risk + heart_rate_factor + chest_pain_risk + vessels_risk
            probability = 1 / (1 + np.exp(-risk_score))
            probability = float(np.clip(probability, 0.05, 0.95))
            
            prediction = 'High Risk' if probability > 0.5 else 'Low Risk'
            confidence = probability if probability > 0.5 else (1 - probability)
            
            # Risk factors analysis
            risk_factors = []
            if float(data['age']) > 55:
                risk_factors.append('Age over 55')
            if float(data['trestbps']) > 140:
                risk_factors.append('High blood pressure')
            if float(data['chol']) > 240:
                risk_factors.append('High cholesterol')
            if float(data['thalach']) < 120:
                risk_factors.append('Low maximum heart rate')
            if float(data['cp']) > 0:
                risk_factors.append('Chest pain symptoms')
            if float(data['ca']) > 0:
                risk_factors.append('Vessel abnormalities detected')
            
            if prediction == 'High Risk':
                recommendation = {
                    'level': 'high' if probability > 0.7 else 'moderate',
                    'message': 'Your risk factors suggest elevated heart disease risk.',
                    'actions': [
                        'Schedule an appointment with a cardiologist',
                        'Get a comprehensive cardiac evaluation',
                        'Monitor blood pressure regularly',
                        'Consider lifestyle modifications',
                        'Discuss medication options with your doctor'
                    ],
                    'risk_factors': risk_factors
                }
            else:
                recommendation = {
                    'level': 'low',
                    'message': 'Your risk factors suggest relatively low heart disease risk.',
                    'actions': [
                        'Continue healthy lifestyle habits',
                        'Exercise regularly (150 min/week)',
                        'Maintain healthy diet',
                        'Annual health checkups',
                        'Monitor blood pressure periodically'
                    ],
                    'risk_factors': risk_factors
                }
            
            return jsonify({
                'success': True,
                'demo_mode': True,
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%",
                'recommendation': recommendation,
                'features_analyzed': required_features,
                'note': 'Demo mode - load trained model for real predictions'
            })
        
        # Real model prediction
        if hasattr(models['heart_risk'], 'predict_proba'):
            probabilities = models['heart_risk'].predict_proba(features)[0]
            probability = float(probabilities[1])
            prediction = 'High Risk' if probability > 0.5 else 'Low Risk'
            confidence = probability if probability > 0.5 else (1 - probability)
        else:
            features = models['heart_scaler'].transform(features)
            pred_class = int(models['heart_risk'].predict(features)[0])
            prediction = 'High Risk' if pred_class == 1 else 'Low Risk'
            probability = 0.85 if pred_class == 1 else 0.15
            confidence = 0.85
        
        # Risk factors analysis
        risk_factors = []
        if float(data['age']) > 55:
            risk_factors.append('Age over 55')
        if float(data['trestbps']) > 140:
            risk_factors.append('High blood pressure')
        if float(data['chol']) > 240:
            risk_factors.append('High cholesterol')
        if float(data['thalach']) < 120:
            risk_factors.append('Low maximum heart rate')
        if float(data['cp']) > 0:
            risk_factors.append('Chest pain symptoms')
        if float(data['ca']) > 0:
            risk_factors.append('Vessel abnormalities detected')
        
        if prediction == 'High Risk':
            recommendation = {
                'level': 'high' if probability > 0.7 else 'moderate',
                'message': 'Your risk factors suggest elevated heart disease risk.',
                'actions': [
                    'Schedule an appointment with a cardiologist',
                    'Get a comprehensive cardiac evaluation',
                    'Monitor blood pressure regularly',
                    'Consider lifestyle modifications',
                    'Discuss medication options with your doctor'
                ],
                'risk_factors': risk_factors
            }
        else:
            recommendation = {
                'level': 'low',
                'message': 'Your risk factors suggest relatively low heart disease risk.',
                'actions': [
                    'Continue healthy lifestyle habits',
                    'Exercise regularly (150 min/week)',
                    'Maintain healthy diet',
                    'Annual health checkups',
                    'Monitor blood pressure periodically'
                ],
                'risk_factors': risk_factors
            }
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'recommendation': recommendation,
            'features_analyzed': required_features
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid feature values: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Heart prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==============================================================================
#                     IMAGE ANALYSIS ENDPOINTS
# ==============================================================================

@app.route('/analyze/skin', methods=['POST', 'OPTIONS'])
@handle_errors
def analyze_skin():
    """Analyze skin image for cancer detection - FIXED"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if not PIL_AVAILABLE:
        return jsonify({'success': False, 'error': 'PIL not available'}), 500
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'}), 400
    
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess for skin model (RGB)
        processed_image = preprocess_image_for_skin(image, target_size=(224, 224))
        
        # Validate image type
        validation = validate_image_type(processed_image, 'skin')
        if not validation['is_valid']:
            return jsonify({
                'success': False,
                'error': 'Invalid image type',
                'validation_error': True,
                'message': validation['message'],
                'suggestion': validation.get('suggestion', 'Please upload a valid skin lesion image.'),
                'expected_type': 'Skin lesion photo (color)',
                'image_size': f"{image.size[0]}x{image.size[1]}"
            }), 400
        
        # Check if model is loaded
        if 'skin_cancer' not in models:
            # Demo mode
            demo_idx = np.random.choice([2, 3, 5, 6], p=[0.3, 0.2, 0.4, 0.1])
            class_info = SKIN_CANCER_CLASSES[demo_idx]
            confidence = np.random.uniform(0.65, 0.90)
            
            return jsonify({
                'success': True,
                'demo_mode': True,
                'prediction': {
                    'name': class_info['name'],
                    'code': class_info['code'],
                    'type': class_info['type'],
                    'confidence': float(confidence),
                    'confidence_percent': f"{confidence * 100:.1f}%"
                },
                'severity': class_info['severity'],
                'staging': get_stage_info('skin', class_info, confidence),
                'recommendations': get_skin_recommendations(class_info, confidence),
                'treatment_options': get_treatment_options('skin', class_info, {}),
                'urgency': get_urgency_timeline(class_info['severity']),
                'note': 'Demo mode - train model for real predictions'
            })
        
        # Real prediction
        predictions = models['skin_cancer'].predict(processed_image, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        class_info = SKIN_CANCER_CLASSES.get(predicted_idx, SKIN_CANCER_CLASSES[5])
        
        # All predictions
        all_predictions = []
        for idx, prob in enumerate(predictions):
            info = SKIN_CANCER_CLASSES.get(idx, SKIN_CANCER_CLASSES[5])
            all_predictions.append({
                'name': info['name'],
                'code': info['code'],
                'type': info['type'],
                'confidence': float(prob),
                'severity': info['severity']
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        stage_info = get_stage_info('skin', class_info, confidence)
        
        return jsonify({
            'success': True,
            'prediction': {
                'name': class_info['name'],
                'code': class_info['code'],
                'type': class_info['type'],
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%"
            },
            'severity': class_info['severity'],
            'all_predictions': all_predictions[:5],
            'staging': stage_info,
            'recommendations': get_skin_recommendations(class_info, confidence),
            'treatment_options': get_treatment_options('skin', class_info, stage_info),
            'urgency': get_urgency_timeline(class_info['severity'])
        })
        
    except Exception as e:
        logger.error(f"Skin analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze/xray', methods=['POST', 'OPTIONS'])
@handle_errors
def analyze_xray():
    """Analyze chest X-ray for pneumonia - FIXED for 2-class output"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if not PIL_AVAILABLE:
        return jsonify({'success': False, 'error': 'PIL not available'}), 500
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess for X-ray model (Grayscale)
        processed_image = preprocess_image_for_xray(image, target_size=(224, 224))
        
        # Validate image type
        validation = validate_image_type(processed_image, 'xray')
        if not validation['is_valid']:
            return jsonify({
                'success': False,
                'error': 'Invalid image type',
                'validation_error': True,
                'message': validation['message'],
                'suggestion': validation.get('suggestion', 'Please upload a valid chest X-ray image.'),
                'expected_type': 'Chest X-ray (grayscale)',
                'image_size': f"{image.size[0]}x{image.size[1]}"
            }), 400
        
        if 'pneumonia' not in models:
            # Demo mode
            demo_idx = np.random.choice([0, 1], p=[0.6, 0.4])
            class_info = PNEUMONIA_CLASSES[demo_idx]
            confidence = np.random.uniform(0.70, 0.92)
            
            stage_info = get_stage_info('pneumonia', class_info, confidence)
            
            return jsonify({
                'success': True,
                'demo_mode': True,
                'prediction': {
                    'name': class_info['name'],
                    'code': class_info['code'],
                    'confidence': float(confidence),
                    'confidence_percent': f"{confidence * 100:.1f}%"
                },
                'severity': class_info['severity'],
                'staging': stage_info,
                'recommendations': get_pneumonia_recommendations(class_info, confidence),
                'treatment_options': get_treatment_options('pneumonia', class_info, stage_info),
                'urgency': get_urgency_timeline(class_info['severity']),
                'note': 'Demo mode - train model for real predictions'
            })
        
        # Real prediction - Handle 2-class output
        predictions = models['pneumonia'].predict(processed_image, verbose=0)[0]
        
        # Check output format
        if len(predictions) == 2:
            # 2-class softmax output [normal_prob, pneumonia_prob]
            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])
            
            all_predictions = [
                {'name': 'Normal', 'code': 'normal', 'confidence': float(predictions[0]), 'severity': 'healthy'},
                {'name': 'Pneumonia', 'code': 'pneumonia', 'confidence': float(predictions[1]), 'severity': 'high'}
            ]
        elif len(predictions) == 1 or isinstance(predictions, (int, float)):
            # Binary sigmoid output
            prob = float(predictions[0]) if hasattr(predictions, '__len__') else float(predictions)
            predicted_idx = 1 if prob > 0.5 else 0
            confidence = prob if prob > 0.5 else (1 - prob)
            
            all_predictions = [
                {'name': 'Normal', 'code': 'normal', 'confidence': float(1 - prob), 'severity': 'healthy'},
                {'name': 'Pneumonia', 'code': 'pneumonia', 'confidence': float(prob), 'severity': 'high'}
            ]
        else:
            # Multi-class (shouldn't happen with new model)
            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])
            all_predictions = []
        
        class_info = PNEUMONIA_CLASSES.get(predicted_idx, PNEUMONIA_CLASSES[0])
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        stage_info = get_stage_info('pneumonia', class_info, confidence)
        
        return jsonify({
            'success': True,
            'prediction': {
                'name': class_info['name'],
                'code': class_info['code'],
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%"
            },
            'severity': class_info['severity'],
            'all_predictions': all_predictions,
            'staging': stage_info,
            'recommendations': get_pneumonia_recommendations(class_info, confidence),
            'treatment_options': get_treatment_options('pneumonia', class_info, stage_info),
            'urgency': get_urgency_timeline(class_info['severity'])
        })
        
    except Exception as e:
        logger.error(f"X-ray analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze/breast', methods=['POST', 'OPTIONS'])
@handle_errors
def analyze_breast():
    """Analyze mammogram/ultrasound for breast cancer - FIXED for 3-class output"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if not PIL_AVAILABLE:
        return jsonify({'success': False, 'error': 'PIL not available'}), 500
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess for breast model (Grayscale)
        processed_image = preprocess_image_for_breast(image, target_size=(224, 224))
        
        # Validate image type
        validation = validate_image_type(processed_image, 'breast')
        if not validation['is_valid']:
            return jsonify({
                'success': False,
                'error': 'Invalid image type',
                'validation_error': True,
                'message': validation['message'],
                'suggestion': validation.get('suggestion', 'Please upload a mammogram or breast ultrasound.'),
                'expected_type': 'Mammogram or breast ultrasound',
                'image_size': f"{image.size[0]}x{image.size[1]}"
            }), 400
        
        if 'breast_cancer' not in models:
            # Demo mode
            demo_idx = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            class_info = BREAST_CANCER_CLASSES_3[demo_idx]
            confidence = np.random.uniform(0.70, 0.90)
            
            stage_info = get_stage_info('breast', class_info, confidence)
            
            return jsonify({
                'success': True,
                'demo_mode': True,
                'prediction': {
                    'name': class_info['name'],
                    'code': class_info['code'],
                    'birads': class_info.get('birads', ''),
                    'confidence': float(confidence),
                    'confidence_percent': f"{confidence * 100:.1f}%"
                },
                'severity': class_info['severity'],
                'staging': stage_info,
                'recommendations': get_breast_recommendations(class_info, confidence),
                'treatment_options': get_treatment_options('breast', class_info, stage_info),
                'urgency': get_urgency_timeline(class_info['severity']),
                'note': 'Demo mode - train model for real predictions'
            })
        
        # Real prediction
        predictions = models['breast_cancer'].predict(processed_image, verbose=0)[0]
        num_classes = len(predictions)
        
        # Use appropriate class definitions
        if num_classes == 3:
            class_definitions = BREAST_CANCER_CLASSES_3
        else:
            class_definitions = BREAST_CANCER_CLASSES_6
        
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        class_info = class_definitions.get(predicted_idx, class_definitions[0])
        
        # All predictions
        all_predictions = []
        for idx, prob in enumerate(predictions):
            info = class_definitions.get(idx, class_definitions[0])
            all_predictions.append({
                'name': info['name'],
                'code': info['code'],
                'birads': info.get('birads', ''),
                'confidence': float(prob),
                'severity': info['severity']
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        stage_info = get_stage_info('breast', class_info, confidence)
        
        return jsonify({
            'success': True,
            'prediction': {
                'name': class_info['name'],
                'code': class_info['code'],
                'birads': class_info.get('birads', ''),
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%"
            },
            'severity': class_info['severity'],
            'all_predictions': all_predictions,
            'staging': stage_info,
            'recommendations': get_breast_recommendations(class_info, confidence),
            'treatment_options': get_treatment_options('breast', class_info, stage_info),
            'urgency': get_urgency_timeline(class_info['severity'])
        })
        
    except Exception as e:
        logger.error(f"Breast analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze/heart', methods=['POST', 'OPTIONS'])
@handle_errors
def analyze_heart():
    """Analyze ECG/heart scan image - FIXED for 5-class output"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if not PIL_AVAILABLE:
        return jsonify({'success': False, 'error': 'PIL not available'}), 500
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess for heart model (Grayscale)
        processed_image = preprocess_image_for_heart(image, target_size=(224, 224))
        
        # Validate image type
        validation = validate_image_type(processed_image, 'heart')
        if not validation['is_valid']:
            return jsonify({
                'success': False,
                'error': 'Invalid image type',
                'validation_error': True,
                'message': validation['message'],
                'suggestion': validation.get('suggestion', 'Please upload an ECG or heart scan image.'),
                'expected_type': 'ECG printout or heart scan',
                'image_size': f"{image.size[0]}x{image.size[1]}"
            }), 400
        
        if 'heart_image' not in models:
            # Demo mode
            demo_idx = np.random.choice([0, 0, 2, 4], p=[0.5, 0.2, 0.2, 0.1])
            class_info = HEART_CONDITIONS[demo_idx]
            confidence = np.random.uniform(0.70, 0.92)
            
            stage_info = get_stage_info('heart', class_info, confidence)
            
            return jsonify({
                'success': True,
                'demo_mode': True,
                'prediction': {
                    'name': class_info['name'],
                    'code': class_info['code'],
                    'confidence': float(confidence),
                    'confidence_percent': f"{confidence * 100:.1f}%"
                },
                'severity': class_info['severity'],
                'staging': stage_info,
                'recommendations': get_heart_recommendations(class_info, confidence),
                'treatment_options': get_treatment_options('heart', class_info, stage_info),
                'urgency': get_urgency_timeline(class_info['severity']),
                'note': 'Demo mode - train model for real predictions'
            })
        
        # Real prediction
        predictions = models['heart_image'].predict(processed_image, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        class_info = HEART_CONDITIONS.get(predicted_idx, HEART_CONDITIONS[0])
        
        # All predictions
        all_predictions = []
        for idx, prob in enumerate(predictions):
            info = HEART_CONDITIONS.get(idx, HEART_CONDITIONS[0])
            all_predictions.append({
                'name': info['name'],
                'code': info['code'],
                'confidence': float(prob),
                'severity': info['severity']
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        stage_info = get_stage_info('heart', class_info, confidence)
        
        return jsonify({
            'success': True,
            'prediction': {
                'name': class_info['name'],
                'code': class_info['code'],
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%"
            },
            'severity': class_info['severity'],
            'all_predictions': all_predictions,
            'staging': stage_info,
            'recommendations': get_heart_recommendations(class_info, confidence),
            'treatment_options': get_treatment_options('heart', class_info, stage_info),
            'urgency': get_urgency_timeline(class_info['severity'])
        })
        
    except Exception as e:
        logger.error(f"Heart analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==============================================================================
#                           ERROR HANDLERS
# ==============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on this server.',
        'available_endpoints': [
            'GET /', 'GET /health', 'GET /symptoms',
            'POST /analyze/skin', 'POST /analyze/heart',
            'POST /analyze/breast', 'POST /analyze/xray',
            'POST /predict-disease', 'POST /predict-heart',
            'POST /predict-cancer'
        ]
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again.'
    }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': f'Maximum file size is {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB'
    }), 413


@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': str(e)
    }), 400


# ==============================================================================
#                           MAIN ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🏥 MediDiagnose-AI Backend Server v4.0 (COMPLETE & FIXED)")
    print("   AI-Powered Medical Diagnosis System")
    print("=" * 60)
    print("\n📋 Available Endpoints:")
    print("-" * 60)
    print("  GET  /                  - API info and documentation")
    print("  GET  /health            - Health check and model status")
    print("  GET  /symptoms          - Get symptoms list")
    print("-" * 60)
    print("  POST /analyze/skin      - Skin cancer detection (image)")
    print("  POST /analyze/heart     - Heart condition (ECG image)")
    print("  POST /analyze/breast    - Breast cancer (mammogram)")
    print("  POST /analyze/xray      - Pneumonia (chest X-ray)")
    print("-" * 60)
    print("  POST /predict-disease   - Disease from symptoms")
    print("  POST /predict-heart     - Heart risk from factors")
    print("  POST /predict-cancer    - Breast cancer from tumor data")
    print("=" * 60)
    
    # Load models
    load_models()
    
    print("\n🚀 Server starting on http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run the Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False,
        threaded=True
    )
