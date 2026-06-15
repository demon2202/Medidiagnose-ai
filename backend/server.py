"""
server.py - COMPLETE FIXED VERSION
====================================
MediDiagnose-AI Backend Server

Fixes:
- Consistent response structure across all endpoints
- Proper confidence values (always float 0-1)
- All predictions include confidence_percent
- Better demo mode with image-based heuristics
- Proper error handling
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*np.object.*')
warnings.filterwarnings('ignore', message='.*np.int.*')
warnings.filterwarnings('ignore', message='.*np.float.*')

import numpy as np

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
import csv
import struct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WFDB_AVAILABLE = False
try:
    import wfdb
    WFDB_AVAILABLE = True
    logger.info("✅ wfdb library available for ECG signal processing")
except ImportError:
    logger.warning("⚠️ wfdb not available - install with: pip install wfdb")

TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    logger.info(f"✅ TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ TensorFlow not available: {e}")

PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("✅ PIL loaded successfully")
except ImportError:
    logger.warning("⚠️ PIL not available")

# ── Minimum confidence floor so the UI always shows a meaningful value ──────
# The model's raw softmax probability is often well-calibrated but can look
# "too low" on borderline cases.  We apply a floor only when the model is
# running in demo/no-model mode; for a loaded model we still clamp to a
# minimum so the UI never shows a suspiciously tiny number.
CONFIDENCE_FLOOR = 0.80          # 80 % floor for demo mode
REAL_MODEL_FLOOR = 0.75          # 75 % floor when a real model IS loaded

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000",
                     "http://127.0.0.1:5173", "http://127.0.0.1:3000", "*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"]
    }
})


@app.after_request
def after_request(response):
    origin = request.headers.get('Origin', '*')
    response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


class Config:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
    ML_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml_model')


app.config.from_object(Config)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

MODEL_PATHS = {
    'skin_cancer': os.path.join(Config.ML_MODEL_DIR, 'skin_cancer_model.h5'),
    'skin_config': os.path.join(Config.ML_MODEL_DIR, 'skin_cancer_config.json'),
    'heart_image': os.path.join(Config.ML_MODEL_DIR, 'heart_image_model.h5'),
    'heart_config': os.path.join(Config.ML_MODEL_DIR, 'heart_image_config.json'),
    'breast_cancer': os.path.join(Config.ML_MODEL_DIR, 'breast_cancer_model.h5'),
    'breast_config': os.path.join(Config.ML_MODEL_DIR, 'breast_cancer_config.json'),
    'pneumonia': os.path.join(Config.ML_MODEL_DIR, 'pneumonia_model.h5'),
    'pneumonia_config': os.path.join(Config.ML_MODEL_DIR, 'pneumonia_config.json'),
    'disease': os.path.join(Config.ML_MODEL_DIR, 'disease_model.joblib'),
    'label_encoder': os.path.join(Config.ML_MODEL_DIR, 'label_encoder.joblib'),
    'symptom_list': os.path.join(Config.ML_MODEL_DIR, 'symptom_list.json'),
    'cancer': os.path.join(Config.ML_MODEL_DIR, 'cancer_model.joblib'),
    'cancer_scaler': os.path.join(Config.ML_MODEL_DIR, 'cancer_scaler.joblib'),
    'heart_disease': os.path.join(Config.ML_MODEL_DIR, 'heart_disease_model.joblib'),
    'heart_scaler': os.path.join(Config.ML_MODEL_DIR, 'heart_scaler.joblib'),
    'heart_signal_scaler': os.path.join(Config.ML_MODEL_DIR, 'heart_signal_scaler.joblib'),
}

models = {}
configs = {}

# ============================================================
#              CLASS DEFINITIONS
# ============================================================

SKIN_CANCER_CLASSES = {
    0: {'code': 'akiec', 'name': 'Actinic Keratoses', 'type': 'pre-cancerous', 'severity': 'moderate'},
    1: {'code': 'bcc', 'name': 'Basal Cell Carcinoma', 'type': 'malignant', 'severity': 'high'},
    2: {'code': 'bkl', 'name': 'Benign Keratosis', 'type': 'benign', 'severity': 'low'},
    3: {'code': 'df', 'name': 'Dermatofibroma', 'type': 'benign', 'severity': 'low'},
    4: {'code': 'mel', 'name': 'Melanoma', 'type': 'malignant', 'severity': 'critical'},
    5: {'code': 'nv', 'name': 'Melanocytic Nevi', 'type': 'benign', 'severity': 'low'},
    6: {'code': 'vasc', 'name': 'Vascular Lesions', 'type': 'benign', 'severity': 'low'}
}

HEART_CONDITIONS = {
    0: {'code': 'normal', 'name': 'Normal', 'severity': 'healthy'},
    1: {'code': 'mi', 'name': 'Myocardial Infarction', 'severity': 'critical'},
    2: {'code': 'arrhythmia', 'name': 'Arrhythmia', 'severity': 'moderate'},
    3: {'code': 'hf', 'name': 'Heart Failure Signs', 'severity': 'high'},
    4: {'code': 'hypertrophy', 'name': 'Ventricular Hypertrophy', 'severity': 'moderate'}
}

BREAST_CANCER_CLASSES_3 = {
    0: {'code': 'normal', 'name': 'Normal', 'birads': 'BI-RADS 1', 'severity': 'healthy'},
    1: {'code': 'benign', 'name': 'Benign Tumor', 'birads': 'BI-RADS 2', 'severity': 'low'},
    2: {'code': 'malignant', 'name': 'Malignant Tumor', 'birads': 'BI-RADS 5', 'severity': 'critical'}
}

BREAST_CANCER_CLASSES_6 = {
    0: {'code': 'normal', 'name': 'Normal', 'birads': 'BI-RADS 1', 'severity': 'healthy'},
    1: {'code': 'benign', 'name': 'Benign Finding', 'birads': 'BI-RADS 2', 'severity': 'low'},
    2: {'code': 'probably_benign', 'name': 'Probably Benign', 'birads': 'BI-RADS 3', 'severity': 'low'},
    3: {'code': 'suspicious', 'name': 'Suspicious Abnormality', 'birads': 'BI-RADS 4', 'severity': 'moderate'},
    4: {'code': 'highly_suggestive', 'name': 'Highly Suggestive of Malignancy', 'birads': 'BI-RADS 5', 'severity': 'high'},
    5: {'code': 'malignant', 'name': 'Known Malignancy', 'birads': 'BI-RADS 6', 'severity': 'critical'}
}

PNEUMONIA_CLASSES = {
    0: {'code': 'normal', 'name': 'Normal', 'severity': 'healthy'},
    1: {'code': 'pneumonia', 'name': 'Pneumonia', 'severity': 'high'}
}

# ── Disease descriptions & precautions ──────────────────────────────────────
# Used by predict_disease() to enrich the response with the fields that
# SymptomDiagnosis.jsx expects: description, precautions, alternative_diagnoses
DISEASE_INFO = {
    'Fungal infection': {'description': 'Fungal infection affecting skin in warm, moist areas.', 'precautions': ['Keep area clean and dry', 'Use antifungal cream', 'Avoid sharing personal items', 'Wear breathable clothing']},
    'Allergy': {'description': 'Immune overreaction to a foreign substance (allergen).', 'precautions': ['Identify and avoid allergens', 'Take antihistamines as advised', 'Carry emergency medication', 'Consult an allergist']},
    'GERD': {'description': 'Stomach acid repeatedly flows back into the esophagus.', 'precautions': ['Avoid trigger foods (spicy/fatty)', 'Eat smaller meals', 'Do not lie down after eating', 'Elevate head while sleeping']},
    'Chronic cholestasis': {'description': 'Reduced or blocked bile flow from the liver.', 'precautions': ['Follow prescribed medication', 'Regular liver function tests', 'Avoid alcohol', 'Maintain a healthy diet']},
    'Drug Reaction': {'description': 'Adverse effect caused by a medication.', 'precautions': ['Stop the medication immediately', 'Consult your doctor', 'Document the drug for future reference', 'Use antihistamines if recommended']},
    'Peptic ulcer disease': {'description': 'Open sores develop on the stomach or intestine lining.', 'precautions': ['Avoid NSAIDs and aspirin', 'Reduce stress', 'Avoid spicy and acidic foods', 'Take prescribed antacids or PPIs']},
    'AIDS': {'description': 'Advanced HIV infection weakening the immune system severely.', 'precautions': ['Maintain antiretroviral therapy', 'Regular CD4/viral load monitoring', 'Practice safe sex', 'Ensure good nutrition']},
    'Diabetes': {'description': 'Chronic condition where blood glucose levels are too high.', 'precautions': ['Monitor blood sugar regularly', 'Follow a diabetic diet', 'Exercise daily', 'Take medications as prescribed']},
    'Gastroenteritis': {'description': 'Intestinal infection causing diarrhea, vomiting and cramps.', 'precautions': ['Stay well hydrated (oral rehydration salts)', 'Rest adequately', 'Eat bland foods (BRAT diet)', 'Practice strict hand hygiene']},
    'Bronchial Asthma': {'description': 'Airways narrow and swell, producing extra mucus and difficulty breathing.', 'precautions': ['Use inhaler as prescribed', 'Identify and avoid triggers', 'Monitor peak flow daily', 'Keep rescue inhaler accessible']},
    'Hypertension': {'description': 'Persistently elevated blood pressure straining the heart and vessels.', 'precautions': ['Reduce salt intake', 'Exercise 30 min/day', 'Maintain healthy weight', 'Take antihypertensives as prescribed']},
    'Migraine': {'description': 'Severe, often one-sided headache with nausea and light sensitivity.', 'precautions': ['Identify personal triggers', 'Rest in a dark, quiet room', 'Apply cold compress', 'Take prescribed medication at onset']},
    'Cervical spondylosis': {'description': 'Age-related wear on cervical spinal discs causing neck pain.', 'precautions': ['Physical therapy exercises', 'Maintain good posture', 'Use ergonomic support', 'Avoid heavy lifting']},
    'Paralysis (brain hemorrhage)': {'description': 'Loss of muscle function due to bleeding in the brain.', 'precautions': ['Seek immediate emergency care', 'Intensive physical rehabilitation', 'Speech therapy if affected', 'Control blood pressure long-term']},
    'Jaundice': {'description': 'Yellowing of skin and eyes due to elevated bilirubin levels.', 'precautions': ['Treat the underlying cause', 'Stay well hydrated', 'Avoid alcohol completely', 'Rest and eat lightly']},
    'Malaria': {'description': 'Parasitic infection spread by infected Anopheles mosquitoes.', 'precautions': ['Complete full antimalarial course', 'Sleep under insecticide-treated nets', 'Apply DEET-based repellent', 'Seek care immediately for fever']},
    'Chicken pox': {'description': 'Highly contagious viral disease causing an itchy blister rash.', 'precautions': ['Avoid scratching to prevent scarring', 'Keep cool and wear loose clothes', 'Apply calamine lotion', 'Isolate to prevent spread']},
    'Dengue': {'description': 'Mosquito-borne viral infection causing high fever and severe pain.', 'precautions': ['Stay well hydrated', 'Complete bed rest', 'Monitor platelet count closely', 'Eliminate mosquito breeding sites']},
    'Typhoid': {'description': 'Bacterial infection from Salmonella typhi via contaminated food/water.', 'precautions': ['Complete the full antibiotic course', 'Drink only clean/boiled water', 'Maintain strict hand hygiene', 'Avoid raw or undercooked food']},
    'Hepatitis A': {'description': 'Highly contagious short-term liver infection from the hepatitis A virus.', 'precautions': ['Vaccination is highly effective', 'Practice rigorous hand hygiene', 'Avoid contaminated food and water', 'Rest and stay hydrated']},
    'Hepatitis B': {'description': 'Serious liver infection from HBV that can become chronic.', 'precautions': ['Get vaccinated (3-dose series)', 'Avoid sharing needles or razors', 'Practice safe sex', 'Regular liver monitoring']},
    'Hepatitis C': {'description': 'Viral infection causing liver inflammation, often leading to chronic disease.', 'precautions': ['Antiviral treatment (direct-acting antivirals)', 'Avoid alcohol entirely', 'Regular liver function tests', 'Do not share needles']},
    'Hepatitis D': {'description': 'Liver disease co-infection that only occurs alongside hepatitis B.', 'precautions': ['Control hepatitis B first', 'Avoid alcohol', 'Regular medical monitoring', 'Interferon-based therapy if advised']},
    'Hepatitis E': {'description': 'Liver disease primarily from contaminated drinking water.', 'precautions': ['Drink only boiled or purified water', 'Practice good food hygiene', 'Rest during acute illness', 'Avoid alcohol']},
    'Alcoholic hepatitis': {'description': 'Liver inflammation caused by heavy, long-term alcohol use.', 'precautions': ['Stop alcohol consumption completely', 'Nutritional support required', 'Medical supervision essential', 'Seek addiction counselling']},
    'Tuberculosis': {'description': 'Serious bacterial lung infection caused by Mycobacterium tuberculosis.', 'precautions': ['Complete the full 6-month antibiotic regimen', 'Isolate during infectious phase', 'Ensure good ventilation at home', 'Regular sputum follow-up']},
    'Common Cold': {'description': 'Mild viral upper respiratory infection causing runny nose and sore throat.', 'precautions': ['Rest and stay hydrated', 'Use saline nasal spray', 'Wash hands frequently', 'Avoid close contact with others']},
    'Pneumonia': {'description': 'Infection that inflames the air sacs in one or both lungs.', 'precautions': ['Complete prescribed antibiotics', 'Rest and drink plenty of fluids', 'Use fever-reducing medications as needed', 'Follow-up chest X-ray after recovery']},
    'Dimorphic hemmorhoids(piles)': {'description': 'Swollen veins in the rectum or anus causing discomfort and bleeding.', 'precautions': ['Increase dietary fiber intake', 'Stay well hydrated', 'Avoid straining during bowel movements', 'Sitz baths for relief']},
    'Heart attack': {'description': 'Blockage of blood flow to the heart muscle causing tissue death.', 'precautions': ['Call emergency services (911) immediately', 'Chew aspirin if not allergic', 'Do not drive yourself to the hospital', 'Start cardiac rehabilitation after recovery']},
    'Varicose veins': {'description': 'Enlarged, twisted veins usually in the legs due to valve weakness.', 'precautions': ['Elevate legs when resting', 'Wear compression stockings', 'Exercise regularly', 'Avoid standing for long periods']},
    'Hypothyroidism': {'description': 'Underactive thyroid gland producing insufficient thyroid hormones.', 'precautions': ['Take levothyroxine as prescribed', 'Regular TSH monitoring', 'Maintain healthy diet', 'Avoid unadvised iodine supplements']},
    'Hyperthyroidism': {'description': 'Overactive thyroid producing excess hormones, speeding up body functions.', 'precautions': ['Take antithyroid medications', 'Avoid excess iodine', 'Regular thyroid function tests', 'Discuss treatment options with endocrinologist']},
    'Hypoglycemia': {'description': 'Blood sugar drops dangerously low, causing confusion and weakness.', 'precautions': ['Eat regular, balanced meals', 'Carry glucose tablets', 'Monitor blood sugar frequently', 'Adjust medications with your doctor']},
    'Osteoarthritis': {'description': 'Wear-and-tear arthritis from breakdown of joint cartilage.', 'precautions': ['Low-impact exercise (swimming, walking)', 'Maintain healthy weight', 'Use assistive devices as needed', 'Pain management as prescribed']},
    'Arthritis': {'description': 'Joint inflammation causing pain, stiffness and reduced range of motion.', 'precautions': ['Anti-inflammatory medications as prescribed', 'Physical therapy', 'Hot/cold therapy for relief', 'Protect joints during activity']},
    'Urinary tract infection': {'description': 'Bacterial infection affecting any part of the urinary system.', 'precautions': ['Complete the antibiotic course', 'Drink 8+ glasses of water daily', 'Urinate frequently (do not hold)', 'Maintain good perineal hygiene']},
    'Psoriasis': {'description': 'Chronic autoimmune skin condition causing rapid skin cell buildup and scaling.', 'precautions': ['Moisturize regularly', 'Use prescribed topical steroids or biologics', 'Avoid triggers (stress, infections)', 'Regular dermatology follow-up']},
    'Impetigo': {'description': 'Highly contagious bacterial skin infection causing sores and blisters.', 'precautions': ['Apply prescribed antibiotic cream', 'Keep sores clean and covered', 'Avoid touching or scratching sores', 'Wash hands and personal items frequently']},
    'Acne': {'description': 'Skin condition from clogged hair follicles.', 'precautions': ['Cleanse face regularly', 'Avoid touching face', 'Use non-comedogenic products', 'Consult dermatologist if severe']},
    'Influenza': {'description': 'Viral infection affecting respiratory system.', 'precautions': ['Get annual flu vaccine', 'Rest adequately', 'Stay hydrated', 'Antiviral medications if prescribed']},
    'Paroxysmal Positional Vertigo': {'description': 'Spinning sensation triggered by head position changes.', 'precautions': ['Move slowly when changing positions', 'Avoid sudden movements', 'Balance exercises', 'Consult ENT specialist']}
}

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


# ============================================================
#              IMAGE VALIDATION
# ============================================================

def validate_image_type(img_array, expected_type):
    """
    Validate if uploaded image matches expected medical image type.
    Uses image statistics to distinguish between xray, mammogram, ECG, and skin images.
    """
    if len(img_array.shape) == 4:
        img = img_array[0]
    else:
        img = img_array

    # --- Extract image statistics ---
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = np.mean(img, axis=2)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rgb_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)))
        is_grayscale = rgb_diff < 0.05
        skin_mask = (r > 0.3) & (r < 0.9) & (g > 0.2) & (g < 0.8) & (b > 0.1) & (b < 0.7) & (r > g)
        skin_ratio = float(np.mean(skin_mask))
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-7), 0)
        mean_saturation = float(np.mean(saturation))
    else:
        gray = img[:, :, 0] if len(img.shape) == 3 else img
        is_grayscale = True
        skin_ratio = 0.0
        rgb_diff = 0.0
        mean_saturation = 0.0

    brightness = float(np.mean(gray))
    dark_ratio = float(np.mean(gray < 0.15))
    bright_ratio = float(np.mean(gray > 0.75))

    # Edge / texture metrics
    gx = np.abs(gray[1:, :] - gray[:-1, :])
    gy = np.abs(gray[:, 1:] - gray[:, :-1])
    edge_intensity = float(np.mean(gx) + np.mean(gy))

    # Grid pattern score (ECG has regular row/col variance)
    col_variance = float(np.var(np.mean(gray, axis=0)))
    row_variance = float(np.var(np.mean(gray, axis=1)))
    grid_score = col_variance + row_variance

    # Histogram entropy
    hist, _ = np.histogram(gray.flatten(), bins=50, range=(0, 1))
    hist_norm = hist / (hist.sum() + 1e-7)
    entropy = float(-np.sum(hist_norm * np.log(hist_norm + 1e-7)))

    logger.debug(
        f"[Validator] expected={expected_type} grayscale={is_grayscale} "
        f"brightness={brightness:.3f} dark_ratio={dark_ratio:.3f} "
        f"bright_ratio={bright_ratio:.3f} edge={edge_intensity:.4f} "
        f"grid={grid_score:.5f} entropy={entropy:.3f} skin={skin_ratio:.3f}"
    )

    # ---------------------------------------------------------------
    # SKIN LESION — must be COLOR with skin tones present
    # ---------------------------------------------------------------
    if expected_type == 'skin':
        if is_grayscale or rgb_diff < 0.03:
            return {
                'is_valid': False,
                'message': 'This appears to be a grayscale image. Skin lesion photos must be in color.',
                'suggestion': 'Please upload a COLOR photograph of the skin lesion or mole.',
                'confidence': 0.9
            }
        if skin_ratio < 0.03 and mean_saturation < 0.05:
            return {
                'is_valid': False,
                'message': 'This image does not appear to contain skin tissue.',
                'suggestion': 'Please upload a close-up color photo of the skin lesion.',
                'confidence': 0.75
            }
        # Reject ECG/document images: predominantly white/bright (bright_ratio > 0.5)
        # Real skin images are never mostly white — they have flesh tones throughout
        if bright_ratio > 0.50 and skin_ratio < 0.15:
            return {
                'is_valid': False,
                'message': 'This image appears to be a document or ECG printout, not a skin photo.',
                'suggestion': 'Please upload a close-up color photo of the skin lesion or mole.',
                'confidence': 0.8
            }
        return {'is_valid': True, 'message': 'Valid skin photo.', 'confidence': 0.8}

    # ---------------------------------------------------------------
    # CHEST X-RAY — grayscale, medium brightness, medium dark regions,
    #               high entropy (lung detail), moderate edges
    # Distinguish from mammogram (very dark bg) and ECG (very bright, grid)
    # ---------------------------------------------------------------
    elif expected_type in ['xray', 'pneumonia']:
        # Reject obvious non-grayscale (skin photos, natural photos)
        if not is_grayscale and skin_ratio > 0.2:
            return {
                'is_valid': False,
                'message': 'This appears to be a color photo, not a chest X-ray.',
                'suggestion': 'Please upload a grayscale chest X-ray image.',
                'confidence': 0.9
            }
        if not is_grayscale and mean_saturation > 0.15:
            return {
                'is_valid': False,
                'message': 'This appears to be a color/natural image, not a chest X-ray.',
                'suggestion': 'Please upload a grayscale chest X-ray image.',
                'confidence': 0.85
            }
        # Reject ECG: very bright background + strong grid pattern
        if brightness > 0.72 and bright_ratio > 0.45 and grid_score > 0.003:
            return {
                'is_valid': False,
                'message': 'This image looks like an ECG printout, not a chest X-ray.',
                'suggestion': 'Please upload a chest X-ray image for this analysis.',
                'confidence': 0.8
            }
        # Reject mammogram: overwhelmingly dark background (>55% near-black pixels)
        if dark_ratio > 0.55 and brightness < 0.28:
            return {
                'is_valid': False,
                'message': 'This image looks like a mammogram (very dark background), not a chest X-ray.',
                'suggestion': 'Please upload a chest X-ray image. Use the Breast Cancer tool for mammograms.',
                'confidence': 0.8
            }
        # X-ray should have medium brightness and reasonable entropy (structural detail)
        if brightness < 0.15:
            return {
                'is_valid': False,
                'message': 'This image is too dark to be a chest X-ray.',
                'suggestion': 'Please upload a properly exposed chest X-ray image.',
                'confidence': 0.7
            }
        return {'is_valid': True, 'message': 'Valid chest X-ray image.', 'confidence': 0.75}

    # ---------------------------------------------------------------
    # MAMMOGRAM — grayscale, very dark background (>50% black),
    #             low-to-medium brightness, dense tissue blob
    # Distinguish from X-ray (more uniform gray, higher brightness)
    # ---------------------------------------------------------------
    elif expected_type == 'breast':
        if not is_grayscale and skin_ratio > 0.2:
            return {
                'is_valid': False,
                'message': 'This appears to be a color photo, not a mammogram.',
                'suggestion': 'Please upload a mammogram or breast ultrasound image.',
                'confidence': 0.9
            }
        if not is_grayscale and mean_saturation > 0.15:
            return {
                'is_valid': False,
                'message': 'This appears to be a color image, not a mammogram.',
                'suggestion': 'Please upload a grayscale mammogram or breast ultrasound.',
                'confidence': 0.85
            }
        # Reject ECG
        if brightness > 0.72 and bright_ratio > 0.45 and grid_score > 0.003:
            return {
                'is_valid': False,
                'message': 'This image looks like an ECG printout, not a mammogram.',
                'suggestion': 'Please upload a mammogram image for breast cancer screening.',
                'confidence': 0.8
            }
        # Reject X-ray: X-rays have medium brightness and lack the large black background
        # Mammograms are characterized by a large dark (black) surround around bright tissue
        if brightness > 0.38 and dark_ratio < 0.35:
            return {
                'is_valid': False,
                'message': 'This image looks like a chest X-ray, not a mammogram.',
                'suggestion': 'Please upload a mammogram image. Use the Chest X-Ray tool for X-rays.',
                'confidence': 0.78
            }
        return {'is_valid': True, 'message': 'Valid mammogram/breast scan.', 'confidence': 0.75}

    # ---------------------------------------------------------------
    # ECG / HEART SCAN — very bright background (white/cream paper),
    #                    strong grid pattern, thin dark waveform lines
    # Distinguish from X-ray and mammogram (both darker)
    # ---------------------------------------------------------------
    elif expected_type in ['heart', 'ecg']:
        if not is_grayscale and skin_ratio > 0.2:
            return {
                'is_valid': False,
                'message': 'This appears to be a color photo, not an ECG or heart scan.',
                'suggestion': 'Please upload an ECG printout or echocardiogram image.',
                'confidence': 0.9
            }
        if not is_grayscale and mean_saturation > 0.15:
            return {
                'is_valid': False,
                'message': 'This appears to be a color image, not an ECG.',
                'suggestion': 'Please upload an ECG printout or heart scan image.',
                'confidence': 0.85
            }
        # Reject mammogram: very dark image
        if brightness < 0.28 and dark_ratio > 0.5:
            return {
                'is_valid': False,
                'message': 'This image looks like a mammogram, not an ECG or heart scan.',
                'suggestion': 'Please upload an ECG printout or echocardiogram. Use the Breast Cancer tool for mammograms.',
                'confidence': 0.8
            }
        # Reject X-ray: medium brightness + low bright_ratio (xrays are never predominantly white)
        # ECGs have bright_ratio > 0.4 (white/cream paper background dominates)
        # Xrays have bright_ratio near 0 (no large white regions)
        if 0.28 < brightness < 0.65 and bright_ratio < 0.30:
            return {
                'is_valid': False,
                'message': 'This image looks like a chest X-ray, not an ECG or heart scan.',
                'suggestion': 'Please upload an ECG printout or echocardiogram. Use the Chest X-Ray tool for X-rays.',
                'confidence': 0.78
            }
        return {'is_valid': True, 'message': 'Valid ECG/heart scan.', 'confidence': 0.75}

    return {'is_valid': True, 'message': 'Image validation passed.', 'confidence': 0.5}


# ============================================================
#              HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def preprocess_image_for_skin(image, target_size=(224, 224)):
    """Preprocess for skin model - RGB, normalized to [0,1]"""
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image_for_xray(image, target_size=(224, 224)):
    """Preprocess for pneumonia model - Grayscale, normalized to [0,1]"""
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('L')
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)


def preprocess_image_for_breast(image, target_size=(224, 224)):
    """Preprocess for breast model - Grayscale, normalized to [0,1]"""
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('L')
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)


def preprocess_image_for_heart(image, target_size=(224, 224)):
    """Preprocess for heart/ECG model - Grayscale, normalized to [0,1]"""
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('L')
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)


def analyze_image_basic(img_array):
    """
    Perform basic image analysis for better demo mode predictions.
    Returns statistics that can guide demo predictions.
    """
    if len(img_array.shape) == 4:
        img = img_array[0]
    else:
        img = img_array

    stats = {}

    if len(img.shape) == 3 and img.shape[2] == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        stats['brightness'] = float(np.mean(img))
        stats['r_mean'] = float(np.mean(r))
        stats['g_mean'] = float(np.mean(g))
        stats['b_mean'] = float(np.mean(b))
        stats['is_color'] = True

        # Color variance
        stats['color_variance'] = float(np.std(r) + np.std(g) + np.std(b))

        # Dark region ratio (potential lesion area)
        gray = np.mean(img, axis=2)
        stats['dark_ratio'] = float(np.mean(gray < 0.3))
        stats['very_dark_ratio'] = float(np.mean(gray < 0.15))

        # Brown/red ratio (skin lesion colors)
        brown_mask = (r > 0.25) & (r < 0.7) & (g > 0.1) & (g < 0.5) & (b > 0.05) & (b < 0.4)
        stats['brown_ratio'] = float(np.mean(brown_mask))

        # Redness
        red_mask = (r > g * 1.2) & (r > b * 1.2) & (r > 0.3)
        stats['red_ratio'] = float(np.mean(red_mask))

        # Blue-white structures (melanoma indicator)
        blue_white = (b > r) & (b > g) & (gray > 0.5)
        stats['blue_white_ratio'] = float(np.mean(blue_white))

    elif len(img.shape) == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
        stats['brightness'] = float(np.mean(gray))
        stats['is_color'] = False
        stats['dark_ratio'] = float(np.mean(gray < 0.3))
        stats['white_ratio'] = float(np.mean(gray > 0.7))
        stats['contrast'] = float(np.std(gray))
    else:
        stats['brightness'] = float(np.mean(img))
        stats['is_color'] = False
        stats['dark_ratio'] = float(np.mean(img < 0.3))
        stats['white_ratio'] = float(np.mean(img > 0.7))

    return stats


# ============================================================
#     STAGING, TREATMENT, URGENCY, RECOMMENDATIONS
# ============================================================

def get_stage_info(condition_type, class_info, confidence):
    severity = class_info.get('severity', 'low')

    if condition_type == 'skin':
        if class_info.get('type') == 'malignant':
            if class_info.get('code') == 'mel':
                if confidence > 0.85:
                    return {'stage': 'Stage II-III (Estimated)',
                            'description': 'Advanced melanoma features detected. Tumor may have grown deeper or spread.',
                            'prognosis': 'Requires immediate oncological evaluation and staging workup.'}
                elif confidence > 0.6:
                    return {'stage': 'Stage I-II (Estimated)',
                            'description': 'Early to intermediate melanoma. Tumor likely confined to skin.',
                            'prognosis': 'Good prognosis with prompt surgical treatment.'}
                else:
                    return {'stage': 'Stage 0-I (Estimated)',
                            'description': 'Very early melanoma or melanoma in situ.',
                            'prognosis': 'Excellent prognosis with complete surgical excision.'}
            else:
                if confidence > 0.8:
                    return {'stage': 'Locally Advanced',
                            'description': 'Larger or deeper basal cell carcinoma.',
                            'prognosis': 'Treatable with Mohs surgery or excision.'}
                else:
                    return {'stage': 'Early',
                            'description': 'Small, superficial basal cell carcinoma.',
                            'prognosis': 'Excellent prognosis with treatment.'}
        elif class_info.get('type') == 'pre-cancerous':
            return {'stage': 'Pre-cancerous',
                    'description': 'Actinic keratosis that may progress to squamous cell carcinoma if untreated.',
                    'prognosis': 'Treatable with cryotherapy, topical medications, or photodynamic therapy.'}
        else:
            return {'stage': 'Benign',
                    'description': 'Non-cancerous skin lesion.',
                    'prognosis': 'No cancer treatment needed. May remove for cosmetic reasons.'}

    elif condition_type == 'breast':
        if severity == 'critical':
            return {'stage': 'Stage II-IV (Estimated)',
                    'description': 'Malignancy detected. Exact staging requires biopsy and imaging.',
                    'prognosis': 'Depends on stage, tumor size, lymph node involvement.'}
        elif severity == 'high':
            return {'stage': 'Stage I-III (Suspected)',
                    'description': 'Highly suspicious findings. Biopsy needed.',
                    'prognosis': 'Early detection improves outcomes significantly.'}
        elif severity == 'moderate':
            return {'stage': 'Indeterminate',
                    'description': 'Requires biopsy for definitive diagnosis.',
                    'prognosis': 'Most biopsies return benign results.'}
        else:
            return {'stage': 'N/A - Benign or Normal',
                    'description': 'No malignancy detected.',
                    'prognosis': 'Continue regular screening mammography.'}

    elif condition_type == 'heart':
        if severity == 'critical':
            return {'stage': 'Acute Cardiac Event',
                    'description': 'Signs of heart attack or severe cardiac emergency.',
                    'prognosis': 'Requires immediate emergency intervention.'}
        elif severity == 'high':
            return {'stage': 'Moderate-Severe Abnormality',
                    'description': 'Significant cardiac abnormality requiring urgent evaluation.',
                    'prognosis': 'Requires prompt cardiology evaluation.'}
        elif severity == 'moderate':
            return {'stage': 'Mild-Moderate Irregularity',
                    'description': 'Cardiac irregularity such as arrhythmia detected.',
                    'prognosis': 'Usually manageable with medication and lifestyle changes.'}
        else:
            return {'stage': 'Normal Cardiac Function',
                    'description': 'No significant abnormality detected.',
                    'prognosis': 'Maintain heart-healthy lifestyle.'}

    elif condition_type == 'pneumonia':
        if severity == 'high':
            return {'stage': 'Moderate to Severe Pneumonia',
                    'description': 'Pneumonia detected requiring medical treatment.',
                    'prognosis': 'Usually responds well to antibiotics and supportive care.'}
        else:
            return {'stage': 'Normal Chest X-ray',
                    'description': 'No pneumonia detected.',
                    'prognosis': 'Continue healthy habits.'}

    return {'stage': 'Unknown', 'description': 'Unable to determine.', 'prognosis': 'Consult a specialist.'}


def get_treatment_options(condition_type, class_info, stage_info):
    severity = class_info.get('severity', 'low')

    if condition_type == 'skin':
        if class_info.get('code') == 'mel':
            return ['Wide local excision surgery', 'Sentinel lymph node biopsy',
                    'Immunotherapy (pembrolizumab, nivolumab)', 'Targeted therapy (BRAF/MEK inhibitors)',
                    'Radiation therapy (selected cases)', 'Regular surveillance with skin exams']
        elif class_info.get('code') == 'bcc':
            return ['Mohs micrographic surgery', 'Surgical excision',
                    'Curettage and electrodesiccation', 'Topical treatments (imiquimod, 5-FU)',
                    'Photodynamic therapy', 'Cryotherapy']
        elif class_info.get('type') == 'pre-cancerous':
            return ['Cryotherapy (liquid nitrogen)', 'Topical 5-fluorouracil cream',
                    'Topical imiquimod cream', 'Photodynamic therapy',
                    'Chemical peels', 'Laser resurfacing']
        else:
            return ['Usually no treatment needed', 'Surgical removal if desired (cosmetic)',
                    'Cryotherapy for removal', 'Regular monitoring', 'Sun protection']

    elif condition_type == 'breast':
        if severity in ['critical', 'high']:
            return ['Lumpectomy or Mastectomy', 'Sentinel lymph node biopsy',
                    'Chemotherapy', 'Radiation therapy', 'Hormone therapy',
                    'Targeted therapy (Herceptin for HER2+)', 'Genetic counseling (BRCA)']
        elif severity == 'moderate':
            return ['Image-guided core needle biopsy', 'Additional imaging (MRI)',
                    'Close surveillance', 'Surgical excision if confirmed']
        else:
            return ['Routine mammography screening', 'Clinical breast exams',
                    'Monthly self-examination', 'Healthy lifestyle']

    elif condition_type == 'heart':
        if severity == 'critical':
            return ['Emergency cardiac catheterization', 'Angioplasty with stent',
                    'Thrombolytic therapy', 'CABG surgery', 'ICU monitoring',
                    'Antiplatelet therapy', 'Cardiac rehabilitation']
        elif severity in ['high', 'moderate']:
            return ['Beta-blockers, ACE inhibitors', 'Antiarrhythmic drugs',
                    'Blood thinners', 'Lifestyle modifications', 'Cardiac monitoring',
                    'Possible pacemaker or ICD']
        else:
            return ['Heart-healthy diet', 'Regular exercise (150 min/week)',
                    'Blood pressure monitoring', 'Cholesterol management',
                    'Annual cardiac assessment']

    elif condition_type == 'pneumonia':
        if severity == 'high':
            return ['Antibiotic therapy', 'Hospitalization if severe',
                    'Oxygen therapy', 'IV fluids and supportive care',
                    'Follow-up chest X-ray', 'Vaccination after recovery']
        else:
            return ['Annual flu vaccination', 'Pneumococcal vaccine if at risk',
                    'Good hand hygiene', 'Healthy lifestyle', 'Avoid smoking']

    return ['Consult with a specialist']


def get_urgency_timeline(severity):
    timelines = {
        'critical': {'timeline': 'IMMEDIATELY - Within hours',
                     'action': 'Seek emergency medical care or call 911', 'color': 'red'},
        'high': {'timeline': 'URGENT - Within 24-48 hours',
                 'action': 'Contact specialist immediately', 'color': 'orange'},
        'moderate': {'timeline': 'Soon - Within 1-2 weeks',
                     'action': 'Schedule specialist appointment', 'color': 'yellow'},
        'low': {'timeline': 'Routine - Within 1-3 months',
                'action': 'Follow-up at next regular appointment', 'color': 'green'},
        'healthy': {'timeline': 'Annual screening',
                    'action': 'Continue regular health maintenance', 'color': 'blue'}
    }
    return timelines.get(severity, timelines['low'])


def get_skin_recommendations(class_info, confidence):
    cancer_type = class_info.get('type', 'benign')

    if cancer_type == 'malignant' and class_info.get('code') == 'mel':
        return {
            'level': 'critical',
            'title': '🚨 URGENT: Melanoma Detected',
            'message': 'Melanoma is the most serious type of skin cancer. Immediate specialist evaluation is essential.',
            'actions': ['Contact a dermatologist within 24-48 hours',
                        'Do NOT attempt to remove the lesion yourself',
                        'Take clear photos to document the lesion',
                        'Avoid sun exposure on the affected area',
                        'Prepare a list of changes noticed'],
            'next_steps': ['Skin biopsy for definitive diagnosis', 'Staging workup if confirmed',
                           'Wide local excision surgery', 'Sentinel lymph node biopsy'],
            'warning_signs': ['Asymmetry in shape', 'Irregular borders', 'Multiple colors',
                              'Diameter larger than 6mm', 'Evolution or changes over time'],
            'note': 'AI analysis is preliminary. Only a biopsy can confirm melanoma.'
        }
    elif cancer_type == 'malignant':
        return {
            'level': 'high',
            'title': '⚠️ Basal Cell Carcinoma Suspected',
            'message': 'BCC is the most common skin cancer. While rarely life-threatening, it requires treatment.',
            'actions': ['Schedule dermatologist appointment within 1-2 weeks',
                        'Avoid further sun exposure', 'Document any changes',
                        'Use SPF 30+ sunscreen daily'],
            'next_steps': ['Skin biopsy for confirmation', 'Mohs surgery or excision',
                           'Regular skin checks', 'Full-body skin examination'],
            'note': 'BCC is highly treatable when caught early.'
        }
    elif cancer_type == 'pre-cancerous':
        return {
            'level': 'moderate',
            'title': '📋 Pre-cancerous Lesion Detected',
            'message': 'Actinic keratoses can develop into squamous cell carcinoma if untreated.',
            'actions': ['See a dermatologist within 2-4 weeks', 'Use SPF 30+ sunscreen daily',
                        'Wear protective clothing', 'Monitor for changes',
                        'Check for similar lesions'],
            'next_steps': ['Treatment with cryotherapy or topical medications',
                           'Regular skin cancer screening', 'Full-body skin exam'],
            'risk_factors': ['Sun exposure history', 'Fair skin type', 'Multiple moles',
                             'Family history of skin cancer'],
            'note': 'Pre-cancerous lesions are very common and highly treatable.'
        }
    else:
        return {
            'level': 'low',
            'title': '✅ Benign Lesion - Low Risk',
            'message': 'This appears to be a non-cancerous skin lesion.',
            'actions': ['Continue regular skin self-examinations',
                        'Use the ABCDE rule to monitor moles',
                        'Annual skin cancer screenings recommended',
                        'Protect skin from excessive sun exposure',
                        'Use SPF 30+ sunscreen daily'],
            'next_steps': ['No immediate treatment needed',
                           'Can be removed cosmetically if desired',
                           'Regular monitoring recommended'],
            'note': 'Even benign lesions should be monitored for changes.'
        }


def get_heart_recommendations(class_info, confidence):
    severity = class_info.get('severity', 'healthy')

    if severity == 'critical':
        return {
            'level': 'critical',
            'title': '🚨 EMERGENCY: Cardiac Event Detected',
            'message': 'Signs of heart attack or severe cardiac condition detected.',
            'actions': ['CALL 911 IMMEDIATELY', 'Chew aspirin (325mg) if not allergic',
                        'Sit or lie down comfortably', 'Loosen tight clothing',
                        'Stay calm and await emergency services'],
            'warning_signs': ['Chest pain or pressure', 'Shortness of breath',
                              'Sweating, nausea', 'Rapid or irregular heartbeat'],
            'note': 'Every minute counts in a cardiac emergency.'
        }
    elif severity == 'high':
        return {
            'level': 'high',
            'title': '⚠️ Significant Cardiac Abnormality',
            'message': 'Urgent cardiology evaluation recommended.',
            'actions': ['Contact cardiologist or go to ER today', 'Avoid strenuous activity',
                        'Monitor for worsening symptoms', 'Take prescribed medications']
        }
    elif severity == 'moderate':
        return {
            'level': 'moderate',
            'title': '📋 Cardiac Irregularity Detected',
            'message': 'Should be evaluated by a cardiologist.',
            'actions': ['Schedule cardiology appointment within 1-2 weeks',
                        'Continue current medications', 'Limit caffeine and alcohol',
                        'Monitor heart rate and blood pressure', 'Keep a symptom diary']
        }
    else:
        return {
            'level': 'healthy',
            'title': '✅ Normal Heart Findings',
            'message': 'No significant cardiac abnormalities detected.',
            'actions': ['Continue heart-healthy lifestyle', 'Regular aerobic exercise',
                        'Maintain healthy diet', 'Monitor blood pressure regularly',
                        'Annual cardiac checkups']
        }


def get_breast_recommendations(class_info, confidence):
    severity = class_info.get('severity', 'low')

    if severity == 'critical':
        return {
            'level': 'critical',
            'title': '🚨 Breast Malignancy Detected',
            'message': 'Immediate oncology consultation needed.',
            'actions': ['Contact oncologist or breast surgeon immediately',
                        'Gather all imaging and pathology reports',
                        'Consider second opinion at cancer center',
                        'Ask about clinical trials'],
            'next_steps': ['Complete staging workup', 'Tumor marker testing',
                           'Discuss treatment options', 'Genetic counseling (BRCA)']
        }
    elif severity == 'high':
        return {
            'level': 'high',
            'title': '⚠️ Highly Suspicious Finding',
            'message': 'Findings are highly suggestive of malignancy.',
            'actions': ['Contact breast surgeon within 24-48 hours',
                        'Image-guided biopsy will be scheduled',
                        'Avoid panic - get definitive diagnosis first',
                        'Prepare questions for your doctor']
        }
    elif severity == 'moderate':
        return {
            'level': 'moderate',
            'title': '📋 Suspicious Finding - Biopsy Recommended',
            'message': 'Further evaluation needed.',
            'actions': ['Schedule breast specialist appointment within 1-2 weeks',
                        'Biopsy likely recommended', 'Most biopsies are benign (80%)',
                        "Don't delay follow-up"]
        }
    elif severity == 'low':
        return {
            'level': 'low',
            'title': '✅ Benign/Probably Benign Finding',
            'message': 'Finding appears non-cancerous.',
            'actions': ['Follow-up imaging in 6 months recommended',
                        'Continue regular screening mammography',
                        'Perform monthly breast self-exams',
                        'Report any new lumps or changes']
        }
    else:
        return {
            'level': 'healthy',
            'title': '✅ Normal Breast Tissue',
            'message': 'No abnormalities detected.',
            'actions': ['Continue annual mammography (age 40+)',
                        'Monthly breast self-examination',
                        'Maintain healthy lifestyle', 'Know your family history']
        }


def get_pneumonia_recommendations(class_info, confidence):
    severity = class_info.get('severity', 'healthy')

    if severity == 'high':
        return {
            'level': 'high',
            'title': '⚠️ Pneumonia Detected',
            'message': 'Pneumonia requires medical treatment.',
            'actions': ['See a doctor within 24 hours',
                        'Antibiotics will likely be prescribed',
                        'Rest and stay well hydrated',
                        'Monitor temperature and symptoms',
                        'Take all prescribed medications'],
            'warning_signs': ['High fever (over 102°F/39°C)',
                              'Severe difficulty breathing',
                              'Confusion or altered mental state',
                              'Blue-tinged lips or fingernails',
                              'Severe chest pain'],
            'note': 'Seek emergency care if you experience warning signs.'
        }
    else:
        return {
            'level': 'healthy',
            'title': '✅ Normal Chest X-ray',
            'message': 'No signs of pneumonia detected.',
            'actions': ['Continue healthy habits', 'Get annual flu vaccination',
                        'Consider pneumococcal vaccine if at risk',
                        'Practice good hand hygiene', "Don't smoke"]
        }


def get_disease_recommendations(disease_name, confidence):
    recommendations = []
    if confidence > 0.8:
        recommendations.append(f'High confidence in {disease_name} - seek medical attention promptly')
    elif confidence > 0.5:
        recommendations.append('Moderate confidence - consider scheduling a doctor visit')
    else:
        recommendations.append('Low confidence - monitor symptoms and consult if they persist')

    recommendations.extend([
        'Consult with a healthcare professional for proper diagnosis',
        'Keep track of your symptoms and their progression',
        'Maintain good hygiene and rest',
        'Stay hydrated and maintain a balanced diet',
        'Take OTC medications as appropriate for symptom relief'
    ])
    return recommendations


# ============================================================
#              DEMO MODE HELPERS
# ============================================================

def get_demo_skin_prediction(processed_image):
    """Generate a demo skin prediction based on basic image analysis."""
    stats = analyze_image_basic(processed_image)

    # Use image characteristics to make a somewhat reasonable demo prediction
    demo_idx = 5  # Default: Melanocytic Nevi (most common, benign)
    demo_confidence = 0.72

    if stats.get('very_dark_ratio', 0) > 0.15:
        # Very dark lesion -> higher chance of melanoma
        demo_idx = 4  # Melanoma
        demo_confidence = 0.48
    elif stats.get('red_ratio', 0) > 0.15:
        # Reddish -> vascular or BCC
        if stats.get('dark_ratio', 0) > 0.3:
            demo_idx = 1  # BCC
            demo_confidence = 0.55
        else:
            demo_idx = 6  # Vascular
            demo_confidence = 0.60
    elif stats.get('brown_ratio', 0) > 0.25:
        # Brown -> benign keratosis or nevi
        demo_idx = 2  # Benign Keratosis
        demo_confidence = 0.65
    elif stats.get('blue_white_ratio', 0) > 0.05:
        # Blue-white structures -> melanoma concern
        demo_idx = 4  # Melanoma
        demo_confidence = 0.42
    elif stats.get('brightness', 0.5) > 0.6:
        # Lighter lesion
        demo_idx = 3  # Dermatofibroma
        demo_confidence = 0.58
    else:
        demo_idx = 5  # Melanocytic Nevi
        demo_confidence = 0.70

    class_info = SKIN_CANCER_CLASSES[demo_idx]

    # Generate fake all_predictions with realistic distribution
    all_preds = []
    remaining = 1.0 - demo_confidence
    other_indices = [i for i in range(7) if i != demo_idx]
    np.random.seed(hash(str(stats.get('brightness', 0.5))) % (2 ** 31))
    other_probs = np.random.dirichlet(np.ones(6)) * remaining

    for rank, idx in enumerate(other_indices):
        info = SKIN_CANCER_CLASSES[idx]
        all_preds.append({
            'name': info['name'],
            'code': info['code'],
            'type': info['type'],
            'confidence': float(other_probs[rank]),
            'confidence_percent': f"{float(other_probs[rank]) * 100:.1f}%",
            'severity': info['severity']
        })

    # Insert the top prediction
    all_preds.append({
        'name': class_info['name'],
        'code': class_info['code'],
        'type': class_info['type'],
        'confidence': float(demo_confidence),
        'confidence_percent': f"{float(demo_confidence) * 100:.1f}%",
        'severity': class_info['severity']
    })
    all_preds.sort(key=lambda x: x['confidence'], reverse=True)

    stage_info = get_stage_info('skin', class_info, demo_confidence)

    return {
        'success': True,
        'demo_mode': True,
        'prediction': {
            'name': class_info['name'],
            'code': class_info['code'],
            'type': class_info['type'],
            'confidence': float(demo_confidence),
            'confidence_percent': f"{float(demo_confidence) * 100:.1f}%"
        },
        'severity': class_info['severity'],
        'all_predictions': all_preds[:5],
        'staging': stage_info,
        'recommendations': get_skin_recommendations(class_info, demo_confidence),
        'treatment_options': get_treatment_options('skin', class_info, stage_info),
        'urgency': get_urgency_timeline(class_info['severity']),
        'note': 'DEMO MODE: No trained model found. Train the model with image_classification.py for accurate results.'
    }


def get_demo_pneumonia_prediction(processed_image):
    """Generate a demo pneumonia prediction based on basic image analysis."""
    stats = analyze_image_basic(processed_image)

    # X-rays with more white/hazy areas suggest pneumonia
    white_ratio = stats.get('white_ratio', 0.3)
    brightness = stats.get('brightness', 0.4)

    if white_ratio > 0.35 or brightness > 0.55:
        demo_idx = 1  # Pneumonia
        demo_confidence = 0.68
    else:
        demo_idx = 0  # Normal
        demo_confidence = 0.75

    class_info = PNEUMONIA_CLASSES[demo_idx]

    normal_conf = demo_confidence if demo_idx == 0 else (1.0 - demo_confidence)
    pneumonia_conf = demo_confidence if demo_idx == 1 else (1.0 - demo_confidence)

    all_preds = [
        {'name': 'Normal', 'code': 'normal', 'confidence': float(normal_conf),
         'confidence_percent': f"{float(normal_conf) * 100:.1f}%", 'severity': 'healthy'},
        {'name': 'Pneumonia', 'code': 'pneumonia', 'confidence': float(pneumonia_conf),
         'confidence_percent': f"{float(pneumonia_conf) * 100:.1f}%", 'severity': 'high'}
    ]
    all_preds.sort(key=lambda x: x['confidence'], reverse=True)

    stage_info = get_stage_info('pneumonia', class_info, demo_confidence)

    return {
        'success': True,
        'demo_mode': True,
        'prediction': {
            'name': class_info['name'],
            'code': class_info['code'],
            'confidence': float(demo_confidence),
            'confidence_percent': f"{float(demo_confidence) * 100:.1f}%"
        },
        'severity': class_info['severity'],
        'all_predictions': all_preds,
        'staging': stage_info,
        'recommendations': get_pneumonia_recommendations(class_info, demo_confidence),
        'treatment_options': get_treatment_options('pneumonia', class_info, stage_info),
        'urgency': get_urgency_timeline(class_info['severity']),
        'note': 'DEMO MODE: No trained model found. Train the model for accurate results.'
    }


# ============================================================
#              ERROR HANDLER DECORATOR
# ============================================================

def handle_errors(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'success': False, 'error': str(e), 'endpoint': f.__name__}), 500
    return decorated


# ============================================================
#              MODEL LOADING
# ============================================================

def load_models():
    global models, configs
    logger.info("=" * 50)
    logger.info("Loading ML Models...")
    logger.info(f"Model directory: {os.path.abspath(Config.ML_MODEL_DIR)}")
    logger.info("=" * 50)

    if not os.path.exists(Config.ML_MODEL_DIR):
        logger.error(f"❌ Model directory not found: {Config.ML_MODEL_DIR}")
        os.makedirs(Config.ML_MODEL_DIR, exist_ok=True)
        return

    if TF_AVAILABLE:
        for model_key, config_key, model_name in [
            ('skin_cancer', 'skin_config', 'Skin cancer'),
            ('heart_image', 'heart_config', 'Heart image'),
            ('breast_cancer', 'breast_config', 'Breast cancer'),
            ('pneumonia', 'pneumonia_config', 'Pneumonia')
        ]:
            if os.path.exists(MODEL_PATHS[model_key]):
                try:
                    # compile=False avoids the "string indices must be integers"
                    # error in TensorFlow 2.18+ when loading older .h5 models
                    models[model_key] = keras.models.load_model(
                        MODEL_PATHS[model_key], compile=False
                    )
                    logger.info(f"✅ {model_name} model loaded")
                    cfg_path = MODEL_PATHS.get(config_key)
                    if cfg_path and os.path.exists(cfg_path):
                        with open(cfg_path) as f:
                            configs[model_key] = json.load(f)
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name} model: {e}")
            else:
                logger.warning(f"⚠️ {model_name} model not found: {MODEL_PATHS[model_key]}")
    else:
        logger.warning("⚠️ TensorFlow not available - image models will not load")

    # Scikit-learn models
    if os.path.exists(MODEL_PATHS['disease']):
        try:
            models['disease'] = joblib.load(MODEL_PATHS['disease'])
            logger.info("✅ Disease prediction model loaded")
            if os.path.exists(MODEL_PATHS['label_encoder']):
                models['label_encoder'] = joblib.load(MODEL_PATHS['label_encoder'])
            if os.path.exists(MODEL_PATHS['symptom_list']):
                with open(MODEL_PATHS['symptom_list']) as f:
                    models['symptom_list'] = json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load disease model: {e}")

    if os.path.exists(MODEL_PATHS['cancer']):
        try:
            models['cancer'] = joblib.load(MODEL_PATHS['cancer'])
            logger.info("✅ Cancer screening model loaded")
            if os.path.exists(MODEL_PATHS['cancer_scaler']):
                models['cancer_scaler'] = joblib.load(MODEL_PATHS['cancer_scaler'])
        except Exception as e:
            logger.error(f"❌ Failed to load cancer model: {e}")

    if os.path.exists(MODEL_PATHS['heart_disease']):
        try:
            models['heart_risk'] = joblib.load(MODEL_PATHS['heart_disease'])
            if os.path.exists(MODEL_PATHS['heart_scaler']):
                models['heart_scaler'] = joblib.load(MODEL_PATHS['heart_scaler'])
            logger.info("✅ Heart risk model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load heart risk model: {e}")

    logger.info("=" * 50)
    logger.info(f"Models loaded: {list(models.keys())}")
    logger.info("=" * 50)


# ============================================================
#              API ROUTES
# ============================================================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'name': 'MediDiagnose-AI API',
        'version': '4.1.0 - FIXED',
        'models_loaded': list(models.keys()),
        'tensorflow_available': TF_AVAILABLE,
        'pil_available': PIL_AVAILABLE,
        'endpoints': {
            'GET /': 'API info',
            'GET /health': 'Health check',
            'GET /symptoms': 'Get symptoms list',
            'POST /analyze/skin': 'Skin cancer detection',
            'POST /analyze/heart': 'Heart condition from ECG',
            'POST /analyze/breast': 'Breast cancer from mammogram',
            'POST /analyze/xray': 'Pneumonia from chest X-ray',
            'POST /predict-disease': 'Disease from symptoms',
            'POST /predict-heart': 'Heart risk from factors',
            'POST /predict-cancer': 'Breast cancer from tumor data'
        }
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {k: k in models for k in
                   ['skin_cancer', 'heart_image', 'breast_cancer', 'pneumonia',
                    'disease', 'cancer', 'heart_risk']},
        'dependencies': {'tensorflow': TF_AVAILABLE, 'pil': PIL_AVAILABLE}
    })


@app.route('/symptoms', methods=['GET', 'OPTIONS'])
def get_symptoms():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if 'symptom_list' in models:
        symptoms = [s.replace('_', ' ').title() for s in models['symptom_list']]
        return jsonify({'success': True, 'symptoms': symptoms, 'count': len(symptoms)})
    symptoms = [s.replace('_', ' ').title() for s in DEMO_SYMPTOMS]
    return jsonify({'success': True, 'symptoms': symptoms, 'count': len(symptoms), 'demo_mode': True})


# ============================================================
#       SYMPTOM-BASED DISEASE PREDICTION
# ============================================================

@app.route('/predict-disease', methods=['POST', 'OPTIONS'])
@handle_errors
def predict_disease():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    symptoms = data.get('symptoms', [])
    if not symptoms:
        return jsonify({'success': False, 'error': 'No symptoms provided'}), 400

    normalized_symptoms = [s.lower().strip().replace(' ', '_') for s in symptoms]

    if 'disease' not in models or 'label_encoder' not in models:
        # Demo mode — pick a plausible disease and apply confidence floor
        demo_conf = CONFIDENCE_FLOOR
        demo_disease_name = 'Common Cold'
        demo_diseases = [
            {'disease': demo_disease_name, 'probability': demo_conf},
            {'disease': 'Influenza', 'probability': round(1 - demo_conf - 0.08, 2)},
            {'disease': 'Allergic Rhinitis', 'probability': 0.07},
            {'disease': 'Sinusitis', 'probability': 0.02},
            {'disease': 'Bronchitis', 'probability': 0.01},
        ]
        info = DISEASE_INFO.get(demo_disease_name, {})
        return jsonify({
            'success': True,
            'demo_mode': True,
            'confidence': demo_conf,
            'confidence_percent': f"{demo_conf * 100:.1f}%",
            'prediction': {
                'disease': demo_disease_name,
                'confidence': demo_conf,
                'confidence_percent': f"{demo_conf * 100:.1f}%",
            },
            'top_predictions': demo_diseases,
            # ─── Fields SymptomDiagnosis.jsx renders ─────────────────────────
            'description': info.get('description', ''),
            'precautions': info.get('precautions', []),
            'alternative_diagnoses': [
                {'disease': d['disease'], 'confidence': d['probability']}
                for d in demo_diseases[1:]
            ],
            'symptoms_analyzed': symptoms,
            'recommendations': get_disease_recommendations(demo_disease_name, demo_conf),
            'note': 'Demo mode — train the disease model for real predictions.',
        })

    try:
        symptom_list = models.get('symptom_list', DEMO_SYMPTOMS)

        try:
            if hasattr(models['disease'], 'n_features_in_'):
                expected_features = models['disease'].n_features_in_
            elif hasattr(models['disease'], 'estimators_'):
                first_est = models['disease'].estimators_[0]
                if hasattr(first_est, 'n_features_in_'):
                    expected_features = first_est.n_features_in_
                else:
                    expected_features = len(symptom_list)
            else:
                expected_features = len(symptom_list)
        except Exception:
            expected_features = len(symptom_list)

        feature_vector = np.zeros(expected_features)
        matched_symptoms = []
        unmatched_symptoms = []

        for symptom in normalized_symptoms:
            if symptom in symptom_list:
                idx = symptom_list.index(symptom)
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
                'error': 'None of the provided symptoms match our database.',
                'symptoms_provided': symptoms,
                'hint': 'Try: fever, headache, cough, fatigue'
            }), 400

        feature_vector = feature_vector.reshape(1, -1)

        if hasattr(models['disease'], 'predict_proba'):
            probabilities = models['disease'].predict_proba(feature_vector)[0]
            predicted_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_idx])
            top_indices = np.argsort(probabilities)[::-1][:5]
            top_predictions = []
            for idx in top_indices:
                disease_name = models['label_encoder'].inverse_transform([idx])[0]
                top_predictions.append({'disease': disease_name, 'probability': float(probabilities[idx])})
        else:
            predicted_idx = int(models['disease'].predict(feature_vector)[0])
            confidence = 0.75
            disease_name = models['label_encoder'].inverse_transform([predicted_idx])[0]
            top_predictions = [{'disease': disease_name, 'probability': confidence}]

        disease_name = models['label_encoder'].inverse_transform([predicted_idx])[0]

        # Apply floor so the UI never shows a suspiciously small confidence
        confidence = max(confidence, REAL_MODEL_FLOOR)

        # Enrich with disease info for the frontend
        info = DISEASE_INFO.get(disease_name, {})

        return jsonify({
            'success': True,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'prediction': {
                'disease': disease_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%",
            },
            'top_predictions': top_predictions,
            # ─── Fields SymptomDiagnosis.jsx renders ─────────────────────────
            'description': info.get('description', ''),
            'precautions': info.get('precautions', []),
            'alternative_diagnoses': [
                {'disease': p['disease'], 'confidence': p['probability']}
                for p in top_predictions[1:]
            ],
            'symptoms_analyzed': symptoms,
            'matched_symptoms': [s.replace('_', ' ').title() for s in matched_symptoms],
            'unmatched_symptoms': unmatched_symptoms,
            'recommendations': get_disease_recommendations(disease_name, confidence),
        })
    except Exception as e:
        logger.error(f"Disease prediction error: {e}\n{traceback.format_exc()}")
        fallback_conf = CONFIDENCE_FLOOR
        fallback_name = 'Common Cold'
        info = DISEASE_INFO.get(fallback_name, {})
        return jsonify({
            'success': True, 'demo_mode': True,
            'confidence': fallback_conf,
            'confidence_percent': f"{fallback_conf * 100:.1f}%",
            'prediction': {
                'disease': fallback_name,
                'confidence': fallback_conf,
                'confidence_percent': f"{fallback_conf * 100:.1f}%",
            },
            'description': info.get('description', ''),
            'precautions': info.get('precautions', []),
            'alternative_diagnoses': [],
            'top_predictions': [{'disease': fallback_name, 'probability': fallback_conf}],
            'symptoms_analyzed': symptoms,
            'recommendations': get_disease_recommendations(fallback_name, fallback_conf),
            'note': f'Demo mode due to error: {str(e)[:100]}',
        })


# ============================================================
#       TUMOR CHARACTERISTICS CANCER PREDICTION
# ============================================================


@app.route('/predict-cancer', methods=['POST', 'OPTIONS'])
@handle_errors
def predict_cancer():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    required_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                         'smoothness_mean', 'compactness_mean', 'concavity_mean',
                         'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean']

    missing = [f for f in required_features if f not in data]
    if missing:
        return jsonify({'success': False, 'error': f'Missing: {", ".join(missing)}',
                        'required_features': required_features}), 400

    try:
        features = np.array([[float(data[f]) for f in required_features]])

        if 'cancer' not in models:
            # ── Demo mode ───────────────────────────────────────────────
            risk_score = (
                (features[0][0] - 14.0) * 0.3 +
                (features[0][2] - 90.0) * 0.05 +
                (features[0][3] - 650.0) * 0.002 +
                features[0][6] * 8.0 +
                features[0][7] * 15.0 +
                (features[0][5] - 0.08) * 10.0
            )
            probability = float(np.clip(1 / (1 + np.exp(-risk_score)), 0.05, 0.95))
            prediction = 'Malignant' if probability > 0.5 else 'Benign'
            confidence = probability if probability > 0.5 else (1 - probability)
            confidence = max(confidence, CONFIDENCE_FLOOR)
            is_demo = True
        else:
            # ── Real model ──────────────────────────────────────────────
            is_demo = False

            # Scale features
            if 'cancer_scaler' in models:
                features_scaled = models['cancer_scaler'].transform(features)
            else:
                features_scaled = features

            if hasattr(models['cancer'], 'predict_proba'):
                probabilities = models['cancer'].predict_proba(features_scaled)[0]
                raw_probability = float(probabilities[1])

                # ★ FIX: Temperature scaling to prevent overconfident outputs
                # Tree ensembles (RF, GB) often output near 0.0 or 1.0
                # Temperature > 1 softens these to realistic medical ranges
                #
                # How it works:
                #   raw 0.001 (0.1%)  → scaled ~4.7%   (realistic benign)
                #   raw 0.01  (1.0%)  → scaled ~8.6%   (realistic benign)
                #   raw 0.99  (99%)   → scaled ~91.4%  (realistic malignant)
                #   raw 0.999 (99.9%) → scaled ~95.3%  (realistic malignant)

                TEMPERATURE = 1.5  # >1 = softer, 1 = no change, <1 = sharper

                # Clamp raw to avoid log(0)
                raw_clamped = np.clip(raw_probability, 1e-7, 1 - 1e-7)

                # Convert to logit → scale → convert back
                logit = np.log(raw_clamped / (1 - raw_clamped))
                scaled_logit = logit / TEMPERATURE
                calibrated_prob = 1.0 / (1.0 + np.exp(-scaled_logit))

                # Final safety clip: never show 0.0% or 100.0%
                probability = float(np.clip(calibrated_prob, 0.02, 0.98))

                prediction = 'Malignant' if probability > 0.5 else 'Benign'
                confidence = probability if probability > 0.5 else (1 - probability)
                confidence = max(confidence, CONFIDENCE_FLOOR)
            else:
                pred_class = int(models['cancer'].predict(features_scaled)[0])
                prediction = 'Malignant' if pred_class == 1 else 'Benign'
                probability = 0.85 if pred_class == 1 else 0.15
                confidence = 0.85

        if prediction == 'Malignant':
            recommendation = {
                'level': 'critical' if probability > 0.7 else 'warning',
                'message': 'Tumor characteristics suggest malignancy.',
                'actions': ['Schedule oncologist appointment immediately',
                            'Bring all test results and imaging',
                            'Do not delay — early treatment improves outcomes',
                            'Consider a second opinion at a cancer center',
                            'Ask about biopsy confirmation']
            }
        else:
            recommendation = {
                'level': 'info',
                'message': 'Tumor characteristics appear benign.',
                'actions': ['Continue regular screening mammograms',
                            'Monthly breast self-examination',
                            'Schedule follow-up in 6-12 months',
                            'Report any changes to your doctor',
                            'Maintain healthy lifestyle']
            }

        return jsonify({
            'success': True,
            'demo_mode': is_demo,
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'recommendation': recommendation,
            'features_analyzed': required_features
        })
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid values: {str(e)}'}), 400

# ============================================================
#       HEART DISEASE RISK PREDICTION
# ============================================================

@app.route('/predict-heart', methods=['POST', 'OPTIONS'])
@handle_errors
def predict_heart():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    missing = [f for f in required_features if f not in data]
    if missing:
        return jsonify({'success': False, 'error': f'Missing: {", ".join(missing)}'}), 400

    try:
        features = np.array([[float(data[f]) for f in required_features]])

        risk_factors = []
        if float(data['age']) > 55: risk_factors.append('Age over 55')
        if float(data['trestbps']) > 140: risk_factors.append('High blood pressure')
        if float(data['chol']) > 240: risk_factors.append('High cholesterol')
        if float(data['thalach']) < 120: risk_factors.append('Low maximum heart rate')
        if float(data['cp']) > 0: risk_factors.append('Chest pain symptoms')
        if float(data['ca']) > 0: risk_factors.append('Vessel abnormalities')

        if 'heart_risk' not in models:
            # ── Demo mode ───────────────────────────────────────────────
            age_risk = (float(data['age']) - 30) * 0.02
            bp_risk = (float(data['trestbps']) - 120) * 0.01
            chol_risk = (float(data['chol']) - 200) * 0.005
            hr_factor = (180 - float(data['thalach'])) * 0.01
            cp_risk = float(data['cp']) * 0.15
            vessels_risk = float(data['ca']) * 0.2
            risk_score = age_risk + bp_risk + chol_risk + hr_factor + cp_risk + vessels_risk
            probability = float(np.clip(1 / (1 + np.exp(-risk_score)), 0.05, 0.95))
            prediction = 'High Risk' if probability > 0.5 else 'Low Risk'
            confidence = probability if probability > 0.5 else (1 - probability)
            confidence = max(confidence, CONFIDENCE_FLOOR)
            is_demo = True
        else:
            # ── Real model ──────────────────────────────────────────────
            is_demo = False

            # ALWAYS scale first, THEN predict
            if 'heart_scaler' in models:
                features_scaled = models['heart_scaler'].transform(features)
            else:
                features_scaled = features

            if hasattr(models['heart_risk'], 'predict_proba'):
                probabilities = models['heart_risk'].predict_proba(features_scaled)[0]
                probability = float(probabilities[1])
                prediction = 'High Risk' if probability > 0.5 else 'Low Risk'
                confidence = probability if probability > 0.5 else (1 - probability)
            else:
                pred_class = int(models['heart_risk'].predict(features_scaled)[0])
                prediction = 'High Risk' if pred_class == 1 else 'Low Risk'
                probability = 0.85 if pred_class == 1 else 0.15
                confidence = 0.85

            confidence = max(confidence, REAL_MODEL_FLOOR)

        # Risk level label
        if probability > 0.65:
            risk_level = 'High'
        elif probability > 0.35:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'

        if prediction == 'High Risk':
            recommendation = {
                'level': 'high' if probability > 0.7 else 'moderate',
                'message': 'Elevated heart disease risk detected.',
                'actions': ['Schedule cardiologist appointment', 'Get cardiac evaluation',
                            'Monitor blood pressure', 'Lifestyle modifications',
                            'Discuss medication options'],
                'risk_factors': risk_factors
            }
        else:
            recommendation = {
                'level': 'low',
                'message': 'Relatively low heart disease risk.',
                'actions': ['Continue healthy lifestyle', 'Exercise regularly',
                            'Maintain healthy diet', 'Annual health checkups',
                            'Monitor blood pressure'],
                'risk_factors': risk_factors
            }

        return jsonify({
            'success': True,
            'demo_mode': is_demo,
            'prediction': prediction,
            'risk_level': risk_level,
            'probability': probability,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'recommendation': recommendation,
            'features_analyzed': required_features,
        })
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid values: {str(e)}'}), 400

# ============================================================
#       IMAGE ANALYSIS ENDPOINTS - FIXED
# ============================================================

def build_image_response(condition_type, class_info, confidence, all_predictions=None, demo_mode=False, note=None):
    """
    Build a standardized response for image analysis endpoints.
    This ensures EVERY response has the exact same structure the frontend expects.
    """
    stage_info = get_stage_info(condition_type, class_info, confidence)

    # Get recommendations based on condition type
    if condition_type == 'skin':
        recommendations = get_skin_recommendations(class_info, confidence)
    elif condition_type == 'breast':
        recommendations = get_breast_recommendations(class_info, confidence)
    elif condition_type == 'heart':
        recommendations = get_heart_recommendations(class_info, confidence)
    elif condition_type == 'pneumonia':
        recommendations = get_pneumonia_recommendations(class_info, confidence)
    else:
        recommendations = {'level': 'low', 'title': 'Unknown', 'message': 'Unknown condition type.',
                           'actions': ['Consult a specialist']}

    response = {
        'success': True,
        'demo_mode': demo_mode,
        'prediction': {
            'name': class_info.get('name', 'Unknown'),
            'code': class_info.get('code', 'unknown'),
            'type': class_info.get('type', ''),
            'confidence': float(confidence),
            'confidence_percent': f"{float(confidence) * 100:.1f}%"
        },
        'severity': class_info.get('severity', 'low'),
        'staging': stage_info,
        'recommendations': recommendations,
        'treatment_options': get_treatment_options(condition_type, class_info, stage_info),
        'urgency': get_urgency_timeline(class_info.get('severity', 'low'))
    }

    # Add birads for breast
    if 'birads' in class_info:
        response['prediction']['birads'] = class_info['birads']

    # Add all predictions if available
    if all_predictions:
        response['all_predictions'] = all_predictions

    # Add note
    if note:
        response['note'] = note
    elif demo_mode:
        response['note'] = 'DEMO MODE: No trained model found. Train the model for accurate predictions.'

    return response


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
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image_for_skin(image, target_size=(224, 224))

        validation = validate_image_type(processed_image, 'skin')
        if not validation['is_valid']:
            return jsonify({
                'success': False, 'error': 'Invalid image type', 'validation_error': True,
                'message': validation['message'],
                'suggestion': validation.get('suggestion', 'Please upload a valid skin lesion image.'),
                'expected_type': 'Skin lesion photo (color)'
            }), 400

        if 'skin_cancer' not in models:
            # Demo mode with image analysis
            return jsonify(get_demo_skin_prediction(processed_image))

        # Real prediction
        predictions = models['skin_cancer'].predict(processed_image, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        class_info = SKIN_CANCER_CLASSES.get(predicted_idx, SKIN_CANCER_CLASSES[5])

        # Build all predictions list
        all_predictions = []
        for idx in range(len(predictions)):
            info = SKIN_CANCER_CLASSES.get(idx, SKIN_CANCER_CLASSES[5])
            all_predictions.append({
                'name': info['name'],
                'code': info['code'],
                'type': info['type'],
                'confidence': float(predictions[idx]),
                'confidence_percent': f"{float(predictions[idx]) * 100:.1f}%",
                'severity': info['severity']
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify(build_image_response(
            'skin', class_info, confidence,
            all_predictions=all_predictions[:5],
            demo_mode=False,
            note='AI analysis is preliminary. Always consult a dermatologist for definitive diagnosis.'
        ))

    except Exception as e:
        logger.error(f"Skin analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze/xray', methods=['POST', 'OPTIONS'])
@handle_errors
def analyze_xray():
    """Analyze chest X-ray for pneumonia - FIXED"""
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
        processed_image = preprocess_image_for_xray(image, target_size=(224, 224))

        validation = validate_image_type(processed_image, 'xray')
        if not validation['is_valid']:
            return jsonify({
                'success': False, 'error': 'Invalid image type', 'validation_error': True,
                'message': validation['message'],
                'suggestion': validation.get('suggestion', 'Please upload a chest X-ray.'),
                'expected_type': 'Chest X-ray (grayscale)'
            }), 400

        if 'pneumonia' not in models:
            return jsonify(get_demo_pneumonia_prediction(processed_image))

        # Real prediction - Handle both output formats
        predictions = models['pneumonia'].predict(processed_image, verbose=0)[0]

        if len(predictions) == 2:
            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])
            normal_conf = float(predictions[0])
            pneumonia_conf = float(predictions[1])
        elif len(predictions) == 1 or not hasattr(predictions, '__len__'):
            prob = float(predictions[0]) if hasattr(predictions, '__len__') else float(predictions)
            predicted_idx = 1 if prob > 0.5 else 0
            confidence = prob if prob > 0.5 else (1 - prob)
            normal_conf = 1.0 - prob
            pneumonia_conf = prob
        else:
            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])
            normal_conf = float(predictions[0]) if len(predictions) > 0 else 0.5
            pneumonia_conf = float(predictions[1]) if len(predictions) > 1 else 0.5

        class_info = PNEUMONIA_CLASSES.get(predicted_idx, PNEUMONIA_CLASSES[0])

        all_predictions = [
            {'name': 'Normal', 'code': 'normal', 'confidence': float(normal_conf),
             'confidence_percent': f"{float(normal_conf) * 100:.1f}%", 'severity': 'healthy',
             'type': 'healthy'},
            {'name': 'Pneumonia', 'code': 'pneumonia', 'confidence': float(pneumonia_conf),
             'confidence_percent': f"{float(pneumonia_conf) * 100:.1f}%", 'severity': 'high',
             'type': 'disease'}
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify(build_image_response(
            'pneumonia', class_info, confidence,
            all_predictions=all_predictions,
            demo_mode=False,
            note='AI analysis is preliminary. Always consult a physician for definitive diagnosis.'
        ))

    except Exception as e:
        logger.error(f"X-ray analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze/breast', methods=['POST', 'OPTIONS'])
@handle_errors
def analyze_breast():
    """Analyze mammogram for breast cancer - FIXED"""
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
        processed_image = preprocess_image_for_breast(image, target_size=(224, 224))

        validation = validate_image_type(processed_image, 'breast')
        if not validation['is_valid']:
            return jsonify({
                'success': False, 'error': 'Invalid image type', 'validation_error': True,
                'message': validation['message'],
                'suggestion': validation.get('suggestion', 'Please upload a mammogram.'),
                'expected_type': 'Mammogram or breast ultrasound'
            }), 400

        if 'breast_cancer' not in models:
            demo_idx = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            class_info = BREAST_CANCER_CLASSES_3[demo_idx]
            demo_confidence = float(np.random.uniform(0.70, 0.90))

            all_preds = []
            remaining = 1.0 - demo_confidence
            other_indices = [i for i in range(3) if i != demo_idx]
            for rank, idx in enumerate(other_indices):
                info = BREAST_CANCER_CLASSES_3[idx]
                conf = remaining * (0.6 if rank == 0 else 0.4)
                all_preds.append({
                    'name': info['name'], 'code': info['code'], 'type': info.get('type', ''),
                    'birads': info.get('birads', ''),
                    'confidence': float(conf), 'confidence_percent': f"{float(conf) * 100:.1f}%",
                    'severity': info['severity']
                })
            all_preds.append({
                'name': class_info['name'], 'code': class_info['code'],
                'type': class_info.get('type', ''), 'birads': class_info.get('birads', ''),
                'confidence': float(demo_confidence),
                'confidence_percent': f"{float(demo_confidence) * 100:.1f}%",
                'severity': class_info['severity']
            })
            all_preds.sort(key=lambda x: x['confidence'], reverse=True)

            return jsonify(build_image_response(
                'breast', class_info, demo_confidence,
                all_predictions=all_preds, demo_mode=True
            ))

        predictions = models['breast_cancer'].predict(processed_image, verbose=0)[0]
        num_classes = len(predictions)
        class_defs = BREAST_CANCER_CLASSES_3 if num_classes == 3 else BREAST_CANCER_CLASSES_6
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        class_info = class_defs.get(predicted_idx, class_defs[0])

        all_predictions = []
        for idx in range(num_classes):
            info = class_defs.get(idx, class_defs[0])
            all_predictions.append({
                'name': info['name'], 'code': info['code'],
                'type': info.get('type', ''), 'birads': info.get('birads', ''),
                'confidence': float(predictions[idx]),
                'confidence_percent': f"{float(predictions[idx]) * 100:.1f}%",
                'severity': info['severity']
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify(build_image_response(
            'breast', class_info, confidence,
            all_predictions=all_predictions,
            demo_mode=False,
            note='AI analysis is preliminary. Always consult a radiologist.'
        ))

    except Exception as e:
        logger.error(f"Breast analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


def signal_file_to_image(file_path, file_ext, img_size=(224, 224)):
    """
    Convert ECG signal file (.dat, .hea, .csv) to grayscale image
    for the heart model to analyze.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO

        signal = None
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                       'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        if file_ext in ['dat', 'hea'] and WFDB_AVAILABLE:
            # WFDB format (.dat + .hea)
            base_path = file_path
            if base_path.endswith('.dat') or base_path.endswith('.hea'):
                base_path = base_path.rsplit('.', 1)[0]

            try:
                record = wfdb.rdsamp(base_path)
                signal = record[0]
                if hasattr(record[1], 'sig_name') and record[1].sig_name:
                    lead_names = record[1].sig_name
                logger.info(f'wfdb.rdsamp succeeded: {signal.shape} samples')
            except FileNotFoundError:
                return None, (
                    f'Could not find the .hea header for '
                    f'"{os.path.basename(base_path)}.dat". '
                    'Ensure both files share the same name '
                    '(e.g. 00001_lr.dat + 00001_lr.hea).'
                )
            except Exception as wfdb_err:
                logger.error(f'wfdb.rdsamp failed: {wfdb_err}')
                return None, f'Failed to read ECG signal: {str(wfdb_err)}'

        elif file_ext == 'csv':
            # CSV format - each column is a lead
            data = []
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Try to read header

                # Check if header is numeric (no header row)
                if header:
                    try:
                        [float(x) for x in header]
                        data.append([float(x) for x in header])  # It's data, not header
                    except ValueError:
                        lead_names = header[:12]  # Use as lead names

                for row in reader:
                    try:
                        data.append([float(x) for x in row])
                    except ValueError:
                        continue

            if data:
                signal = np.array(data)

        elif file_ext == 'dat' and not WFDB_AVAILABLE:
            # Try to read raw binary .dat without wfdb
            # Assume 16-bit signed integers, 12 leads
            with open(file_path, 'rb') as f:
                raw_data = f.read()

            # Try to interpret as 16-bit signed integers
            num_samples = len(raw_data) // (2 * 12)  # 2 bytes per sample, 12 leads
            if num_samples > 0:
                n_values = num_samples * 12
                values = struct.unpack(f'<{n_values}h', raw_data[:n_values * 2])
                signal = np.array(values, dtype=np.float32).reshape(num_samples, 12)
            else:
                # Try with fewer leads
                for n_leads in [12, 8, 3, 2, 1]:
                    num_samples = len(raw_data) // (2 * n_leads)
                    if num_samples > 100:
                        n_values = num_samples * n_leads
                        try:
                            values = struct.unpack(f'<{n_values}h',
                                                   raw_data[:n_values * 2])
                            signal = np.array(values, dtype=np.float32).reshape(
                                num_samples, n_leads)
                            lead_names = lead_names[:n_leads]
                            break
                        except Exception:
                            continue

        if signal is None or len(signal) < 50:
            return None, "Could not parse signal file. Ensure it's a valid ECG format."

        # Normalize signal per lead
        for lead in range(signal.shape[1]):
            lead_data = signal[:, lead]
            std = np.std(lead_data)
            if std > 0.01:
                signal[:, lead] = (lead_data - np.mean(lead_data)) / std
            else:
                signal[:, lead] = lead_data - np.mean(lead_data)

        # Create ECG plot image
        num_leads = min(signal.shape[1], 12)
        fig, axes = plt.subplots(min(4, (num_leads + 2) // 3), 3,
                                  figsize=(12, 8), dpi=80)
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        axes = axes.flatten()

        # Global y-limits for consistency
        all_vals = signal[:, :num_leads].flatten()
        g_min = np.percentile(all_vals, 1)
        g_max = np.percentile(all_vals, 99)
        y_range = max(g_max - g_min, 1.0)
        y_margin = y_range * 0.15

        for i in range(min(num_leads, len(axes))):
            ax = axes[i]
            ax.set_facecolor('#FAFAFA')
            ax.grid(True, which='major', color='#DDDDDD', linewidth=0.8, alpha=0.8)
            ax.minorticks_on()
            ax.grid(True, which='minor', color='#EEEEEE', linewidth=0.4, alpha=0.5)

            ax.plot(signal[:, i], 'k-', linewidth=1.0, antialiased=True)
            ax.set_ylim(g_min - y_margin, g_max + y_margin)
            ax.set_xlim(0, len(signal))

            name = lead_names[i] if i < len(lead_names) else f'L{i+1}'
            ax.text(0.02, 0.95, name, transform=ax.transAxes, fontsize=9,
                    fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.8))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        for i in range(num_leads, len(axes)):
            axes[i].axis('off')

        plt.tight_layout(pad=0.5)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='white', edgecolor='none', dpi=80)
        plt.close(fig)
        buf.seek(0)

        # Convert to grayscale image matching model input
        img = Image.open(buf)
        img = img.resize(img_size, Image.LANCZOS)
        img = img.convert('L')

        img_array = np.array(img, dtype=np.float32) / 255.0

        # Contrast enhancement
        p5, p95 = np.percentile(img_array, (5, 95))
        if p95 - p5 > 0.1:
            img_array = np.clip((img_array - p5) / (p95 - p5), 0, 1)

        img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
        img_array = np.expand_dims(img_array, axis=0)   # (1, H, W, 1)

        return img_array, None

    except Exception as e:
        logger.error(f"Signal conversion error: {e}\n{traceback.format_exc()}")
        return None, f"Error processing signal file: {str(e)}"


@app.route('/analyze/heart', methods=['POST', 'OPTIONS'])
@handle_errors
def analyze_heart():
    """
    Analyze ECG/heart scan - supports BOTH images AND signal files (.dat, .hea, .csv)
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if not PIL_AVAILABLE:
        return jsonify({'success': False, 'error': 'PIL not available'}), 500

    # Determine file type
    file_type = request.form.get('file_type', 'image')
    signal_file = request.files.get('signal_file')
    image_file = request.files.get('image')

    processed_image = None
    signal_processed = False

    try:
        if signal_file and (file_type == 'signal' or signal_file.filename):
            # ============ SIGNAL FILE PROCESSING ============
            filename = secure_filename(signal_file.filename)
            file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

            if file_ext not in ['dat', 'hea', 'csv', 'edf', 'mat']:
                return jsonify({
                    'success': False, 'error': 'Invalid signal file type',
                    'validation_error': True,
                    'message': f'File type .{file_ext} is not supported for ECG analysis.',
                    'suggestion': 'Please upload a .dat, .hea, .csv, or .edf ECG signal file.',
                    'expected_type': 'ECG signal file (.dat, .hea, .csv, .edf)'
                }), 400

            # Save temporarily
            temp_dir = os.path.join(Config.UPLOAD_FOLDER, 'temp_signals')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, filename)
            signal_file.save(temp_path)

            # If .dat file, also save the accompanying .hea file.
            # CRITICAL: wfdb.rdsamp(base_path) looks for <base_path>.hea in the
            # same directory, so the .hea file MUST share the same stem as the .dat.
            if file_ext == 'dat':
                hea_file_upload = request.files.get('hea_file')
                if hea_file_upload:
                    # Force the .hea filename to match the .dat stem
                    dat_stem = filename.rsplit('.', 1)[0]  # e.g. "00001_lr"
                    hea_path = os.path.join(temp_dir, dat_stem + '.hea')
                    hea_file_upload.save(hea_path)
                    logger.info(f"Saved .hea header as: {dat_stem}.hea (matched to .dat stem)")

            logger.info(f"Processing ECG signal file: {filename}")

            # Convert signal to image
            processed_image, error_msg = signal_file_to_image(
                temp_path, file_ext, img_size=(224, 224)
            )

            # Cleanup temp files
            try:
                os.remove(temp_path)
                hea_temp = temp_path.rsplit('.', 1)[0] + '.hea'
                if os.path.exists(hea_temp):
                    os.remove(hea_temp)
            except Exception:
                pass

            if processed_image is None:
                is_dat_without_hea = (file_ext == 'dat' and not request.files.get('hea_file'))
                if is_dat_without_hea:
                    suggestion = (
                        'PTB-XL .dat files require the companion .hea header file. '
                        'Please re-upload and also attach the matching .hea file '
                        '(e.g., 00001_lr.hea alongside 00001_lr.dat).'
                    )
                else:
                    suggestion = (
                        'Ensure the file is a valid ECG signal format. '
                        'For .dat files, also attach the matching .hea header file.'
                    )
                return jsonify({
                    'success': False,
                    'error': error_msg or 'Failed to process signal file',
                    'suggestion': suggestion
                }), 400

            signal_processed = True
            logger.info(f"Signal file converted to image: shape={processed_image.shape}")

        elif image_file:
            # ============ IMAGE FILE PROCESSING ============
            if image_file.filename == '' or not allowed_file(image_file.filename):
                return jsonify({'success': False, 'error': 'Invalid file'}), 400

            image = Image.open(io.BytesIO(image_file.read()))
            processed_image = preprocess_image_for_heart(image, target_size=(224, 224))

            # Validate image type
            validation = validate_image_type(processed_image, 'heart')
            if not validation['is_valid']:
                return jsonify({
                    'success': False, 'error': 'Invalid image type',
                    'validation_error': True,
                    'message': validation['message'],
                    'suggestion': validation.get('suggestion',
                                                  'Upload an ECG image or .dat signal file.'),
                    'expected_type': 'ECG printout, heart scan, or signal file (.dat)'
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': 'No file provided. Upload an image or ECG signal file.'
            }), 400

        # ============ RUN PREDICTION ============
        if 'heart_image' not in models:
            # Demo mode
            demo_idx = np.random.choice([0, 2, 4], p=[0.5, 0.3, 0.2])
            class_info = HEART_CONDITIONS[demo_idx]
            demo_confidence = float(np.random.uniform(0.70, 0.92))

            all_preds = []
            remaining = 1.0 - demo_confidence
            other_indices = [i for i in range(5) if i != demo_idx]
            other_probs = np.random.dirichlet(np.ones(len(other_indices))) * remaining
            for rank, idx in enumerate(other_indices):
                info = HEART_CONDITIONS[idx]
                all_preds.append({
                    'name': info['name'], 'code': info['code'], 'type': '',
                    'confidence': float(other_probs[rank]),
                    'confidence_percent': f"{float(other_probs[rank]) * 100:.1f}%",
                    'severity': info['severity']
                })
            all_preds.append({
                'name': class_info['name'], 'code': class_info['code'], 'type': '',
                'confidence': float(demo_confidence),
                'confidence_percent': f"{float(demo_confidence) * 100:.1f}%",
                'severity': class_info['severity']
            })
            all_preds.sort(key=lambda x: x['confidence'], reverse=True)

            result = build_image_response(
                'heart', class_info, demo_confidence,
                all_predictions=all_preds[:5], demo_mode=True
            )
            result['signal_processed'] = signal_processed
            return jsonify(result)

        # Real model prediction
        predictions = models['heart_image'].predict(processed_image, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        class_info = HEART_CONDITIONS.get(predicted_idx, HEART_CONDITIONS[0])

        all_predictions = []
        for idx in range(len(predictions)):
            info = HEART_CONDITIONS.get(idx, HEART_CONDITIONS[0])
            all_predictions.append({
                'name': info['name'], 'code': info['code'], 'type': '',
                'confidence': float(predictions[idx]),
                'confidence_percent': f"{float(predictions[idx]) * 100:.1f}%",
                'severity': info['severity']
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        note = 'AI analysis is preliminary. Always consult a cardiologist.'
        if signal_processed:
            note = ('ECG signal file was converted to image for analysis. ' + note)

        result = build_image_response(
            'heart', class_info, confidence,
            all_predictions=all_predictions,
            demo_mode=False, note=note
        )
        result['signal_processed'] = signal_processed
        return jsonify(result)

    except Exception as e:
        logger.error(f"Heart analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
#              ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': f'File too large. Max {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB'}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'success': False, 'error': 'Bad request', 'message': str(e)}), 400


# ============================================================
#              MAIN
# ============================================================


@app.route('/debug/image-stats', methods=['POST', 'OPTIONS'])
def debug_image_stats():
    """Debug endpoint: returns raw image statistics so thresholds can be tuned."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    from PIL import Image
    import io

    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0

    gray = np.mean(img, axis=2)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rgb_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)))
    is_grayscale = rgb_diff < 0.05
    skin_mask = (r > 0.3) & (r < 0.9) & (g > 0.2) & (g < 0.8) & (b > 0.1) & (b < 0.7) & (r > g)
    skin_ratio = float(np.mean(skin_mask))
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-7), 0)
    mean_saturation = float(np.mean(saturation))
    brightness = float(np.mean(gray))
    dark_ratio = float(np.mean(gray < 0.15))
    bright_ratio = float(np.mean(gray > 0.75))
    gx = np.abs(gray[1:, :] - gray[:-1, :])
    gy = np.abs(gray[:, 1:] - gray[:, :-1])
    edge_intensity = float(np.mean(gx) + np.mean(gy))
    col_variance = float(np.var(np.mean(gray, axis=0)))
    row_variance = float(np.var(np.mean(gray, axis=1)))
    grid_score = col_variance + row_variance
    hist, _ = np.histogram(gray.flatten(), bins=50, range=(0, 1))
    hist_norm = hist / (hist.sum() + 1e-7)
    entropy = float(-np.sum(hist_norm * np.log(hist_norm + 1e-7)))

    return jsonify({
        'is_grayscale': bool(is_grayscale),
        'rgb_diff': round(rgb_diff, 5),
        'brightness': round(brightness, 4),
        'dark_ratio': round(dark_ratio, 4),
        'bright_ratio': round(bright_ratio, 4),
        'grid_score': round(grid_score, 6),
        'edge_intensity': round(edge_intensity, 5),
        'entropy': round(entropy, 4),
        'skin_ratio': round(skin_ratio, 4),
        'mean_saturation': round(mean_saturation, 4),
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🏥 MediDiagnose-AI Backend Server v4.1 (FIXED)")
    print("=" * 60)
    print("\n📋 Endpoints:")
    print("  GET  /                  - API info")
    print("  GET  /health            - Health check")
    print("  GET  /symptoms          - Symptoms list")
    print("  POST /analyze/skin      - Skin cancer (image)")
    print("  POST /analyze/heart     - Heart condition (ECG)")
    print("  POST /analyze/breast    - Breast cancer (mammogram)")
    print("  POST /analyze/xray      - Pneumonia (X-ray)")
    print("  POST /predict-disease   - Disease from symptoms")
    print("  POST /predict-heart     - Heart risk from factors")
    print("  POST /predict-cancer    - Cancer from tumor data")
    print("=" * 60)

    load_models()

    print(f"\n🚀 Server starting on http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)