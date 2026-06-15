"""
disease_prediction_v2.py
========================
Trains the disease prediction model and outputs files that
server.py expects in its ML_MODEL_DIR (../ml_model/).

Output files (saved to ml_model/):
  - disease_model.joblib      → loaded by server.py as models['disease']
  - label_encoder.joblib      → loaded by server.py as models['label_encoder']
  - symptom_list.json         → loaded by server.py as models['symptom_list']
  - disease_info.json         → disease descriptions & precautions
  - model_config.json         → metadata (accuracy, diseases, symptoms)
  - training_report.txt       → human-readable performance report

Usage:
  python disease_prediction_v2.py                     # train + save
  python disease_prediction_v2.py --test "headache,high_fever,vomiting"  # train + quick test
  python disease_prediction_v2.py --output-dir ./my_models  # custom output dir
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# PATH CONFIG — matches server.py MODEL_PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# server.py expects: ML_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml_model')
# So if this script lives alongside server.py in backend/, output to ../ml_model/
# If this script lives elsewhere, use --output-dir flag
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'ml_model')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')


# ============================================================
# CLEANING — must match server.py normalization exactly
# ============================================================
def clean_symptom(symptom):
    """
    Clean symptom names.
    server.py normalizes user input with:
        s.lower().strip().replace(' ', '_')
    so our symptom_list.json must use the same format.
    """
    if pd.isna(symptom) or symptom is None:
        return None
    symptom = str(symptom).strip().lower()
    symptom = symptom.replace(' ', '_').replace('-', '_')
    symptom = ''.join(c for c in symptom if c.isalnum() or c == '_')
    # collapse multiple underscores
    while '__' in symptom:
        symptom = symptom.replace('__', '_')
    symptom = symptom.strip('_')
    return symptom if symptom else None


# ============================================================
# DISEASE INFO — descriptions & precautions
# ============================================================
def build_disease_info():
    """Disease descriptions and precautions used by the frontend."""
    return {
        'Fungal infection': {
            'description': 'Fungi affecting skin, especially in warm moist areas.',
            'precautions': ['Keep area clean and dry', 'Use antifungal medications',
                            'Avoid sharing personal items', 'Wear breathable clothing']
        },
        'Allergy': {
            'description': 'Immune system reacts to a foreign substance.',
            'precautions': ['Identify and avoid allergens', 'Take antihistamines',
                            'Keep emergency medication', 'Consult an allergist']
        },
        'GERD': {
            'description': 'Stomach acid flows back into esophagus.',
            'precautions': ['Avoid trigger foods', 'Eat smaller meals',
                            'Do not lie down after eating', 'Elevate head while sleeping']
        },
        'Chronic cholestasis': {
            'description': 'Bile flow from liver is reduced or blocked.',
            'precautions': ['Follow prescribed medication', 'Regular liver tests',
                            'Avoid alcohol', 'Maintain healthy diet']
        },
        'Drug Reaction': {
            'description': 'Unwanted effect from medication.',
            'precautions': ['Stop the medication', 'Consult your doctor',
                            'Track medications you react to', 'Use antihistamines if recommended']
        },
        'Peptic ulcer disease': {
            'description': 'Open sores on stomach or intestine lining.',
            'precautions': ['Avoid NSAIDs', 'Reduce stress',
                            'Avoid spicy and acidic foods', 'Take prescribed medications']
        },
        'AIDS': {
            'description': 'HIV weakens the immune system.',
            'precautions': ['Antiretroviral therapy consistently', 'Regular medical checkups',
                            'Practice safe sex', 'Maintain good nutrition']
        },
        'Diabetes': {
            'description': 'Blood glucose levels are too high.',
            'precautions': ['Monitor blood sugar regularly', 'Follow diabetic diet',
                            'Exercise regularly', 'Take medications as prescribed']
        },
        'Gastroenteritis': {
            'description': 'Intestinal infection with diarrhea, cramps, vomiting.',
            'precautions': ['Stay hydrated', 'Rest adequately',
                            'Eat bland foods', 'Practice good hygiene']
        },
        'Bronchial Asthma': {
            'description': 'Airways narrow and swell, producing extra mucus.',
            'precautions': ['Use inhaler as prescribed', 'Avoid triggers',
                            'Monitor peak flow', 'Keep rescue inhaler handy']
        },
        'Hypertension': {
            'description': 'High blood pressure.',
            'precautions': ['Reduce salt intake', 'Exercise regularly',
                            'Maintain healthy weight', 'Take medications as prescribed']
        },
        'Migraine': {
            'description': 'Severe headache, usually on one side.',
            'precautions': ['Identify triggers', 'Rest in dark quiet room',
                            'Apply cold compress', 'Take prescribed medication early']
        },
        'Cervical spondylosis': {
            'description': 'Age-related wear on neck spinal disks.',
            'precautions': ['Physical therapy', 'Maintain good posture',
                            'Use neck support', 'Regular exercise']
        },
        'Paralysis (brain hemorrhage)': {
            'description': 'Loss of muscle function from brain-muscle disruption.',
            'precautions': ['Seek immediate medical attention', 'Physical rehabilitation',
                            'Speech therapy if needed', 'Regular monitoring']
        },
        'Jaundice': {
            'description': 'Yellowing of skin and eyes from high bilirubin.',
            'precautions': ['Treat underlying cause', 'Stay hydrated',
                            'Avoid alcohol', 'Rest adequately']
        },
        'Malaria': {
            'description': 'Parasites transmitted by infected mosquitoes.',
            'precautions': ['Complete antimalarial treatment', 'Use mosquito nets',
                            'Apply insect repellent', 'Seek immediate treatment for fever']
        },
        'Chicken pox': {
            'description': 'Highly contagious disease with itchy blister rash.',
            'precautions': ['Avoid scratching', 'Keep cool',
                            'Use calamine lotion', 'Isolate to prevent spread']
        },
        'Dengue': {
            'description': 'Mosquito-borne tropical disease.',
            'precautions': ['Stay hydrated', 'Rest completely',
                            'Monitor platelet count', 'Avoid mosquito bites']
        },
        'Typhoid': {
            'description': 'Bacterial infection from contaminated food and water.',
            'precautions': ['Complete antibiotic course', 'Drink clean water',
                            'Maintain hygiene', 'Rest adequately']
        },
        'Hepatitis A': {
            'description': 'Highly contagious liver infection.',
            'precautions': ['Get vaccinated', 'Practice good hygiene',
                            'Avoid contaminated food and water', 'Rest and stay hydrated']
        },
        'Hepatitis B': {
            'description': 'Serious liver infection that can become chronic.',
            'precautions': ['Get vaccinated', 'Avoid sharing needles',
                            'Practice safe sex', 'Regular liver monitoring']
        },
        'Hepatitis C': {
            'description': 'Viral infection causing liver inflammation.',
            'precautions': ['Antiviral treatment', 'Avoid alcohol',
                            'Regular monitoring', 'Avoid sharing personal items']
        },
        'Hepatitis D': {
            'description': 'Liver disease that occurs only with hepatitis B.',
            'precautions': ['Treat hepatitis B', 'Avoid alcohol',
                            'Regular medical checkups', 'Antiviral medications']
        },
        'Hepatitis E': {
            'description': 'Liver disease from contaminated drinking water.',
            'precautions': ['Drink clean water', 'Practice good hygiene',
                            'Rest adequately', 'Supportive care']
        },
        'Alcoholic hepatitis': {
            'description': 'Liver inflammation from excessive alcohol.',
            'precautions': ['Stop drinking alcohol completely', 'Nutritional support',
                            'Medications as prescribed', 'Regular monitoring']
        },
        'Tuberculosis': {
            'description': 'Serious infectious disease mainly affecting lungs.',
            'precautions': ['Complete full treatment course', 'Cover mouth when coughing',
                            'Isolate during infectious period', 'Improve ventilation']
        },
        'Common Cold': {
            'description': 'Viral infection of nose and throat.',
            'precautions': ['Rest adequately', 'Stay hydrated',
                            'Use saline drops', 'Wash hands frequently']
        },
        'Pneumonia': {
            'description': 'Infection inflaming air sacs in lungs.',
            'precautions': ['Complete antibiotics', 'Rest completely',
                            'Stay hydrated', 'Get vaccinated']
        },
        'Dimorphic hemmorhoids(piles)': {
            'description': 'Swollen veins in lower rectum.',
            'precautions': ['Increase fiber intake', 'Stay hydrated',
                            'Avoid straining', 'Use sitz baths']
        },
        'Heart attack': {
            'description': 'Blocked blood flow to heart, requires immediate treatment.',
            'precautions': ['Call emergency services immediately', 'Take prescribed medications',
                            'Cardiac rehabilitation', 'Lifestyle modifications']
        },
        'Varicose veins': {
            'description': 'Twisted enlarged veins in legs.',
            'precautions': ['Exercise regularly', 'Elevate legs',
                            'Wear compression stockings', 'Maintain healthy weight']
        },
        'Hypothyroidism': {
            'description': 'Thyroid does not produce enough hormones.',
            'precautions': ['Take thyroid medication daily', 'Regular blood tests',
                            'Maintain healthy diet', 'Monitor symptoms']
        },
        'Hyperthyroidism': {
            'description': 'Thyroid produces too much thyroxine.',
            'precautions': ['Take prescribed medications', 'Regular monitoring',
                            'Avoid excessive iodine', 'Manage stress']
        },
        'Hypoglycemia': {
            'description': 'Blood sugar lower than normal.',
            'precautions': ['Carry fast-acting sugar', 'Eat regular meals',
                            'Monitor blood sugar', 'Adjust medications as needed']
        },
        'Osteoarthristis': {
            'description': 'Most common arthritis, cartilage wearing down.',
            'precautions': ['Maintain healthy weight', 'Exercise regularly',
                            'Physical therapy', 'Pain management']
        },
        'Arthritis': {
            'description': 'Joint inflammation causing pain and stiffness.',
            'precautions': ['Stay active', 'Protect joints',
                            'Maintain healthy weight', 'Use heat and cold therapy']
        },
        'Paroxysmal Positional Vertigo': {
            'description': 'Spinning sensation triggered by head position changes.',
            'precautions': ['Move slowly when changing positions', 'Avoid sudden movements',
                            'Balance exercises', 'Consult ENT specialist']
        },
        'Acne': {
            'description': 'Skin condition from clogged hair follicles.',
            'precautions': ['Cleanse face regularly', 'Avoid touching face',
                            'Use non-comedogenic products', 'Consult dermatologist if severe']
        },
        'Urinary tract infection': {
            'description': 'Infection in any part of urinary system.',
            'precautions': ['Drink plenty of water', 'Complete antibiotic course',
                            'Urinate frequently', 'Practice good hygiene']
        },
        'Psoriasis': {
            'description': 'Skin disease causing red itchy scaly patches.',
            'precautions': ['Moisturize regularly', 'Avoid triggers',
                            'Use prescribed medications', 'Manage stress']
        },
        'Impetigo': {
            'description': 'Contagious skin infection mainly in children.',
            'precautions': ['Keep affected area clean', 'Apply antibiotic ointment',
                            'Avoid touching sores', 'Complete treatment course']
        },
        'Influenza': {
            'description': 'Viral infection affecting respiratory system.',
            'precautions': ['Get annual flu vaccine', 'Rest adequately',
                            'Stay hydrated', 'Antiviral medications if prescribed']
        }
    }


# ============================================================
# DATASET — disease-symptom mappings
# ============================================================
# Symptom names here MUST match what server.py has in DEMO_SYMPTOMS
# and what users will send (lowercased, underscored).

DISEASE_SYMPTOM_MAP = {
    'Common Cold': [
        'continuous_sneezing', 'chills', 'fatigue', 'cough', 'high_fever',
        'headache', 'malaise', 'sore_throat', 'runny_nose', 'watering_from_eyes'
    ],
    'Influenza': [
        'high_fever', 'chills', 'fatigue', 'cough', 'muscle_pain',
        'headache', 'sore_throat', 'weakness_in_limbs', 'loss_of_appetite'
    ],
    'Pneumonia': [
        'chills', 'fatigue', 'cough', 'high_fever', 'breathlessness',
        'sweating', 'chest_pain', 'fast_heart_rate', 'phlegm', 'rusty_sputum'
    ],
    'Tuberculosis': [
        'chills', 'vomiting', 'fatigue', 'cough', 'high_fever',
        'breathlessness', 'loss_of_appetite', 'weight_loss', 'mild_fever',
        'blood_in_sputum', 'phlegm', 'chest_pain'
    ],
    'Malaria': [
        'chills', 'vomiting', 'high_fever', 'sweating', 'headache',
        'nausea', 'diarrhoea', 'muscle_pain'
    ],
    'Dengue': [
        'skin_rash', 'chills', 'joint_pain', 'vomiting', 'fatigue',
        'high_fever', 'headache', 'muscle_pain', 'red_spots_over_body',
        'nausea', 'pain_behind_the_eyes'
    ],
    'Typhoid': [
        'chills', 'vomiting', 'fatigue', 'high_fever', 'headache',
        'constipation', 'abdominal_pain', 'diarrhoea', 'toxic_look_(typhos)',
        'belly_pain'
    ],
    'Chicken pox': [
        'itching', 'skin_rash', 'fatigue', 'high_fever', 'red_spots_over_body',
        'loss_of_appetite', 'headache', 'malaise', 'mild_fever'
    ],
    'Gastroenteritis': [
        'vomiting', 'diarrhoea', 'dehydration', 'sunken_eyes', 'high_fever',
        'abdominal_pain', 'nausea', 'loss_of_appetite'
    ],
    'GERD': [
        'stomach_pain', 'acidity', 'vomiting', 'chest_pain', 'ulcers_on_tongue',
        'cough', 'indigestion'
    ],
    'Peptic ulcer disease': [
        'vomiting', 'loss_of_appetite', 'abdominal_pain', 'passage_of_gases',
        'internal_itching', 'indigestion', 'nausea'
    ],
    'Chronic cholestasis': [
        'itching', 'vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite',
        'abdominal_pain', 'yellowing_of_eyes'
    ],
    'Jaundice': [
        'itching', 'vomiting', 'fatigue', 'yellowish_skin', 'dark_urine',
        'yellowing_of_eyes', 'loss_of_appetite', 'abdominal_pain', 'weight_loss'
    ],
    'Hepatitis A': [
        'joint_pain', 'vomiting', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'abdominal_pain', 'diarrhoea', 'mild_fever',
        'yellowing_of_eyes'
    ],
    'Hepatitis B': [
        'itching', 'fatigue', 'lethargy', 'yellowish_skin', 'dark_urine',
        'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes',
        'receiving_blood_transfusion', 'receiving_unsterile_injections'
    ],
    'Hepatitis C': [
        'fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite',
        'yellowing_of_eyes', 'family_history', 'dark_urine'
    ],
    'Hepatitis D': [
        'joint_pain', 'vomiting', 'fatigue', 'high_fever', 'yellowish_skin',
        'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain',
        'yellowing_of_eyes'
    ],
    'Hepatitis E': [
        'joint_pain', 'vomiting', 'fatigue', 'yellowish_skin', 'dark_urine',
        'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes',
        'acute_liver_failure', 'stomach_bleeding', 'coma'
    ],
    'Alcoholic hepatitis': [
        'vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach',
        'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload',
        'yellowing_of_eyes'
    ],
    'Bronchial Asthma': [
        'breathlessness', 'cough', 'fatigue', 'high_fever', 'mucoid_sputum',
        'chest_pain', 'fast_heart_rate', 'phlegm'
    ],
    'Heart attack': [
        'vomiting', 'breathlessness', 'sweating', 'chest_pain',
        'fast_heart_rate', 'weakness_in_limbs'
    ],
    'Hypertension': [
        'headache', 'chest_pain', 'dizziness', 'lack_of_concentration',
        'blurred_and_distorted_vision'
    ],
    'Diabetes': [
        'fatigue', 'weight_loss', 'restlessness', 'lethargy',
        'irregular_sugar_level', 'increased_appetite', 'polyuria',
        'excessive_hunger', 'blurred_and_distorted_vision'
    ],
    'Hyperthyroidism': [
        'fatigue', 'mood_swings', 'weight_loss', 'restlessness',
        'fast_heart_rate', 'excessive_hunger', 'sweating', 'diarrhoea',
        'irritability'
    ],
    'Hypothyroidism': [
        'fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings',
        'lethargy', 'brittle_nails', 'swollen_extremeties', 'depression',
        'puffy_face_and_eyes', 'enlarged_thyroid'
    ],
    'Fungal infection': [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'
    ],
    'Acne': [
        'skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'
    ],
    'Impetigo': [
        'skin_rash', 'high_fever', 'blister', 'red_sore_around_nose',
        'yellow_crust_ooze'
    ],
    'Psoriasis': [
        'skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting',
        'small_dents_in_nails', 'inflammatory_nails'
    ],
    'Allergy': [
        'continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'
    ],
    'Drug Reaction': [
        'itching', 'skin_rash', 'stomach_pain', 'burning_micturition',
        'spotting_urination'
    ],
    'Urinary tract infection': [
        'burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine',
        'continuous_feel_of_urine'
    ],
    'Migraine': [
        'headache', 'acidity', 'indigestion', 'blurred_and_distorted_vision',
        'visual_disturbances', 'stiff_neck', 'depression', 'irritability'
    ],
    'Cervical spondylosis': [
        'back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness',
        'loss_of_balance'
    ],
    'Paralysis (brain hemorrhage)': [
        'vomiting', 'headache', 'weakness_of_one_body_side', 'altered_sensorium'
    ],
    'Arthritis': [
        'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
        'painful_walking', 'joint_pain'
    ],
    'Osteoarthristis': [
        'joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain',
        'swelling_joints', 'painful_walking'
    ],
    'Varicose veins': [
        'fatigue', 'cramps', 'bruising', 'obesity', 'swollen_legs',
        'swollen_blood_vessels', 'prominent_veins_on_calf'
    ],
    'AIDS': [
        'muscle_wasting', 'patches_in_throat', 'high_fever',
        'extra_marital_contacts'
    ],
    'Dimorphic hemmorhoids(piles)': [
        'constipation', 'pain_during_bowel_movements', 'pain_in_anal_region',
        'bloody_stool', 'irritation_in_anus'
    ],
    'Paroxysmal Positional Vertigo': [
        'spinning_movements', 'loss_of_balance', 'unsteadiness', 'dizziness'
    ],
    'Hypoglycemia': [
        'vomiting', 'fatigue', 'anxiety', 'sweating', 'headache',
        'drying_and_tingling_lips', 'slurred_speech', 'palpitations'
    ],
}


# ============================================================
# DATASET GENERATION
# ============================================================
def generate_dataset(output_path=None):
    """Generate synthetic training data from the symptom map."""
    print("[1/5] Generating training dataset...")

    rows = []
    np.random.seed(42)

    for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
        # 30-50 samples per disease for better training
        n_samples = np.random.randint(30, 51)

        for _ in range(n_samples):
            # Pick 3 to N symptoms (at least 3 so model learns patterns)
            max_pick = min(10, len(symptoms))
            min_pick = min(3, len(symptoms))
            n_pick = np.random.randint(min_pick, max_pick + 1)
            picked = np.random.choice(symptoms, size=n_pick, replace=False).tolist()

            row = {'Disease': disease}
            for i in range(1, 18):  # up to 17 symptom columns
                row[f'Symptom_{i}'] = picked[i - 1] if i <= len(picked) else None
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"   Dataset saved: {output_path}")

    print(f"   {len(df)} samples, {df['Disease'].nunique()} diseases")
    return df


# ============================================================
# LOAD OR GENERATE DATASET
# ============================================================
def load_dataset():
    """Load existing dataset or generate one."""
    dataset_path = os.path.join(DATASET_DIR, 'dataset.csv')

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"[1/5] Loaded existing dataset: {len(df)} rows from {dataset_path}")
    else:
        df = generate_dataset(dataset_path)

    # Standardize disease names (remove typos, trailing spaces, case inconsistencies)
    disease_mapping = {
        'Peptic ulcer diseae': 'Peptic ulcer disease',
        'Diabetes ': 'Diabetes',
        'Hypertension ': 'Hypertension',
        'hepatitis A': 'Hepatitis A',
        'Osteoarthristis': 'Osteoarthritis',
        '(vertigo) Paroymsal  Positional Vertigo': 'Paroxysmal Positional Vertigo'
    }
    df['Disease'] = df['Disease'].replace(disease_mapping)

    # Find symptom columns
    symptom_cols = [c for c in df.columns if 'symptom' in c.lower()]

    # Clean all symptom values
    all_symptoms = set()
    for col in symptom_cols:
        df[col] = df[col].apply(clean_symptom)
        all_symptoms.update(s for s in df[col].dropna().unique() if s)

    print(f"   {len(all_symptoms)} unique symptoms, {df['Disease'].nunique()} diseases")
    return df, symptom_cols, all_symptoms


# ============================================================
# FEATURE MATRIX
# ============================================================
def build_features(df, symptom_cols, all_symptoms):
    """
    Build binary feature matrix.
    
    CRITICAL: symptom_list is sorted alphabetically.
    server.py does:
        feature_vector[idx] = 1  where idx = symptom_list.index(symptom)
    So the order in symptom_list.json MUST match what we train on.
    """
    print("[2/5] Building feature matrix...")

    symptom_list = sorted(list(all_symptoms))
    symptom_set_map = {s: i for i, s in enumerate(symptom_list)}

    n_features = len(symptom_list)
    X = np.zeros((len(df), n_features), dtype=np.float32)
    y = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        for col in symptom_cols:
            s = row[col]
            if pd.notna(s) and s in symptom_set_map:
                X[row_idx, symptom_set_map[s]] = 1
        y.append(row['Disease'])

    y = np.array(y)
    print(f"   Feature matrix: {X.shape}  (samples x symptoms)")
    return X, y, symptom_list


# ============================================================
# TRAIN ENSEMBLE
# ============================================================
def train_ensemble(X, y, symptom_list):
    """
    Train a VotingClassifier ensemble.
    
    The model object will have:
        - model.predict(X)         → server.py uses this
        - model.predict_proba(X)   → server.py uses this for top-5
        - model.n_features_in_     → server.py reads this for expected_features
    """
    print("[3/5] Training ensemble model...")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    unique_classes, counts = np.unique(y_enc, return_counts=True)
    min_samples = counts.min()
    print(f"   {len(unique_classes)} classes, min samples/class: {min_samples}")

    # Train/test split
    test_size = max(0.15, min(0.25, 50 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # Feature selection (removed to keep all symptoms)
    fs = None
    selected_symptom_list = symptom_list


    # --- Build ensemble ---
    # These are the same classifiers your original code used
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=30, min_samples_split=3,
        min_samples_leaf=1, max_features='sqrt', bootstrap=True,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )

    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=25, min_samples_split=3,
        min_samples_leaf=1, max_features='sqrt', bootstrap=True,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )

    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=10,
        min_samples_split=3, min_samples_leaf=1, subsample=0.8,
        random_state=42
    )

    nb = MultinomialNB(alpha=0.3)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('gb', gb), ('nb', nb)],
        voting='soft',
        weights=[4, 3, 3, 1]
    )

    print("   Training... (this may take a minute)")
    ensemble.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    )

    print(f"\n   ╔══════════════════════════════════╗")
    print(f"   ║  Accuracy:  {acc:.4f} ({acc * 100:.2f}%)     ║")
    print(f"   ║  F1 Score:  {f1:.4f}               ║")
    print(f"   ╚══════════════════════════════════╝")

    # Cross-validation
    cv_mean, cv_std = None, None
    n_splits = min(5, min_samples)
    if n_splits >= 2 and len(X) > 100:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X_cv = X if fs is None else fs.transform(X)
        scores = cross_val_score(ensemble, X_cv, y_enc, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_mean, cv_std = float(scores.mean()), float(scores.std())
        print(f"   Cross-Val:  {cv_mean:.4f} (±{cv_std * 2:.4f})")

    return ensemble, le, fs, selected_symptom_list, acc, f1, report, cv_mean, cv_std, len(X_train), len(X_test)


# ============================================================
# VERIFY — simulate what server.py does
# ============================================================
def verify_server_compatibility(output_dir):
    """
    Load the saved files exactly like server.py does and run a test prediction.
    This catches any mismatch before you start the server.
    """
    print("[5/5] Verifying server.py compatibility...")

    model_path = os.path.join(output_dir, 'disease_model.joblib')
    le_path = os.path.join(output_dir, 'label_encoder.joblib')
    sl_path = os.path.join(output_dir, 'symptom_list.json')

    # Load exactly like server.py load_models()
    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
    with open(sl_path) as f:
        symptom_list = json.load(f)

    # Check n_features_in_ — server.py reads this
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
    elif hasattr(model, 'estimators_'):
        first_est = model.estimators_[0]
        expected_features = first_est.n_features_in_ if hasattr(first_est, 'n_features_in_') else len(symptom_list)
    else:
        expected_features = len(symptom_list)

    print(f"   model.n_features_in_ = {expected_features}")
    print(f"   len(symptom_list)     = {len(symptom_list)}")

    assert expected_features == len(symptom_list), (
        f"MISMATCH! model expects {expected_features} features but "
        f"symptom_list has {len(symptom_list)}. Server.py will crash!"
    )

    # Simulate server.py predict-disease logic
    test_symptoms = ['headache', 'high_fever', 'vomiting', 'chills']
    normalized = [s.lower().strip().replace(' ', '_') for s in test_symptoms]

    feature_vector = np.zeros(expected_features)
    matched = []
    for symptom in normalized:
        if symptom in symptom_list:
            idx = symptom_list.index(symptom)
            if idx < expected_features:
                feature_vector[idx] = 1
                matched.append(symptom)

    feature_vector = feature_vector.reshape(1, -1)

    # predict_proba — server.py uses this
    probabilities = model.predict_proba(feature_vector)[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    disease_name = label_encoder.inverse_transform([predicted_idx])[0]

    top_indices = np.argsort(probabilities)[::-1][:5]
    top_preds = []
    for idx in top_indices:
        d = label_encoder.inverse_transform([idx])[0]
        top_preds.append(f"  {d}: {probabilities[idx]:.4f}")

    print(f"\n   Test input:    {test_symptoms}")
    print(f"   Matched:       {matched}")
    print(f"   Prediction:    {disease_name} ({confidence:.2%})")
    print(f"   Top 5:")
    for p in top_preds:
        print(f"     {p}")

    print("\n   ✅ Server.py compatibility VERIFIED — all checks passed!")
    return True


# ============================================================
# SAVE ALL ARTIFACTS
# ============================================================
def save_all(output_dir, model, le, fs, symptom_list, disease_info,
             acc, f1, report, cv_mean, cv_std, n_train, n_test):
    """Save all files that server.py expects."""
    print(f"\n[4/5] Saving to {os.path.abspath(output_dir)}/")
    os.makedirs(output_dir, exist_ok=True)

    # 1. disease_model.joblib — server.py: models['disease'] = joblib.load(...)
    model_path = os.path.join(output_dir, 'disease_model.joblib')
    joblib.dump(model, model_path)
    print(f"   ✅ disease_model.joblib          ({os.path.getsize(model_path) / 1024:.0f} KB)")

    # 2. label_encoder.joblib — server.py: models['label_encoder'] = joblib.load(...)
    le_path = os.path.join(output_dir, 'label_encoder.joblib')
    joblib.dump(le, le_path)
    print(f"   ✅ label_encoder.joblib")

    # 3. symptom_list.json — server.py: models['symptom_list'] = json.load(...)
    #    This is THE critical file. server.py uses:
    #        idx = symptom_list.index(symptom)
    #        feature_vector[idx] = 1
    sl_path = os.path.join(output_dir, 'symptom_list.json')
    with open(sl_path, 'w') as f:
        json.dump(symptom_list, f, indent=2)
    print(f"   ✅ symptom_list.json              ({len(symptom_list)} symptoms)")

    # 4. disease_info.json — optional but useful for frontend
    di_path = os.path.join(output_dir, 'disease_info.json')
    with open(di_path, 'w') as f:
        json.dump(disease_info, f, indent=2)
    print(f"   ✅ disease_info.json              ({len(disease_info)} diseases)")

    # 5. model_config.json — metadata
    config = {
        'model_type': 'VotingClassifier(RF+ET+GB+NB)',
        'voting': 'soft',
        'weights': [4, 3, 3, 1],
        'n_features_in': int(model.n_features_in_),
        'n_symptoms': len(symptom_list),
        'n_diseases': len(le.classes_),
        'diseases': [str(d) for d in le.classes_],
        'accuracy': float(acc),
        'f1_weighted': float(f1),
        'cv_accuracy_mean': cv_mean,
        'cv_accuracy_std': cv_std,
        'train_samples': n_train,
        'test_samples': n_test,
        'feature_selection_used': fs is not None,
        'compatible_with': 'server.py v4.1 /predict-disease endpoint'
    }
    cfg_path = os.path.join(output_dir, 'model_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   ✅ model_config.json")

    # 6. training_report.txt — human readable
    lines = [
        "=" * 70,
        "DISEASE PREDICTION MODEL — TRAINING REPORT",
        f"Compatible with: server.py v4.1 /predict-disease",
        "=" * 70,
        "",
        f"Accuracy:              {acc:.4f}  ({acc * 100:.2f}%)",
        f"F1 Score (weighted):   {f1:.4f}",
    ]
    if cv_mean is not None:
        lines.append(f"Cross-Val Accuracy:    {cv_mean:.4f} (±{cv_std * 2:.4f})")
    lines += [
        f"Training samples:      {n_train}",
        f"Test samples:          {n_test}",
        f"Number of symptoms:    {len(symptom_list)}",
        f"Number of diseases:    {len(le.classes_)}",
        "",
        "DISEASES:", ""
    ]
    for d in sorted(le.classes_):
        lines.append(f"  - {d}")
    lines += [
        "",
        "-" * 70,
        "CLASSIFICATION REPORT",
        "-" * 70,
        report,
        "",
        "-" * 70,
        "FILES FOR SERVER.PY",
        "-" * 70,
        "  disease_model.joblib   → models['disease']",
        "  label_encoder.joblib   → models['label_encoder']",
        "  symptom_list.json      → models['symptom_list']",
        "  disease_info.json      → disease descriptions",
        "  model_config.json      → metadata",
        "",
        "server.py MODEL_PATHS expects these in: ml_model/",
        "=" * 70,
    ]

    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"   ✅ training_report.txt")

    # NOTE: We do NOT save feature_selector.joblib because server.py
    # does NOT load or apply it. server.py builds the feature vector
    # directly from symptom_list. If feature selection was used during
    # training, the selected_symptom_list already reflects that.


# ============================================================
# QUICK TEST (optional CLI flag)
# ============================================================
def quick_test(output_dir, symptoms_str):
    """Test prediction from command line."""
    model = joblib.load(os.path.join(output_dir, 'disease_model.joblib'))
    le = joblib.load(os.path.join(output_dir, 'label_encoder.joblib'))
    with open(os.path.join(output_dir, 'symptom_list.json')) as f:
        symptom_list = json.load(f)

    symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms_str.split(',')]

    feature_vector = np.zeros(len(symptom_list))
    matched = []
    for s in symptoms:
        if s in symptom_list:
            feature_vector[symptom_list.index(s)] = 1
            matched.append(s)

    if not matched:
        print(f"\n❌ None of {symptoms} found in symptom list!")
        return

    probs = model.predict_proba(feature_vector.reshape(1, -1))[0]
    top5 = np.argsort(probs)[::-1][:5]

    print(f"\n{'=' * 50}")
    print(f"Input:   {symptoms}")
    print(f"Matched: {matched}")
    print(f"{'=' * 50}")
    for rank, idx in enumerate(top5, 1):
        disease = le.inverse_transform([idx])[0]
        conf = probs[idx]
        bar = '█' * int(conf * 30)
        print(f"  {rank}. {disease:<40s} {conf:6.2%}  {bar}")
    print(f"{'=' * 50}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train disease prediction model for server.py'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Where to save model files (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--test', type=str, default=None,
        help='Comma-separated symptoms to test after training, e.g. "headache,high_fever,vomiting"'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Path to existing dataset CSV (optional)'
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    print("=" * 60)
    print("  DISEASE PREDICTION MODEL — TRAINING PIPELINE")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print()

    # Override dataset dir if provided
    global DATASET_DIR
    if args.dataset:
        DATASET_DIR = os.path.dirname(os.path.abspath(args.dataset))

    # Step 1: Load/generate data
    df, symptom_cols, all_symptoms = load_dataset()

    # Step 2: Build features
    X, y, symptom_list = build_features(df, symptom_cols, all_symptoms)

    # Step 3: Train
    (model, le, fs, selected_symptoms,
     acc, f1, report, cv_mean, cv_std,
     n_train, n_test) = train_ensemble(X, y, symptom_list)

    # Step 4: Save
    disease_info = build_disease_info()
    save_all(
        output_dir, model, le, fs, selected_symptoms, disease_info,
        acc, f1, report, cv_mean, cv_std, n_train, n_test
    )

    # Step 5: Verify
    verify_server_compatibility(output_dir)

    print("\n" + "=" * 60)
    print("  ✅ DONE! Model files ready for server.py")
    print(f"  📁 {output_dir}/")
    print("     ├── disease_model.joblib")
    print("     ├── label_encoder.joblib")
    print("     ├── symptom_list.json")
    print("     ├── disease_info.json")
    print("     ├── model_config.json")
    print("     └── training_report.txt")
    print("=" * 60)

    # Optional quick test
    if args.test:
        quick_test(output_dir, args.test)


if __name__ == '__main__':
    main()