import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_selection import SelectKBest, chi2
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')

# Output paths
MODEL_PATH = os.path.join(SCRIPT_DIR, 'disease_model.joblib')
LABEL_ENCODER_PATH = os.path.join(SCRIPT_DIR, 'label_encoder.joblib')
SYMPTOM_LIST_PATH = os.path.join(SCRIPT_DIR, 'symptom_list.json')
FEATURE_SELECTOR_PATH = os.path.join(SCRIPT_DIR, 'feature_selector.joblib')  # NEW!
MODEL_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'disease_model_config.json')  # NEW!


def clean_symptom(symptom):
    """Clean and normalize symptom names"""
    if pd.isna(symptom) or symptom is None:
        return None
    symptom = str(symptom).strip().lower()
    symptom = symptom.replace(' ', '_').replace('-', '_')
    symptom = ''.join(c for c in symptom if c.isalnum() or c == '_')
    return symptom if symptom and symptom != '_' else None


def create_enhanced_sample_dataset():
    """Create enhanced sample dataset with more diseases and realistic symptom patterns"""
    print("Creating enhanced sample dataset...")
    
    # Comprehensive disease-symptom mappings
    disease_symptoms = {
        # Infectious diseases
        'Common Cold': ['continuous_sneezing', 'chills', 'fatigue', 'cough', 'high_fever', 
                       'headache', 'malaise', 'sore_throat', 'runny_nose', 'watering_from_eyes'],
        'Influenza': ['high_fever', 'chills', 'fatigue', 'cough', 'muscle_aches',
                     'headache', 'sore_throat', 'weakness_of_one_body_side', 'loss_of_appetite'],
        'Pneumonia': ['chills', 'fatigue', 'cough', 'high_fever', 'breathlessness',
                     'sweating', 'chest_pain', 'fast_heart_rate', 'phlegm'],
        'Tuberculosis': ['chills', 'vomiting', 'fatigue', 'cough', 'high_fever',
                        'breathlessness', 'loss_of_appetite', 'weight_loss', 'mild_fever'],
        'Malaria': ['chills', 'vomiting', 'high_fever', 'sweating', 'headache',
                   'nausea', 'diarrhoea', 'muscle_pain'],
        'Dengue': ['skin_rash', 'chills', 'joint_pain', 'vomiting', 'fatigue',
                  'high_fever', 'headache', 'muscle_pain', 'red_spots_over_body'],
        'Typhoid': ['chills', 'vomiting', 'fatigue', 'high_fever', 'headache',
                   'constipation', 'abdominal_pain', 'diarrhoea', 'toxic_look'],
        'Chicken pox': ['itching', 'skin_rash', 'fatigue', 'high_fever', 'red_spots_over_body',
                       'loss_of_appetite', 'headache'],
        
        # Gastrointestinal diseases
        'Gastroenteritis': ['vomiting', 'diarrhoea', 'dehydration', 'sunken_eyes', 'high_fever',
                           'abdominal_pain', 'nausea', 'loss_of_appetite'],
        'GERD': ['stomach_pain', 'acidity', 'vomiting', 'chest_pain', 'ulcers_on_tongue',
                'cough', 'burning_micturition'],
        'Peptic ulcer disease': ['vomiting', 'loss_of_appetite', 'abdominal_pain', 'passage_of_gases',
                                'internal_itching', 'indigestion', 'nausea'],
        'Chronic cholestasis': ['itching', 'vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite',
                               'abdominal_pain', 'yellowing_of_eyes'],
        
        # Liver diseases
        'Jaundice': ['itching', 'vomiting', 'fatigue', 'yellowish_skin', 'dark_urine',
                    'yellowing_of_eyes', 'loss_of_appetite', 'abdominal_pain'],
        'Hepatitis A': ['joint_pain', 'vomiting', 'yellowish_skin', 'dark_urine', 'nausea',
                       'loss_of_appetite', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellowing_of_eyes'],
        'Hepatitis B': ['itching', 'fatigue', 'lethargy', 'yellowish_skin', 'dark_urine',
                       'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'],
        'Hepatitis C': ['fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite',
                       'yellowing_of_eyes', 'family_history', 'dark_urine'],
        'Hepatitis D': ['joint_pain', 'vomiting', 'fatigue', 'high_fever', 'yellowish_skin',
                       'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'],
        'Hepatitis E': ['joint_pain', 'vomiting', 'fatigue', 'yellowish_skin', 'dark_urine',
                       'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes', 'acute_liver_failure'],
        'Alcoholic hepatitis': ['vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach',
                               'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload'],
        
        # Respiratory diseases
        'Bronchial Asthma': ['breathlessness', 'cough', 'fatigue', 'high_fever', 'mucoid_sputum',
                            'chest_pain', 'fast_heart_rate'],
        
        # Cardiovascular diseases
        'Heart attack': ['vomiting', 'breathlessness', 'sweating', 'chest_pain', 'fast_heart_rate',
                        'weakness_in_limbs'],
        'Hypertension': ['headache', 'chest_pain', 'dizziness', 'lack_of_concentration',
                        'blurred_and_distorted_vision'],
        
        # Metabolic diseases
        'Diabetes': ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level',
                    'increased_appetite', 'polyuria', 'excessive_hunger'],
        'Hyperthyroidism': ['fatigue', 'mood_swings', 'weight_loss', 'restlessness', 
                           'fast_heart_rate', 'excessive_hunger', 'sweating', 'diarrhoea'],
        'Hypothyroidism': ['fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings',
                          'lethargy', 'brittle_nails', 'swollen_extremeties', 'depression'],
        
        # Skin diseases
        'Fungal infection': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches',
                            'scurring'],
        'Acne': ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'],
        'Impetigo': ['skin_rash', 'high_fever', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'],
        'Psoriasis': ['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting',
                     'small_dents_in_nails', 'inflammatory_nails'],
        
        # Allergic conditions
        'Allergy': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes',
                   'runny_nose', 'congestion'],
        'Drug Reaction': ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition',
                         'spotting_urination'],
        
        # Urinary diseases
        'Urinary tract infection': ['burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine',
                                   'continuous_feel_of_urine'],
        
        # Neurological diseases
        'Migraine': ['headache', 'acidity', 'indigestion', 'blurred_and_distorted_vision',
                    'visual_disturbances', 'stiff_neck', 'depression', 'irritability'],
        'Cervical spondylosis': ['back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness',
                                'loss_of_balance'],
        'Paralysis (brain hemorrhage)': ['vomiting', 'headache', 'weakness_of_one_body_side', 'altered_sensorium'],
        
        # Arthritis
        'Arthritis': ['muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
                     'painful_walking', 'joint_pain'],
        'Osteoarthristis': ['joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain',
                           'swelling_joints', 'painful_walking'],
        
        # Other diseases
        'Varicose veins': ['fatigue', 'cramps', 'bruising', 'obesity', 'swollen_legs',
                          'swollen_blood_vessels', 'prominent_veins_on_calf'],
        'AIDS': ['muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts'],
        'Dimorphic hemmorhoids(piles)': ['constipation', 'pain_during_bowel_movements', 'pain_in_anal_region',
                                         'bloody_stool', 'irritation_in_anus']
    }
    
    data = {
        'Disease': [],
        'Symptom_1': [], 'Symptom_2': [], 'Symptom_3': [], 'Symptom_4': [], 'Symptom_5': [],
        'Symptom_6': [], 'Symptom_7': [], 'Symptom_8': [], 'Symptom_9': [], 'Symptom_10': []
    }
    
    # Generate samples for each disease
    np.random.seed(42)
    for disease, symptoms in disease_symptoms.items():
        # Create 15-20 samples per disease with variations
        for _ in range(np.random.randint(15, 21)):
            data['Disease'].append(disease)
            
            # Randomly select 3-8 symptoms
            n_symptoms = np.random.randint(3, min(9, len(symptoms) + 1))
            selected_symptoms = np.random.choice(symptoms, size=n_symptoms, replace=False).tolist()
            
            # Fill symptom columns
            for i in range(1, 11):
                if i <= len(selected_symptoms):
                    data[f'Symptom_{i}'].append(selected_symptoms[i-1])
                else:
                    data[f'Symptom_{i}'].append(None)
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    os.makedirs(DATASET_DIR, exist_ok=True)
    sample_path = os.path.join(DATASET_DIR, 'dataset.csv')
    df.to_csv(sample_path, index=False)
    print(f"✓ Enhanced dataset saved: {sample_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique diseases: {df['Disease'].nunique()}")
    
    return df


def load_dataset():
    """Load Disease Symptom Description Dataset"""
    dataset_path = os.path.join(DATASET_DIR, 'dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found, creating enhanced version...")
        return create_enhanced_sample_dataset()
    
    try:
        df = pd.read_csv(dataset_path)
        
        if len(df) == 0 or 'Disease' not in df.columns:
            return create_enhanced_sample_dataset()
        
        symptom_cols = [col for col in df.columns if 'Symptom' in col or 'symptom' in col]
        
        if len(symptom_cols) == 0:
            return create_enhanced_sample_dataset()
        
        # Clean disease names
        df['Disease'] = df['Disease'].str.strip()
        
        # Collect unique symptoms
        all_symptoms = set()
        for col in symptom_cols:
            df[col] = df[col].apply(clean_symptom)
            symptoms = df[col].dropna().unique()
            all_symptoms.update(s for s in symptoms if s)
        
        print(f"✓ Dataset loaded: {len(df)} records, {len(all_symptoms)} unique symptoms")
        
        return df, symptom_cols, all_symptoms
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return create_enhanced_sample_dataset()


def create_training_data(df, symptom_cols, all_symptoms):
    """Create unified training dataset"""
    
    symptom_list = sorted(list(all_symptoms))
    symptom_to_idx = {s: i for i, s in enumerate(symptom_list)}
    
    X = []
    y = []
    
    for _, row in df.iterrows():
        feature_vector = np.zeros(len(symptom_list), dtype=np.int8)
        
        for col in symptom_cols:
            symptom = row[col]
            if symptom and symptom in symptom_to_idx:
                feature_vector[symptom_to_idx[symptom]] = 1
        
        if feature_vector.sum() >= 2:  # At least 2 symptoms
            X.append(feature_vector)
            y.append(row['Disease'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✓ Training data prepared")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(symptom_list)}")
    print(f"  Diseases: {len(np.unique(y))}")
    
    return X, y, symptom_list


def train_model(X, y, symptom_list):
    """Train enhanced ensemble classifier - FIXED to save selector"""
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    n_classes = len(label_encoder.classes_)
    print(f"\n✓ Disease classes: {n_classes}")
    
    # Check class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    min_samples = counts.min()
    print(f"  Min samples per class: {min_samples}")
    
    # Split data
    test_size = min(0.2, max(0.1, 10 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"\n✓ Data split")
    print(f"  Training: {len(X_train)}")
    print(f"  Testing: {len(X_test)}")
    
    # Feature selection - FIXED: Save the selector
    feature_selector = None
    selected_symptom_list = symptom_list  # Default: use all symptoms
    use_feature_selection = False  # Disable by default for simplicity
    
    if use_feature_selection and X_train.shape[1] > 100:
        k_features = min(100, X_train.shape[1])
        feature_selector = SelectKBest(chi2, k=k_features)
        X_train = feature_selector.fit_transform(X_train, y_train)
        X_test = feature_selector.transform(X_test)
        
        # Get selected feature indices
        selected_indices = feature_selector.get_support(indices=True)
        selected_symptom_list = [symptom_list[i] for i in selected_indices]
        
        print(f"✓ Selected top {k_features} features")
    
    # Build ensemble
    print("\n" + "-"*70)
    print("Building Ensemble Model")
    print("-"*70)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    et = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    nb = MultinomialNB(alpha=0.5)
    
    # Ensemble with optimized weights
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('et', et),
            ('gb', gb),
            ('nb', nb)
        ],
        voting='soft',
        weights=[3, 2, 3, 1]
    )
    
    print("Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)
    print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (weighted): {f1:.4f}")
    
    # Cross-validation
    if len(X) > 100:
        n_splits = min(5, min_samples)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            # Use original X if no feature selection
            X_cv = X if feature_selector is None else feature_selector.transform(X)
            cv_scores = cross_val_score(ensemble, X_cv, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
            print(f"\nCross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return ensemble, label_encoder, feature_selector, selected_symptom_list, accuracy


def main():
    """Main training pipeline"""
    print("="*70)
    print("DISEASE PREDICTION MODEL TRAINING - FIXED VERSION")
    print("="*70)
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Load dataset
    result = load_dataset()
    
    if isinstance(result, tuple):
        df, symptom_cols, all_symptoms = result
    else:
        df = result
        symptom_cols = [col for col in df.columns if 'Symptom' in col]
        all_symptoms = set()
        for col in symptom_cols:
            df[col] = df[col].apply(clean_symptom)
            symptoms = df[col].dropna().unique()
            all_symptoms.update(s for s in symptoms if s)
    
    # Create training data
    X, y, symptom_list = create_training_data(df, symptom_cols, all_symptoms)
    
    if len(X) < 20:
        print("ERROR: Insufficient training data!")
        return
    
    # Train model
    model, label_encoder, feature_selector, selected_symptom_list, accuracy = train_model(X, y, symptom_list)
    
    # Save artifacts
    print("\n" + "="*70)
    print("SAVING MODEL ARTIFACTS")
    print("="*70)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Model saved: {MODEL_PATH}")
    
    # Save label encoder
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"✓ Label encoder saved: {LABEL_ENCODER_PATH}")
    
    # Save feature selector if used
    if feature_selector is not None:
        joblib.dump(feature_selector, FEATURE_SELECTOR_PATH)
        print(f"✓ Feature selector saved: {FEATURE_SELECTOR_PATH}")
    else:
        # Delete old feature selector if exists
        if os.path.exists(FEATURE_SELECTOR_PATH):
            os.remove(FEATURE_SELECTOR_PATH)
        print("✓ Feature selection: DISABLED (using all symptoms)")
    
    # IMPORTANT: Save the CORRECT symptom list that matches the model
    with open(SYMPTOM_LIST_PATH, 'w') as f:
        json.dump(selected_symptom_list, f, indent=2)
    print(f"✓ Symptom list saved: {SYMPTOM_LIST_PATH}")
    print(f"  ({len(selected_symptom_list)} symptoms)")
    
    # Save config for verification
    config = {
        'model_features': model.n_features_in_,
        'symptom_count': len(selected_symptom_list),
        'disease_count': len(label_encoder.classes_),
        'diseases': list(label_encoder.classes_),
        'accuracy': float(accuracy),
        'feature_selection_enabled': feature_selector is not None
    }
    
    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {MODEL_CONFIG_PATH}")
    
    # Verification
    print("\n" + "="*70)
    print("✓ VERIFICATION")
    print("="*70)
    print(f"  Model expects: {model.n_features_in_} features")
    print(f"  Symptom list:  {len(selected_symptom_list)} symptoms")
    print(f"  MATCH: {'✓ YES' if model.n_features_in_ == len(selected_symptom_list) else '✗ NO'}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()