import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, 
    StratifiedKFold, GridSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, 
    roc_auc_score, confusion_matrix,
    precision_recall_curve, f1_score,
    matthews_corrcoef
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
HEART_DATASET_PATH = os.path.join(DATASET_DIR, 'heart.csv')

# Output paths
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

HEART_MODEL_PATH = os.path.join(MODEL_DIR, 'heart_risk_model.joblib')
HEART_SCALER_PATH = os.path.join(MODEL_DIR, 'heart_scaler.joblib')
HEART_FEATURES_PATH = os.path.join(MODEL_DIR, 'heart_features.json')
HEART_METRICS_PATH = os.path.join(MODEL_DIR, 'heart_metrics.json')

# Feature descriptions
HEART_FEATURES = {
    'age': {'description': 'Age in years', 'type': 'numeric', 'range': [20, 100]},
    'sex': {'description': 'Sex (0=Female, 1=Male)', 'type': 'categorical', 'options': [0, 1]},
    'cp': {
        'description': 'Chest pain type', 'type': 'categorical', 'options': [0, 1, 2, 3],
        'labels': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
    },
    'trestbps': {'description': 'Resting blood pressure (mm Hg)', 'type': 'numeric', 'range': [90, 200]},
    'chol': {'description': 'Serum cholesterol (mg/dl)', 'type': 'numeric', 'range': [100, 600]},
    'fbs': {'description': 'Fasting blood sugar > 120 mg/dl', 'type': 'categorical', 'options': [0, 1]},
    'restecg': {
        'description': 'Resting ECG results', 'type': 'categorical', 'options': [0, 1, 2],
        'labels': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'LV hypertrophy'}
    },
    'thalach': {'description': 'Maximum heart rate achieved', 'type': 'numeric', 'range': [60, 220]},
    'exang': {'description': 'Exercise induced angina', 'type': 'categorical', 'options': [0, 1]},
    'oldpeak': {'description': 'ST depression induced by exercise', 'type': 'numeric', 'range': [0, 7]},
    'slope': {
        'description': 'Slope of peak exercise ST segment', 'type': 'categorical', 'options': [0, 1, 2],
        'labels': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
    },
    'ca': {'description': 'Number of major vessels (0-3)', 'type': 'categorical', 'options': [0, 1, 2, 3]},
    'thal': {
        'description': 'Thalassemia', 'type': 'categorical', 'options': [0, 1, 2, 3],
        'labels': {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect', 3: 'Unknown'}
    }
}


# =====================================================================
# DATA GENERATION
# =====================================================================

def create_sample_heart_dataset():
    """Generate realistic synthetic heart disease dataset"""
    print("Creating enhanced sample dataset...")
    
    np.random.seed(42)
    n_samples = 1500
    
    # Generate correlated features based on medical knowledge
    age = np.random.normal(55, 10, n_samples).clip(29, 80).astype(int)
    sex = np.random.binomial(1, 0.68, n_samples)  # More males in heart disease studies
    
    # Chest pain correlated with age and sex
    cp_probs = np.zeros((n_samples, 4))
    for i in range(n_samples):
        if age[i] > 60:
            cp_probs[i] = [0.15, 0.25, 0.25, 0.35]  # Older: more likely asymptomatic
        else:
            cp_probs[i] = [0.25, 0.30, 0.30, 0.15]
        cp_probs[i] = cp_probs[i] / cp_probs[i].sum()
    
    cp = np.array([np.random.choice(4, p=cp_probs[i]) for i in range(n_samples)])
    
    # Blood pressure increases with age
    trestbps_base = 100 + (age - 29) * 0.8
    trestbps = (trestbps_base + np.random.normal(0, 15, n_samples)).clip(94, 200).astype(int)
    
    # Cholesterol increases with age
    chol_base = 180 + (age - 29) * 1.5
    chol = (chol_base + np.random.normal(0, 40, n_samples)).clip(126, 564).astype(int)
    
    # Other features
    fbs = np.random.binomial(1, 0.15, n_samples)
    restecg = np.random.choice(3, n_samples, p=[0.50, 0.35, 0.15])
    
    # Max heart rate decreases with age
    thalach_base = 220 - age
    thalach = (thalach_base + np.random.normal(0, 20, n_samples)).clip(71, 202).astype(int)
    
    exang = np.random.binomial(1, 0.33, n_samples)
    oldpeak = np.abs(np.random.normal(1.0, 1.5, n_samples)).clip(0, 6.2).round(1)
    slope = np.random.choice(3, n_samples, p=[0.25, 0.50, 0.25])
    ca = np.random.choice(4, n_samples, p=[0.55, 0.25, 0.15, 0.05])
    thal = np.random.choice(4, n_samples, p=[0.15, 0.15, 0.65, 0.05])
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    })
    
    # Create target with realistic medical risk factors
    risk_score = (
        (df['age'] > 55).astype(float) * 0.18 +
        (df['sex'] == 1).astype(float) * 0.15 +
        (df['cp'] == 0).astype(float) * 0.25 +
        (df['cp'] == 3).astype(float) * 0.10 +
        (df['trestbps'] > 140).astype(float) * 0.20 +
        (df['chol'] > 240).astype(float) * 0.15 +
        (df['fbs'] == 1).astype(float) * 0.12 +
        (df['thalach'] < 120).astype(float) * 0.20 +
        (df['exang'] == 1).astype(float) * 0.25 +
        (df['oldpeak'] > 2).astype(float) * 0.20 +
        (df['slope'] == 2).astype(float) * 0.15 +
        (df['ca'] > 0).astype(float) * 0.30 +
        (df['thal'] == 2).astype(float) * 0.20
    )
    
    # Add realistic noise
    risk_score += np.random.normal(0, 0.15, n_samples)
    
    # Create binary target with threshold
    threshold = np.percentile(risk_score, 55)  # ~45% positive class
    df['target'] = (risk_score > threshold).astype(int)
    
    # Save dataset
    os.makedirs(DATASET_DIR, exist_ok=True)
    df.to_csv(HEART_DATASET_PATH, index=False)
    print(f"✓ Sample dataset created: {HEART_DATASET_PATH}")
    print(f"  Samples: {len(df)} | Positive class: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    
    return df


# =====================================================================
# DATA PREPROCESSING
# =====================================================================

def preprocess_heart_data(df):
    """Enhanced preprocessing with comprehensive data cleaning"""
    df = df.copy()
    
    # Map string values to numeric for all categorical columns
    mappings = {
        'sex': {'Male': 1, 'male': 1, 'M': 1, 'm': 1, 'Female': 0, 'female': 0, 'F': 0, 'f': 0},
        'cp': {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3,
               'ta': 0, 'ata': 1, 'nap': 2, 'asy': 3},
        'fbs': {True: 1, False: 0, 'True': 1, 'False': 0, 'Yes': 1, 'No': 0, 'yes': 1, 'no': 0},
        'exang': {'Yes': 1, 'No': 0, 'Y': 1, 'N': 0, 'yes': 1, 'no': 0},
        'restecg': {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2,
                    'Normal': 0, 'ST': 1, 'LVH': 2},
        'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2, 'Up': 0, 'Flat': 1, 'Down': 2},
        'thal': {'normal': 0, 'fixed defect': 1, 'reversible defect': 2, 'reversable defect': 2,
                 'fixed': 1, 'reversible': 2}
    }
    
    for col, mapping in mappings.items():
        if col in df.columns and df[col].dtype in ['object', 'bool']:
            if df[col].dtype == 'object':
                df[col] = df[col].str.lower() if df[col].dtype == 'object' else df[col]
            df[col] = df[col].map(mapping)
            print(f"✓ Converted '{col}' to numeric")
    
    # Handle column name variations
    column_aliases = {
        'thalch': 'thalach', 'num': 'target', 'condition': 'target',
        'disease': 'target', 'heart_disease': 'target'
    }
    
    for old, new in column_aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
            print(f"✓ Mapped '{old}' → '{new}'")
    
    # Convert multi-class target to binary
    if 'target' in df.columns and df['target'].max() > 1:
        df['target'] = (df['target'] > 0).astype(int)
        print("✓ Converted multi-class target to binary")
    
    # Handle missing values intelligently
    for col in df.columns:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                df[col].fillna(mode_val, inplace=True)
            print(f"✓ Filled {null_count} missing values in '{col}'")
    
    return df


def load_heart_data():
    """Load and validate heart disease dataset"""
    if not os.path.exists(HEART_DATASET_PATH):
        print(f"⚠ Dataset not found: {HEART_DATASET_PATH}")
        return create_sample_heart_dataset()
    
    print(f"Loading dataset: {HEART_DATASET_PATH}")
    df = pd.read_csv(HEART_DATASET_PATH)
    print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    
    df = preprocess_heart_data(df)
    
    # Drop rows with remaining NaN
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"✓ Dropped {initial_len - len(df)} incomplete rows")
    
    if 'target' not in df.columns:
        print("❌ ERROR: No target column found!")
        return create_sample_heart_dataset()
    
    print(f"✓ Final dataset: {len(df)} records")
    print(f"  Target distribution: {dict(df['target'].value_counts())}")
    
    return df


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================

def engineer_features(df, feature_cols):
    """Create advanced features based on medical domain knowledge"""
    X = df[feature_cols].copy()
    
    # Interaction features (clinically relevant combinations)
    if 'age' in feature_cols and 'thalach' in feature_cols:
        X['age_thalach_ratio'] = X['age'] / (X['thalach'] + 1)
    
    if 'age' in feature_cols and 'chol' in feature_cols:
        X['age_chol_interaction'] = X['age'] * X['chol'] / 1000
    
    if 'trestbps' in feature_cols and 'chol' in feature_cols:
        X['bp_chol_risk'] = (X['trestbps'] > 140).astype(int) * (X['chol'] > 240).astype(int)
    
    if 'exang' in feature_cols and 'oldpeak' in feature_cols:
        X['exang_oldpeak_risk'] = X['exang'] * X['oldpeak']
    
    if 'age' in feature_cols:
        X['age_squared'] = X['age'] ** 2
        X['age_group'] = pd.cut(X['age'], bins=[0, 45, 55, 65, 100], labels=[0, 1, 2, 3]).astype(int)
    
    if 'chol' in feature_cols:
        X['chol_risk'] = (X['chol'] > 240).astype(int)
    
    if 'trestbps' in feature_cols:
        X['hypertension'] = (X['trestbps'] > 140).astype(int)
    
    # Heart rate reserve (if both age and thalach available)
    if 'age' in feature_cols and 'thalach' in feature_cols:
        max_hr = 220 - X['age']
        X['hr_reserve_pct'] = (X['thalach'] / max_hr * 100).clip(0, 100)
    
    return X


# =====================================================================
# MODEL TRAINING
# =====================================================================

def train_heart_model():
    """Train enhanced heart disease prediction model with ensemble methods"""
    print("\n" + "=" * 70)
    print("ENHANCED HEART DISEASE PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    df = load_heart_data()
    if len(df) < 50:
        print("❌ Insufficient data for training!")
        return None, None
    
    # Define features
    base_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    available_features = [f for f in base_features if f in df.columns]
    
    if len(available_features) < 5:
        print(f"⚠ Warning: Only {len(available_features)} features available")
        # Add any numeric columns
        for col in df.columns:
            if col not in ['target', 'id', 'dataset'] and col not in available_features:
                if df[col].dtype in ['float64', 'int64']:
                    available_features.append(col)
    
    print(f"\n✓ Using {len(available_features)} base features: {available_features}")
    
    # Engineer features
    X_df = engineer_features(df, available_features)
    y = df['target'].values
    
    print(f"✓ Engineered features: {X_df.shape[1]} total features")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===== Model 1: Optimized Random Forest =====
    print("\n" + "-" * 70)
    print("Training Random Forest...")
    print("-" * 70)
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        criterion='gini'
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # ===== Model 2: Gradient Boosting =====
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # ===== Model 3: Logistic Regression =====
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # ===== Ensemble: Voting Classifier =====
    print("\n" + "-" * 70)
    print("Creating Ensemble Model...")
    print("-" * 70)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model)
        ],
        voting='soft',
        weights=[2, 2, 1]  # Give more weight to RF and GB
    )
    ensemble_model.fit(X_train_scaled, y_train)
    
    # ===== Evaluation =====
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 70)
    
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Logistic Regression': lr_model,
        'Ensemble': ensemble_model
    }
    
    best_model = None
    best_score = 0
    metrics_summary = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  ROC-AUC:   {roc:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  MCC:       {mcc:.4f}")
        
        metrics_summary[name] = {
            'accuracy': float(acc),
            'roc_auc': float(roc),
            'f1_score': float(f1),
            'mcc': float(mcc)
        }
        
        if roc > best_score:
            best_score = roc
            best_model = model
            best_model_name = name
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name} (ROC-AUC: {best_score:.4f})")
    print(f"{'='*70}")
    
    # Cross-validation on best model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                 cv=skf, scoring='roc_auc', n_jobs=-1)
    
    print(f"\n5-Fold Cross-Validation (ROC-AUC):")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std:  {cv_scores.std():.4f}")
    print(f"  Scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Confusion Matrix
    y_pred_final = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}")
    print(f"  FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}")
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n{'='*70}")
        print("TOP 10 FEATURE IMPORTANCE")
        print(f"{'='*70}")
        
        feature_importance = pd.DataFrame({
            'feature': X_df.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        for idx, row in feature_importance.iterrows():
            bar = '█' * int(row['importance'] * 100)
            print(f"  {row['feature']:25s} {bar} {row['importance']:.4f}")
    
    # ===== Save Models and Artifacts =====
    print(f"\n{'='*70}")
    print("SAVING MODEL ARTIFACTS")
    print(f"{'='*70}")
    
    joblib.dump(best_model, HEART_MODEL_PATH)
    print(f"✓ Model saved: {HEART_MODEL_PATH}")
    
    joblib.dump(scaler, HEART_SCALER_PATH)
    print(f"✓ Scaler saved: {HEART_SCALER_PATH}")
    
    # Save feature information
    feature_info = {
        'base_features': available_features,
        'engineered_features': list(X_df.columns),
        'feature_details': HEART_FEATURES,
        'model_type': best_model_name,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    with open(HEART_FEATURES_PATH, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✓ Features saved: {HEART_FEATURES_PATH}")
    
    # Save metrics
    with open(HEART_METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✓ Metrics saved: {HEART_METRICS_PATH}")
    
    print(f"\n{'='*70}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*70}\n")
    
    return best_model, scaler


# =====================================================================
# PREDICTION
# =====================================================================

def predict_heart_disease(model, scaler, features):
    """
    Make enhanced prediction for heart disease
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        features: Dictionary or array of feature values (base features only)
    
    Returns:
        Dictionary with prediction, probability, and risk assessment
    """
    # Load feature info to know what features to expect
    try:
        with open(HEART_FEATURES_PATH, 'r') as f:
            feature_info = json.load(f)
        base_features = feature_info['base_features']
    except:
        base_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Convert features to DataFrame if array/list
    if isinstance(features, dict):
        feature_df = pd.DataFrame([features])
    else:
        feature_df = pd.DataFrame([features], columns=base_features)
    
    # Apply the SAME feature engineering as during training
    X_engineered = engineer_features(feature_df, base_features)
    
    # Scale the engineered features
    scaled_features = scaler.transform(X_engineered)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    disease_prob = float(probabilities[1])
    
    # Risk assessment
    if disease_prob >= 0.75:
        risk_level = 'Very High'
        recommendation = 'Immediate medical consultation recommended'
    elif disease_prob >= 0.60:
        risk_level = 'High'
        recommendation = 'Schedule medical checkup soon'
    elif disease_prob >= 0.40:
        risk_level = 'Moderate'
        recommendation = 'Regular monitoring advised'
    elif disease_prob >= 0.25:
        risk_level = 'Low-Moderate'
        recommendation = 'Maintain healthy lifestyle'
    else:
        risk_level = 'Low'
        recommendation = 'Continue preventive care'
    
    return {
        'prediction': int(prediction),
        'probability': disease_prob,
        'risk_level': risk_level,
        'confidence': float(max(probabilities)),
        'recommendation': recommendation,
        'probabilities': {
            'no_disease': float(probabilities[0]),
            'disease': float(probabilities[1])
        }
    }



if __name__ == '__main__':
    try:
        model, scaler = train_heart_model()
        
        if model is not None and scaler is not None:
            print("\n" + "=" * 70)
            print("Testing prediction with sample data...")
            print("=" * 70)
            
            # Test prediction with sample
            sample_features = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
            result = predict_heart_disease(model, scaler, sample_features)
            
            print(f"\nSample Prediction:")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Disease Probability: {result['probability']:.1%}")
            print(f"  Recommendation: {result['recommendation']}")
            
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()