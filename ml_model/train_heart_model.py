import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix,
    f1_score, matthews_corrcoef
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
HEART_DATASET_PATH = os.path.join(DATASET_DIR, 'heart.csv')

MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'ml_model')
os.makedirs(MODEL_DIR, exist_ok=True)

HEART_MODEL_PATH = os.path.join(MODEL_DIR, 'heart_disease_model.joblib')
HEART_SCALER_PATH = os.path.join(MODEL_DIR, 'heart_scaler.joblib')
HEART_FEATURES_PATH = os.path.join(MODEL_DIR, 'heart_features.json')
HEART_METRICS_PATH = os.path.join(MODEL_DIR, 'heart_metrics.json')

# ── The 13 features that server.py sends ────────────────────────────────────
# This list MUST match the required_features list in server.py's /predict-heart
BASE_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

HEART_FEATURES = {
    'age':      {'description': 'Age in years',                         'type': 'numeric',     'range': [20, 100]},
    'sex':      {'description': 'Sex (0=Female, 1=Male)',               'type': 'categorical', 'options': [0, 1]},
    'cp':       {'description': 'Chest pain type (0-3)',                'type': 'categorical', 'options': [0, 1, 2, 3]},
    'trestbps': {'description': 'Resting blood pressure (mm Hg)',       'type': 'numeric',     'range': [90, 200]},
    'chol':     {'description': 'Serum cholesterol (mg/dl)',            'type': 'numeric',     'range': [100, 600]},
    'fbs':      {'description': 'Fasting blood sugar > 120 mg/dl',     'type': 'categorical', 'options': [0, 1]},
    'restecg':  {'description': 'Resting ECG results (0-2)',           'type': 'categorical', 'options': [0, 1, 2]},
    'thalach':  {'description': 'Maximum heart rate achieved',          'type': 'numeric',     'range': [60, 220]},
    'exang':    {'description': 'Exercise induced angina',              'type': 'categorical', 'options': [0, 1]},
    'oldpeak':  {'description': 'ST depression induced by exercise',    'type': 'numeric',     'range': [0, 7]},
    'slope':    {'description': 'Slope of peak exercise ST segment',    'type': 'categorical', 'options': [0, 1, 2]},
    'ca':       {'description': 'Number of major vessels (0-3)',        'type': 'categorical', 'options': [0, 1, 2, 3]},
    'thal':     {'description': 'Thalassemia (0-3)',                    'type': 'categorical', 'options': [0, 1, 2, 3]}
}


# ═════════════════════════════════════════════════════════════════════════════
#                     SYNTHETIC DATA (fallback)
# ═════════════════════════════════════════════════════════════════════════════

def create_sample_heart_dataset():
    """Generate realistic synthetic heart disease dataset."""
    print("Creating sample dataset...")
    np.random.seed(42)
    n = 1500

    age = np.random.normal(55, 10, n).clip(29, 80).astype(int)
    sex = np.random.binomial(1, 0.68, n)

    cp_probs = np.where(
        age[:, None] > 60,
        np.tile([0.15, 0.25, 0.25, 0.35], (n, 1)),
        np.tile([0.25, 0.30, 0.30, 0.15], (n, 1))
    )
    cp_probs = cp_probs / cp_probs.sum(axis=1, keepdims=True)
    cp = np.array([np.random.choice(4, p=cp_probs[i]) for i in range(n)])

    trestbps = (100 + (age - 29) * 0.8 + np.random.normal(0, 15, n)).clip(94, 200).astype(int)
    chol = (180 + (age - 29) * 1.5 + np.random.normal(0, 40, n)).clip(126, 564).astype(int)
    fbs = np.random.binomial(1, 0.15, n)
    restecg = np.random.choice(3, n, p=[0.50, 0.35, 0.15])
    thalach = (220 - age + np.random.normal(0, 20, n)).clip(71, 202).astype(int)
    exang = np.random.binomial(1, 0.33, n)
    oldpeak = np.abs(np.random.normal(1.0, 1.5, n)).clip(0, 6.2).round(1)
    slope = np.random.choice(3, n, p=[0.25, 0.50, 0.25])
    ca = np.random.choice(4, n, p=[0.55, 0.25, 0.15, 0.05])
    thal = np.random.choice(4, n, p=[0.15, 0.15, 0.65, 0.05])

    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    })

    risk = (
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
        (df['thal'] == 2).astype(float) * 0.20 +
        np.random.normal(0, 0.15, n)
    )
    df['target'] = (risk > np.percentile(risk, 55)).astype(int)

    os.makedirs(DATASET_DIR, exist_ok=True)
    df.to_csv(HEART_DATASET_PATH, index=False)
    print(f"✓ Dataset created: {HEART_DATASET_PATH}")
    print(f"  Samples: {len(df)}  Positive: {df['target'].sum()} "
          f"({df['target'].mean()*100:.1f}%)")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#                       DATA LOADING / PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_heart_data(df):
    """Clean and standardize heart disease dataset."""
    df = df.copy()

    # ── Map string values to numeric ────────────────────────────────────
    mappings = {
        'sex': {'Male': 1, 'male': 1, 'M': 1, 'm': 1,
                'Female': 0, 'female': 0, 'F': 0, 'f': 0},
        'cp':  {'typical angina': 0, 'atypical angina': 1,
                'non-anginal pain': 2, 'asymptomatic': 3,
                'ta': 0, 'ata': 1, 'nap': 2, 'asy': 3,
                'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
        'fbs': {True: 1, False: 0, 'True': 1, 'False': 0,
                'Yes': 1, 'No': 0, 'yes': 1, 'no': 0},
        'exang': {'Yes': 1, 'No': 0, 'Y': 1, 'N': 0,
                  'yes': 1, 'no': 0},
        'restecg': {'normal': 0, 'st-t abnormality': 1,
                    'lv hypertrophy': 2, 'Normal': 0, 'ST': 1, 'LVH': 2},
        'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2,
                  'Up': 0, 'Flat': 1, 'Down': 2},
        'thal': {'normal': 0, 'fixed defect': 1, 'reversible defect': 2,
                 'reversable defect': 2, 'fixed': 1, 'reversible': 2}
    }

    for col, mapping in mappings.items():
        if col in df.columns and df[col].dtype in ['object', 'bool']:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
            df[col] = df[col].map(mapping)
            converted = df[col].notna().sum()
            print(f"  ✓ Converted '{col}' ({converted} values)")

    # ── Column name aliases ─────────────────────────────────────────────
    aliases = {
        'thalch': 'thalach', 'num': 'target', 'condition': 'target',
        'disease': 'target', 'heart_disease': 'target',
        'HeartDisease': 'target', 'output': 'target'
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
            print(f"  ✓ Mapped '{old}' → '{new}'")

    # ── Binary target ───────────────────────────────────────────────────
    if 'target' in df.columns and df['target'].max() > 1:
        df['target'] = (df['target'] > 0).astype(int)
        print("  ✓ Multi-class target → binary")

    # ── Fill missing values ─────────────────────────────────────────────
    for col in df.columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                mode = df[col].mode()
                df[col].fillna(mode[0] if len(mode) > 0 else 0, inplace=True)
            print(f"  ✓ Filled {nulls} missing in '{col}'")

    return df


def load_heart_data():
    """Load heart disease dataset (real CSV or generated synthetic)."""
    if not os.path.exists(HEART_DATASET_PATH):
        print(f"⚠  Dataset not found: {HEART_DATASET_PATH}")
        return create_sample_heart_dataset()

    print(f"📂 Loading: {HEART_DATASET_PATH}")
    df = pd.read_csv(HEART_DATASET_PATH)
    print(f"  {len(df)} records, {len(df.columns)} columns")

    df = preprocess_heart_data(df)
    initial = len(df)
    df = df.dropna()
    if len(df) < initial:
        print(f"  ✓ Dropped {initial - len(df)} incomplete rows")

    if 'target' not in df.columns:
        print("❌ No target column found!")
        return create_sample_heart_dataset()

    # Verify all 13 base features exist
    missing_cols = [f for f in BASE_FEATURES if f not in df.columns]
    if missing_cols:
        print(f"⚠  Missing columns: {missing_cols}")
        print("  Generating synthetic dataset instead...")
        return create_sample_heart_dataset()

    print(f"  ✓ Final: {len(df)} records")
    print(f"  Target: {dict(df['target'].value_counts())}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#                           TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_heart_model():
    """
    Train heart disease risk prediction model.

    Method: Soft-voting ensemble of RF + GB + LR.
    Features: 13 raw clinical features (NO engineering).
    Scaler: StandardScaler fitted on the same 13 features.

    This guarantees server.py can send its 13-feature vector
    directly to scaler.transform() → model.predict() with
    zero chance of a dimension mismatch.
    """
    print("\n" + "=" * 70)
    print("  HEART DISEASE PREDICTION — Ensemble Model")
    print("  Features: 13 raw clinical (no engineering)")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────
    df = load_heart_data()
    if len(df) < 50:
        print("❌ Insufficient data!"); return None, None

    # Use ONLY the 13 base features — same as server.py sends
    available = [f for f in BASE_FEATURES if f in df.columns]
    if len(available) < len(BASE_FEATURES):
        print(f"⚠  Only {len(available)}/{len(BASE_FEATURES)} features available")

    # Drop duplicate records to prevent identical rows leaking into train/test splits
    df_unique = df[available + ['target']].drop_duplicates()
    X = df_unique[available].values.astype(np.float64)
    y = df_unique['target'].values.astype(int)

    print(f"\n  Deduplicated dataset: {len(df_unique)} unique records (down from {len(df)})")
    print(f"  Features: {len(available)}  →  {available}")
    print(f"  Samples:  {len(X)}")
    print(f"  Class 0:  {(y == 0).sum()}   Class 1: {(y == 1).sum()}")

    # ── Split ───────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Scale ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"  Scaler fitted on {X_train_s.shape[1]} features ✓")

    # ── Tuning & training individual models ──────────────────────────────
    print("\n" + "-" * 50)
    print("  Tuning and training individual models...")
    print("-" * 50)

    # Random Forest Hyperparameter Search
    rf_param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=15, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    rf_search.fit(X_train_s, y_train)
    rf = rf_search.best_estimator_
    print(f"  ✓ Random Forest (Best params: {rf_search.best_params_})")

    # Gradient Boosting Hyperparameter Search
    gb_param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    }
    gb_base = GradientBoostingClassifier(random_state=42)
    gb_search = RandomizedSearchCV(gb_base, gb_param_dist, n_iter=15, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    gb_search.fit(X_train_s, y_train)
    gb = gb_search.best_estimator_
    print(f"  ✓ Gradient Boosting (Best params: {gb_search.best_params_})")

    # Logistic Regression Hyperparameter Search
    lr_param_dist = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    lr_base = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr_search = RandomizedSearchCV(lr_base, lr_param_dist, n_iter=10, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    lr_search.fit(X_train_s, y_train)
    lr = lr_search.best_estimator_
    print(f"  ✓ Logistic Regression (Best params: {lr_search.best_params_})")

    # ── Ensemble & Calibration ──────────────────────────────────────────
    print("\n  Creating ensemble & calibrating probabilities...")
    ensemble_base = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft', weights=[2, 2, 1]
    )
    ensemble_base.fit(X_train_s, y_train)
    
    # Sigmoid calibration with 5-fold CV to prevent extreme probabilities
    ensemble = CalibratedClassifierCV(
        estimator=ensemble_base, method='sigmoid', cv=5
    )
    ensemble.fit(X_train_s, y_train)
    print("  ✓ Calibrated Ensemble (sigmoid, 5-fold CV)")

    # ── Evaluate all models ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)

    all_models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Logistic Regression': lr,
        'Ensemble': ensemble_base,
        'Calibrated Ensemble': ensemble
    }

    best_model = None
    best_auc = 0
    best_name = ''
    metrics_summary = {}

    for name, mdl in all_models.items():
        yp = mdl.predict(X_test_s)
        yprob = mdl.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, yp)
        auc = roc_auc_score(y_test, yprob)
        f1 = f1_score(y_test, yp)
        mcc = matthews_corrcoef(y_test, yp)

        print(f"\n  {name}:")
        print(f"    Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
        print(f"    ROC-AUC:  {auc:.4f}")
        print(f"    F1:       {f1:.4f}")
        print(f"    MCC:      {mcc:.4f}")

        metrics_summary[name] = {
            'accuracy': float(acc), 'roc_auc': float(auc),
            'f1_score': float(f1), 'mcc': float(mcc)
        }

        if auc > best_auc:
            best_auc = auc
            best_model = mdl
            best_name = name

    print(f"\n{'='*70}")
    print(f"  BEST: {best_name}  (ROC-AUC = {best_auc:.4f})")
    print(f"{'='*70}")

    # ── Cross-validation ────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = ensemble_base if 'Calibrated' in best_name else best_model
    cv = cross_val_score(cv_model, X_train_s, y_train,
                         cv=skf, scoring='roc_auc', n_jobs=-1)
    print(f"\n  5-Fold CV (ROC-AUC):")
    print(f"    Mean: {cv.mean():.4f}  Std: {cv.std():.4f}")
    print(f"    Folds: {[f'{s:.4f}' for s in cv]}")

    # ── Confusion matrix ────────────────────────────────────────────────
    yp_final = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, yp_final)
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"              No-Dis  Disease")
    print(f"  No-Disease  {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  Disease     {cm[1,0]:5d}    {cm[1,1]:5d}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, yp_final,
                                target_names=['No Disease', 'Disease'],
                                zero_division=0))

    # ── Feature importance ──────────────────────────────────────────────
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif best_name in ['Ensemble', 'Calibrated Ensemble']:
        # Average importance from tree-based sub-models
        importances = (rf.feature_importances_ * 2 +
                       gb.feature_importances_ * 2) / 4
    else:
        importances = None

    if importances is not None:
        print(f"\n  Feature Importance (top 13):")
        fi = sorted(zip(available, importances),
                     key=lambda x: x[1], reverse=True)
        for name_f, imp in fi:
            bar = '█' * int(imp * 100)
            print(f"    {name_f:12s} {bar} {imp:.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SAVING ARTIFACTS")
    print(f"{'='*70}")

    joblib.dump(best_model, HEART_MODEL_PATH)
    print(f"  ✓ Model:    {HEART_MODEL_PATH}")

    joblib.dump(scaler, HEART_SCALER_PATH)
    print(f"  ✓ Scaler:   {HEART_SCALER_PATH}")

    feature_info = {
        'features': available,
        'num_features': len(available),
        'feature_details': {k: v for k, v in HEART_FEATURES.items()
                            if k in available},
        'model_type': best_name,
        'note': 'NO feature engineering — 13 raw features only',
        'training_date': pd.Timestamp.now().isoformat()
    }
    with open(HEART_FEATURES_PATH, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"  ✓ Features: {HEART_FEATURES_PATH}")

    metrics_summary['best_model'] = best_name
    metrics_summary['cv_mean_auc'] = float(cv.mean())
    with open(HEART_METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  ✓ Metrics:  {HEART_METRICS_PATH}")

    # ── Test prediction ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TEST PREDICTION")
    print(f"{'='*70}")

    sample = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }
    sample_arr = np.array([[sample[f] for f in available]])
    sample_s = scaler.transform(sample_arr)
    prob = best_model.predict_proba(sample_s)[0]

    print(f"  Input:  {sample}")
    print(f"  P(no disease): {prob[0]:.4f}")
    print(f"  P(disease):    {prob[1]:.4f}")
    print(f"  Prediction:    {'Disease' if prob[1] > 0.5 else 'No Disease'}")

    # Verify feature count matches
    print(f"\n  ✅ Scaler expects {scaler.n_features_in_} features")
    print(f"  ✅ Model trained on {len(available)} features")
    print(f"  ✅ server.py sends {len(BASE_FEATURES)} features")
    assert scaler.n_features_in_ == len(available) == len(BASE_FEATURES), \
        "Feature count mismatch!"
    print(f"  ✅ All counts match — no dimension mismatch possible!")

    print(f"\n{'='*70}")
    print("  ✅ TRAINING COMPLETE!")
    print(f"  ⚠️  Restart server.py to load the new model")
    print(f"{'='*70}\n")

    return best_model, scaler


# ═════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        model, scaler = train_heart_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()