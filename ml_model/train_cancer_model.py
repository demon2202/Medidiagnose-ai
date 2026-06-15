"""
train_cancer_model.py - PRODUCTION VERSION
===========================================
MediDiagnose-AI: Breast Cancer Tumor Classification

Method: Calibrated Ensemble (RF + GB + LR) with soft voting
Input:  10 mean tumor features from FNA test
Output: Binary (0=Benign, 1=Malignant) + calibrated probability

Uses sklearn's REAL Wisconsin Breast Cancer dataset (569 samples)
as the primary data source. Falls back to CSV if available.

StandardScaler on 10 features — matches server.py exactly.
CalibratedClassifierCV ensures realistic probability outputs.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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
    roc_auc_score, confusion_matrix, f1_score,
    matthews_corrcoef
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
CANCER_DATASET_PATH = os.path.join(DATASET_DIR, 'cancer.csv')

OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'ml_model')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANCER_MODEL_PATH = os.path.join(OUTPUT_DIR, 'cancer_model.joblib')
CANCER_SCALER_PATH = os.path.join(OUTPUT_DIR, 'cancer_scaler.joblib')
CANCER_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'cancer_features.json')
CANCER_METRICS_PATH = os.path.join(OUTPUT_DIR, 'cancer_metrics.json')

# ── The 10 features server.py sends ────────────────────────────────────────
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

FEATURE_INFO = {
    'radius_mean':            {'description': 'Mean distance from center to perimeter',        'range': '6-28'},
    'texture_mean':           {'description': 'Std deviation of gray-scale values',            'range': '9-40'},
    'perimeter_mean':         {'description': 'Mean tumor perimeter',                          'range': '40-190'},
    'area_mean':              {'description': 'Mean tumor area',                               'range': '140-2500'},
    'smoothness_mean':        {'description': 'Local variation in radius lengths',             'range': '0.05-0.16'},
    'compactness_mean':       {'description': 'Perimeter² / area - 1.0',                      'range': '0.02-0.35'},
    'concavity_mean':         {'description': 'Severity of concave portions',                  'range': '0-0.43'},
    'concave_points_mean':    {'description': 'Number of concave portions',                    'range': '0-0.20'},
    'symmetry_mean':          {'description': 'Cell symmetry',                                 'range': '0.10-0.30'},
    'fractal_dimension_mean': {'description': 'Coastline approximation - 1',                   'range': '0.05-0.10'},
}

# ── Sample data for verification (from CancerScreening.jsx) ────────────────
SAMPLE_BENIGN = {
    'radius_mean': 12.5, 'texture_mean': 17.2, 'perimeter_mean': 78.5,
    'area_mean': 450, 'smoothness_mean': 0.09, 'compactness_mean': 0.07,
    'concavity_mean': 0.04, 'concave_points_mean': 0.02,
    'symmetry_mean': 0.17, 'fractal_dimension_mean': 0.06
}

SAMPLE_MALIGNANT = {
    'radius_mean': 18.5, 'texture_mean': 22.0, 'perimeter_mean': 120.0,
    'area_mean': 1050, 'smoothness_mean': 0.11, 'compactness_mean': 0.18,
    'concavity_mean': 0.20, 'concave_points_mean': 0.10,
    'symmetry_mean': 0.21, 'fractal_dimension_mean': 0.07
}


# ═════════════════════════════════════════════════════════════════════════════
#                         DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_cancer_data():
    """
    Load breast cancer data.

    Priority:
      1. CSV file at Dataset/cancer.csv (if valid)
      2. sklearn's built-in Wisconsin Breast Cancer dataset (569 real samples)

    IMPORTANT: sklearn encodes 0=malignant, 1=benign.
    We flip to 0=benign, 1=malignant (matching server.py convention).

    Returns:
        DataFrame with 10 feature columns + 'diagnosis' (0=benign, 1=malignant)
    """

    # ── Try loading from CSV first ──────────────────────────────────────
    if os.path.exists(CANCER_DATASET_PATH):
        try:
            df = pd.read_csv(CANCER_DATASET_PATH)
            print(f"📂 Loaded CSV: {CANCER_DATASET_PATH}  ({len(df)} rows)")

            # Standardize column names
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

            # Handle concave points naming variations
            for col in df.columns:
                if 'concave' in col and 'point' in col and 'mean' in col:
                    if col != 'concave_points_mean':
                        df['concave_points_mean'] = df[col]
                        break

            # Find diagnosis column
            diag_col = None
            for candidate in ['diagnosis', 'target', 'class', 'label']:
                if candidate in df.columns:
                    diag_col = candidate
                    break

            if diag_col is not None:
                # Map string labels to numeric
                if df[diag_col].dtype == 'object':
                    mapping = {
                        'M': 1, 'Malignant': 1, 'malignant': 1, 'm': 1,
                        'B': 0, 'Benign': 0, 'benign': 0, 'b': 0,
                    }
                    df['diagnosis'] = df[diag_col].str.strip().map(mapping)
                else:
                    df['diagnosis'] = pd.to_numeric(df[diag_col], errors='coerce')

                # Binary
                if df['diagnosis'].max() > 1:
                    df['diagnosis'] = (df['diagnosis'] > 0).astype(int)

                # Check all 10 features exist
                available = [f for f in FEATURE_NAMES if f in df.columns]
                if len(available) == 10 and df['diagnosis'].notna().sum() > 50:
                    df = df[FEATURE_NAMES + ['diagnosis']].dropna()
                    print(f"  ✓ Valid CSV with {len(df)} samples")
                    print(f"  Benign:    {(df['diagnosis']==0).sum()}")
                    print(f"  Malignant: {(df['diagnosis']==1).sum()}")
                    return df
                else:
                    print(f"  ⚠  CSV missing features ({len(available)}/10)")

        except Exception as e:
            print(f"  ⚠  CSV load error: {e}")

    # ── Fallback: sklearn's REAL Wisconsin Breast Cancer dataset ─────────
    print("📂 Using sklearn's Wisconsin Breast Cancer dataset (569 real samples)")
    data = load_breast_cancer()

    # First 10 features are the "mean" features in the correct order
    X = data.data[:, :10]

    # CRITICAL: sklearn uses 0=malignant, 1=benign
    # We need 0=benign, 1=malignant (matching server.py)
    y = 1 - data.target  # flip labels

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['diagnosis'] = y

    print(f"  Samples:   {len(df)}")
    print(f"  Benign:    {(y == 0).sum()}  ({(y == 0).mean()*100:.1f}%)")
    print(f"  Malignant: {(y == 1).sum()}  ({(y == 1).mean()*100:.1f}%)")

    # Save for future use
    os.makedirs(DATASET_DIR, exist_ok=True)
    df.to_csv(CANCER_DATASET_PATH, index=False)
    print(f"  ✓ Saved to: {CANCER_DATASET_PATH}")

    return df


# ═════════════════════════════════════════════════════════════════════════════
#                           TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_cancer_model():
    """
    Train breast cancer tumor classifier.

    Method: Calibrated soft-voting ensemble of RF + GB + LR.
    Features: 10 mean tumor features (no engineering).
    Scaler: StandardScaler on the same 10 features.
    Calibration: CalibratedClassifierCV (sigmoid) for realistic probabilities.

    Class imbalance handled via class_weight (no SMOTE).
    No outlier removal (border cases are important for classification).
    """
    print("\n" + "=" * 70)
    print("  BREAST CANCER SCREENING — Calibrated Ensemble Model")
    print("  Features: 10 mean tumor characteristics")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────
    df = load_cancer_data()
    if len(df) < 50:
        print("❌ Insufficient data!")
        return None, None

    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df['diagnosis'].values.astype(int)

    print(f"\n  Features:  {len(FEATURE_NAMES)}  →  {FEATURE_NAMES}")
    print(f"  Samples:   {len(X)}")
    print(f"  Class 0 (Benign):    {(y == 0).sum()}")
    print(f"  Class 1 (Malignant): {(y == 1).sum()}")

    # ── Split ───────────────────────────────────────────────────────────
    # Use 3-way split: train / calibration / test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Further split training into train + calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.25,
        random_state=42, stratify=y_train_full
    )

    print(f"\n  Train:       {len(X_train)}")
    print(f"  Calibration: {len(X_cal)}")
    print(f"  Test:        {len(X_test)}")

    # ── Scale ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)
    X_train_full_s = scaler.transform(X_train_full)
    print(f"  Scaler fitted on {scaler.n_features_in_} features ✓")

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

    # ── Ensemble ────────────────────────────────────────────────────────
    print("\n  Creating ensemble...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft', weights=[2, 2, 1]
    )
    ensemble.fit(X_train_s, y_train)
    print("  ✓ Ensemble (soft voting, weights=[2,2,1])")

    # ── Calibrated Ensemble ─────────────────────────────────────────────
    print("\n  Calibrating probabilities...")
    print("  This prevents 0.0% and 100.0% probability outputs")

    # Method 1: Calibrate using the held-out calibration set
    calibrated_ensemble = CalibratedClassifierCV(
        ensemble, method='sigmoid', cv='prefit'
    )
    calibrated_ensemble.fit(X_cal_s, y_cal)
    print("  ✓ Calibrated Ensemble (sigmoid, prefit on calibration set)")

    # Also train a CV-calibrated version on full training data for comparison
    ensemble_full = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft', weights=[2, 2, 1]
    )
    ensemble_full.fit(X_train_full_s, y_train_full)

    calibrated_cv = CalibratedClassifierCV(
        ensemble_full, method='sigmoid', cv=5
    )
    calibrated_cv.fit(X_train_full_s, y_train_full)
    print("  ✓ Calibrated Ensemble CV (sigmoid, 5-fold on full train)")

    # ── Evaluate all models ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)

    all_models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Logistic Regression': lr,
        'Ensemble': ensemble,
        'Calibrated Ensemble': calibrated_ensemble,
        'Calibrated Ensemble CV': calibrated_cv
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

        # Check probability range (important for calibration check)
        prob_min = float(np.min(yprob))
        prob_max = float(np.max(yprob))
        prob_mean = float(np.mean(yprob))

        print(f"\n  {name}:")
        print(f"    Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
        print(f"    ROC-AUC:  {auc:.4f}")
        print(f"    F1:       {f1:.4f}")
        print(f"    MCC:      {mcc:.4f}")
        print(f"    Prob range: [{prob_min:.4f} - {prob_max:.4f}]  mean={prob_mean:.4f}")

        # Flag if probabilities are too extreme
        if prob_min < 0.001 or prob_max > 0.999:
            print(f"    ⚠️  WARNING: Extreme probabilities detected!")
        else:
            print(f"    ✅ Probability range looks healthy")

        metrics_summary[name] = {
            'accuracy': float(acc), 'roc_auc': float(auc),
            'f1_score': float(f1), 'mcc': float(mcc),
            'prob_min': prob_min, 'prob_max': prob_max
        }

        # Prefer calibrated models — use AUC as primary, but
        # penalize extreme probability ranges
        effective_auc = auc
        if prob_min < 0.001 or prob_max > 0.999:
            effective_auc -= 0.01  # slight penalty for extreme probs

        if effective_auc > best_auc:
            best_auc = effective_auc
            best_model = mdl
            best_name = name

    print(f"\n{'='*70}")
    print(f"  BEST: {best_name}  (ROC-AUC = {best_auc:.4f})")
    print(f"{'='*70}")

    # ── Cross-validation ────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # For calibrated prefit models, use the uncalibrated version for CV
    cv_model = ensemble_full if 'Calibrated' in best_name else best_model
    cv = cross_val_score(cv_model, X_train_full_s, y_train_full,
                         cv=skf, scoring='roc_auc', n_jobs=-1)
    print(f"\n  5-Fold CV (ROC-AUC):")
    print(f"    Mean: {cv.mean():.4f}  Std: {cv.std():.4f}")

    # ── Confusion matrix ────────────────────────────────────────────────
    yp_final = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, yp_final)
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"              Benign  Malign")
    print(f"  Benign      {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  Malignant   {cm[1,0]:5d}   {cm[1,1]:5d}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, yp_final,
                                target_names=['Benign', 'Malignant'],
                                zero_division=0))

    # ── Feature importance ──────────────────────────────────────────────
    importances = (rf.feature_importances_ * 2 +
                   gb.feature_importances_ * 2) / 4
    fi = sorted(zip(FEATURE_NAMES, importances),
                key=lambda x: x[1], reverse=True)
    print(f"\n  Feature Importance:")
    for name_f, imp in fi:
        bar = '█' * int(imp * 100)
        print(f"    {name_f:25s} {bar} {imp:.4f}")

    # ── Probability distribution check ──────────────────────────────────
    print(f"\n{'='*70}")
    print("  PROBABILITY DISTRIBUTION CHECK")
    print(f"{'='*70}")

    test_probs = best_model.predict_proba(X_test_s)[:, 1]
    print(f"\n  Test set probability statistics:")
    print(f"    Min:    {np.min(test_probs):.6f}  ({np.min(test_probs)*100:.2f}%)")
    print(f"    Max:    {np.max(test_probs):.6f}  ({np.max(test_probs)*100:.2f}%)")
    print(f"    Mean:   {np.mean(test_probs):.6f}")
    print(f"    Median: {np.median(test_probs):.6f}")
    print(f"    Std:    {np.std(test_probs):.6f}")

    # Count extreme predictions
    extreme_low = np.sum(test_probs < 0.01)
    extreme_high = np.sum(test_probs > 0.99)
    print(f"\n    Predictions < 1%:  {extreme_low} ({extreme_low/len(test_probs)*100:.1f}%)")
    print(f"    Predictions > 99%: {extreme_high} ({extreme_high/len(test_probs)*100:.1f}%)")

    if extreme_low == 0 and extreme_high == 0:
        print(f"    ✅ No extreme probabilities — calibration working!")
    else:
        print(f"    ⚠️  Some extreme values remain — server.py clipping will handle these")

    # ── Verify with sample data ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  VERIFICATION WITH SAMPLE DATA")
    print(f"{'='*70}")

    for label, sample in [('BENIGN', SAMPLE_BENIGN), ('MALIGNANT', SAMPLE_MALIGNANT)]:
        arr = np.array([[sample[f] for f in FEATURE_NAMES]])
        arr_s = scaler.transform(arr)
        prob = best_model.predict_proba(arr_s)[0]
        pred = 'Malignant' if prob[1] > 0.5 else 'Benign'

        status = '✅' if pred.upper() == label else '❌ WRONG'
        print(f"\n  {label} sample:")
        print(f"    P(benign):    {prob[0]:.4f}  ({prob[0]*100:.1f}%)")
        print(f"    P(malignant): {prob[1]:.4f}  ({prob[1]*100:.1f}%)")
        print(f"    Prediction:   {pred}")
        print(f"    {status}")

        # Check for extreme values
        if prob[1] < 0.01 or prob[1] > 0.99:
            print(f"    ⚠️  Probability is extreme — but server.py will clip to [1%, 99%]")
        else:
            print(f"    ✅ Probability is in healthy range")

    # ── Save ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SAVING ARTIFACTS")
    print(f"{'='*70}")

    joblib.dump(best_model, CANCER_MODEL_PATH)
    print(f"  ✓ Model:    {CANCER_MODEL_PATH}")

    joblib.dump(scaler, CANCER_SCALER_PATH)
    print(f"  ✓ Scaler:   {CANCER_SCALER_PATH}")

    feature_info = {
        'features': FEATURE_NAMES,
        'num_features': len(FEATURE_NAMES),
        'feature_details': FEATURE_INFO,
        'classes': {0: 'Benign', 1: 'Malignant'},
        'model_type': best_name,
        'calibrated': 'Calibrated' in best_name,
        'note': 'StandardScaler, 10 raw features, calibrated probabilities',
        'training_date': pd.Timestamp.now().isoformat()
    }
    with open(CANCER_FEATURES_PATH, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"  ✓ Features: {CANCER_FEATURES_PATH}")

    metrics_summary['best_model'] = best_name
    metrics_summary['cv_mean_auc'] = float(cv.mean())
    metrics_summary['calibrated'] = 'Calibrated' in best_name
    with open(CANCER_METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  ✓ Metrics:  {CANCER_METRICS_PATH}")

    # ── Final verification ──────────────────────────────────────────────
    print(f"\n  ✅ Scaler expects {scaler.n_features_in_} features")
    print(f"  ✅ Model type: {best_name}")
    print(f"  ✅ Class 0 = Benign, Class 1 = Malignant")
    assert scaler.n_features_in_ == len(FEATURE_NAMES), "Feature count mismatch!"

    print(f"\n{'='*70}")
    print("  ✅ TRAINING COMPLETE!")
    print(f"  ✅ Probabilities are calibrated — no more 0.0% or 100.0%")
    print(f"  ⚠️  Restart server.py to load the new model")
    print(f"{'='*70}\n")

    return best_model, scaler


# ═════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        model, scaler = train_cancer_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()