import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
CANCER_DATASET_PATH = os.path.join(DATASET_DIR, 'cancer.csv')

# Output paths
CANCER_MODEL_PATH = os.path.join(SCRIPT_DIR, 'cancer_model.joblib')
CANCER_SCALER_PATH = os.path.join(SCRIPT_DIR, 'cancer_scaler.joblib')
CANCER_FEATURES_PATH = os.path.join(SCRIPT_DIR, 'cancer_features.json')

# Feature descriptions
CANCER_FEATURES = {
    'radius_mean': {'description': 'Mean of distances from center to points on perimeter', 'type': 'numeric'},
    'texture_mean': {'description': 'Standard deviation of gray-scale values', 'type': 'numeric'},
    'perimeter_mean': {'description': 'Mean size of core tumor', 'type': 'numeric'},
    'area_mean': {'description': 'Mean area of core tumor', 'type': 'numeric'},
    'smoothness_mean': {'description': 'Mean of local variation in radius lengths', 'type': 'numeric'},
    'compactness_mean': {'description': 'Mean of perimeter^2 / area - 1.0', 'type': 'numeric'},
    'concavity_mean': {'description': 'Mean of severity of concave portions of contour', 'type': 'numeric'},
    'concave_points_mean': {'description': 'Mean number of concave portions of contour', 'type': 'numeric'},
    'symmetry_mean': {'description': 'Mean symmetry of cell', 'type': 'numeric'},
    'fractal_dimension_mean': {'description': 'Mean coastline approximation - 1', 'type': 'numeric'},
}


def create_enhanced_cancer_dataset():
    """Create enhanced sample dataset with realistic distributions"""
    print("Creating enhanced cancer dataset...")
    
    np.random.seed(42)
    n_samples = 569  # Match real dataset size
    
    # 357 benign (62.7%), 212 malignant (37.3%) - real distribution
    n_benign = 357
    n_malignant = 212
    
    # Benign tumors - based on real statistics
    benign_data = {
        'radius_mean': np.random.normal(12.15, 1.78, n_benign),
        'texture_mean': np.random.normal(17.91, 4.28, n_benign),
        'perimeter_mean': np.random.normal(78.08, 11.68, n_benign),
        'area_mean': np.random.normal(462.79, 134.13, n_benign),
        'smoothness_mean': np.random.normal(0.0925, 0.014, n_benign),
        'compactness_mean': np.random.normal(0.0801, 0.052, n_benign),
        'concavity_mean': np.random.normal(0.0461, 0.051, n_benign),
        'concave_points_mean': np.random.normal(0.0257, 0.023, n_benign),
        'symmetry_mean': np.random.normal(0.174, 0.027, n_benign),
        'fractal_dimension_mean': np.random.normal(0.0628, 0.007, n_benign),
        'diagnosis': np.zeros(n_benign, dtype=int)
    }
    
    # Malignant tumors - based on real statistics
    malignant_data = {
        'radius_mean': np.random.normal(17.46, 3.20, n_malignant),
        'texture_mean': np.random.normal(21.60, 3.78, n_malignant),
        'perimeter_mean': np.random.normal(115.36, 21.75, n_malignant),
        'area_mean': np.random.normal(978.38, 368.13, n_malignant),
        'smoothness_mean': np.random.normal(0.1028, 0.018, n_malignant),
        'compactness_mean': np.random.normal(0.1458, 0.080, n_malignant),
        'concavity_mean': np.random.normal(0.1606, 0.097, n_malignant),
        'concave_points_mean': np.random.normal(0.0880, 0.039, n_malignant),
        'symmetry_mean': np.random.normal(0.193, 0.034, n_malignant),
        'fractal_dimension_mean': np.random.normal(0.0627, 0.008, n_malignant),
        'diagnosis': np.ones(n_malignant, dtype=int)
    }
    
    # Combine and shuffle
    df_benign = pd.DataFrame(benign_data)
    df_malignant = pd.DataFrame(malignant_data)
    df = pd.concat([df_benign, df_malignant], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clip values to realistic ranges
    df['radius_mean'] = df['radius_mean'].clip(6.981, 28.11)
    df['texture_mean'] = df['texture_mean'].clip(9.71, 39.28)
    df['perimeter_mean'] = df['perimeter_mean'].clip(43.79, 188.5)
    df['area_mean'] = df['area_mean'].clip(143.5, 2501)
    df['smoothness_mean'] = df['smoothness_mean'].clip(0.05263, 0.1634)
    df['compactness_mean'] = df['compactness_mean'].clip(0.01938, 0.3454)
    df['concavity_mean'] = df['concavity_mean'].clip(0, 0.4268)
    df['concave_points_mean'] = df['concave_points_mean'].clip(0, 0.2012)
    df['symmetry_mean'] = df['symmetry_mean'].clip(0.106, 0.304)
    df['fractal_dimension_mean'] = df['fractal_dimension_mean'].clip(0.04996, 0.09744)
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    df.to_csv(CANCER_DATASET_PATH, index=False)
    print(f"✓ Enhanced dataset saved: {CANCER_DATASET_PATH}")
    
    return df


def preprocess_cancer_data(df):
    """Enhanced preprocessing with better error handling"""
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Handle diagnosis column
    diagnosis_cols = ['diagnosis', 'target', 'class', 'label']
    diagnosis_col = next((col for col in diagnosis_cols if col in df.columns), None)
    
    if diagnosis_col and diagnosis_col != 'diagnosis':
        df['diagnosis'] = df[diagnosis_col]
    
    if 'diagnosis' in df.columns:
        # Handle M/B or string values
        if df['diagnosis'].dtype == 'object':
            df['diagnosis'] = df['diagnosis'].map({
                'M': 1, 'Malignant': 1, 'malignant': 1, 'm': 1, 1: 1, '1': 1,
                'B': 0, 'Benign': 0, 'benign': 0, 'b': 0, 0: 0, '0': 0
            })
        df['diagnosis'] = pd.to_numeric(df['diagnosis'], errors='coerce')
    
    # Handle concave points column name variations
    if 'concave_points_mean' not in df.columns:
        for col in df.columns:
            if 'concave' in col and 'point' in col and 'mean' in col:
                df['concave_points_mean'] = df[col]
                break
    
    # Remove ID columns and unnamed columns
    cols_to_drop = [col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Handle missing values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Remove outliers using IQR method
    for col in numeric_cols:
        if col != 'diagnosis':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df


def load_cancer_data():
    """Load and validate cancer dataset"""
    if not os.path.exists(CANCER_DATASET_PATH):
        print(f"Dataset not found at {CANCER_DATASET_PATH}")
        return create_enhanced_cancer_dataset()
    
    try:
        df = pd.read_csv(CANCER_DATASET_PATH)
        print(f"✓ Loaded dataset: {len(df)} records")
        
        if len(df) == 0:
            return create_enhanced_cancer_dataset()
        
        df = preprocess_cancer_data(df)
        
        if 'diagnosis' not in df.columns or len(df) == 0:
            print("Invalid dataset structure, creating new one...")
            return create_enhanced_cancer_dataset()
        
        print(f"✓ Final dataset: {len(df)} records")
        print(f"  Benign: {(df['diagnosis'] == 0).sum()} ({(df['diagnosis'] == 0).sum()/len(df)*100:.1f}%)")
        print(f"  Malignant: {(df['diagnosis'] == 1).sum()} ({(df['diagnosis'] == 1).sum()/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return create_enhanced_cancer_dataset()


def train_cancer_model():
    """Train improved cancer prediction model"""
    print("\n" + "="*70)
    print("TRAINING ENHANCED CANCER PREDICTION MODEL")
    print("="*70)
    
    df = load_cancer_data()
    
    if len(df) < 50:
        print("ERROR: Insufficient data for training!")
        return None, None
    
    # Define feature columns
    feature_cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                   'smoothness_mean', 'compactness_mean', 'concavity_mean',
                   'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean']
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 5:
        print("ERROR: Insufficient features found!")
        df = create_enhanced_cancer_dataset()
        available_features = feature_cols
    
    print(f"\n✓ Using {len(available_features)} features")
    
    X = df[available_features].values
    y = df['diagnosis'].values
    
    print(f"✓ Feature matrix: {X.shape}")
    print(f"✓ Target distribution: {np.bincount(y)}")
    
    # Use RobustScaler instead of StandardScaler (better for outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE only if imbalanced
    minority_class_size = min(np.bincount(y_train))
    if minority_class_size > 6:  # SMOTE needs at least 6 samples
        smote = SMOTE(random_state=42, k_neighbors=min(5, minority_class_size-1))
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"✓ Applied SMOTE - New training size: {len(X_train)}")
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    
    # Build advanced ensemble
    print("\n" + "-"*70)
    print("Building Advanced Ensemble Model")
    print("-"*70)
    
    # Random Forest with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Gradient Boosting with optimized parameters
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    # Logistic Regression with regularization
    lr = LogisticRegression(
        C=0.1,
        penalty='l2',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    # SVM with RBF kernel
    svm = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    
    # Voting ensemble with optimized weights
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr),
            ('svm', svm)
        ],
        voting='soft',
        weights=[3, 3, 1, 2]  # RF and GB get higher weights
    )
    
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='roc_auc')
    print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Detailed metrics
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT")
    print("-"*70)
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'], digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    print(f"True Negatives:  {cm[0,0]:4d} | False Positives: {cm[0,1]:4d}")
    print(f"False Negatives: {cm[1,0]:4d} | True Positives:  {cm[1,1]:4d}")
    
    # Feature importance from Random Forest
    rf_model = ensemble.named_estimators_['rf']
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "-"*70)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("-"*70)
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Save artifacts
    print("\n" + "="*70)
    print("SAVING MODEL ARTIFACTS")
    print("="*70)
    
    joblib.dump(ensemble, CANCER_MODEL_PATH)
    print(f"✓ Model saved: {CANCER_MODEL_PATH}")
    
    joblib.dump(scaler, CANCER_SCALER_PATH)
    print(f"✓ Scaler saved: {CANCER_SCALER_PATH}")
    
    feature_info = {
        'features': available_features,
        'feature_details': CANCER_FEATURES,
        'classes': ['Benign', 'Malignant'],
        'performance': {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'f1_score': float(f1)
        },
        'feature_importance': feature_importance.to_dict('records')
    }
    
    with open(CANCER_FEATURES_PATH, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✓ Feature info saved: {CANCER_FEATURES_PATH}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    
    return ensemble, scaler


if __name__ == '__main__':
    train_cancer_model()