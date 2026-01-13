import os
import numpy as np
import json
import warnings
import ast
import glob
warnings.filterwarnings('ignore')

# Check for WFDB library (required for PTB-XL)
WFDB_AVAILABLE = False
try:
    import wfdb
    WFDB_AVAILABLE = True
    print("✓ wfdb library available")
except ImportError:
    print("✗ wfdb not installed. Run: pip install wfdb")

# Check for TensorFlow
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} available")
except ImportError:
    print("✗ TensorFlow not available")

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import joblib

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
PTBXL_DIR = os.path.join(DATASET_DIR, 'ptb-xl')

# Output paths
HEART_IMAGE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'heart_image_model.h5')
HEART_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'heart_image_config.json')
HEART_SCALER_PATH = os.path.join(SCRIPT_DIR, 'heart_signal_scaler.joblib')

# ============== IMPORTANT: Image size MUST match server.py ==============
IMG_SIZE = 224

# ==============================================================================
#                           CLASS DEFINITIONS
# ==============================================================================

# Heart condition classes - MUST MATCH server.py HEART_CONDITIONS!
HEART_CLASSES = {
    0: {'code': 'normal', 'name': 'Normal', 'severity': 'healthy'},
    1: {'code': 'mi', 'name': 'Myocardial Infarction', 'severity': 'critical'},
    2: {'code': 'arrhythmia', 'name': 'Arrhythmia', 'severity': 'moderate'},
    3: {'code': 'hf', 'name': 'Heart Failure Signs', 'severity': 'high'},
    4: {'code': 'hypertrophy', 'name': 'Ventricular Hypertrophy', 'severity': 'moderate'}
}

# PTB-XL SCP code to class mapping
# Based on PTB-XL diagnostic superclasses
SCP_TO_CLASS = {
    # Normal
    'NORM': 0,
    'SR': 0,  # Sinus rhythm (normal)
    
    # Myocardial Infarction (MI) - Class 1
    'IMI': 1,   # Inferior MI
    'AMI': 1,   # Anterior MI
    'LMI': 1,   # Lateral MI
    'PMI': 1,   # Posterior MI
    'ASMI': 1,  # Anteroseptal MI
    'ILMI': 1,  # Inferolateral MI
    'IPLMI': 1, # Inferoposterolateral MI
    'IPMI': 1,  # Inferoposterior MI
    'INJAL': 1, # Injury in anterolateral
    'INJIL': 1, # Injury in inferolateral
    'INJLA': 1, # Injury in lateral anterior
    'INJIN': 1, # Injury inferior
    'INJAS': 1, # Injury anteroseptal
    'MI': 1,    # Generic MI
    
    # Arrhythmia - Class 2
    'AFIB': 2,  # Atrial fibrillation
    'AFLT': 2,  # Atrial flutter
    'SVTAC': 2, # Supraventricular tachycardia
    'PSVT': 2,  # Paroxysmal SVT
    'STACH': 2, # Sinus tachycardia
    'SBRAD': 2, # Sinus bradycardia
    'SARRH': 2, # Sinus arrhythmia
    'BIGU': 2,  # Bigeminy
    'TRIGU': 2, # Trigeminy
    'PAC': 2,   # Premature atrial contraction
    'PVC': 2,   # Premature ventricular contraction
    'VPRE': 2,  # Ventricular preexcitation
    'WPW': 2,   # Wolff-Parkinson-White
    
    # Heart Failure / Conduction Disturbance - Class 3
    'LAFB': 3,  # Left anterior fascicular block
    'LPFB': 3,  # Left posterior fascicular block
    'IRBBB': 3, # Incomplete RBBB
    'CRBBB': 3, # Complete RBBB
    'CLBBB': 3, # Complete LBBB
    'ILBBB': 3, # Incomplete LBBB
    '1AVB': 3,  # First degree AV block
    '2AVB': 3,  # Second degree AV block
    '3AVB': 3,  # Third degree AV block
    'CD': 3,    # Conduction disturbance
    'IVCD': 3,  # Intraventricular conduction delay
    
    # Hypertrophy - Class 4
    'LVH': 4,   # Left ventricular hypertrophy
    'RVH': 4,   # Right ventricular hypertrophy
    'LAO': 4,   # Left atrial overload
    'LAE': 4,   # Left atrial enlargement
    'RAO': 4,   # Right atrial overload
    'RAE': 4,   # Right atrial enlargement
    'SEHYP': 4, # Septal hypertrophy
    'HYP': 4,   # Generic hypertrophy
    
    # ST/T Changes - Map to appropriate class based on context
    'STTC': 2,    # ST/T changes (can indicate various issues)
    'STD_': 1,    # ST depression (often MI)
    'STE_': 1,    # ST elevation (often MI)
    'NST_': 2,    # Non-specific ST changes
    'ISC_': 1,    # Ischemia
    'ISCA': 1,    # Ischemia anterior
    'ISCI': 1,    # Ischemia inferior
}

# ECG Lead names
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ==============================================================================
#                           MODEL ARCHITECTURE
# ==============================================================================

def create_heart_image_model(input_shape=(224, 224, 1), num_classes=5):
    """
    Create CNN model for ECG IMAGE classification.
    
    IMPORTANT: Uses GRAYSCALE input to match server.py!
    Server.py calls: preprocess_image(image, target_size=(224, 224), grayscale=True)
    
    Input: Grayscale ECG plot image (224, 224, 1)
    Output: 5 classes (Normal, MI, Arrhythmia, HF, Hypertrophy)
    """
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 5
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Output: 5 classes
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model


# ==============================================================================
#                           SIGNAL TO IMAGE CONVERSION
# ==============================================================================

def signal_to_grayscale_image(signal, img_size=(224, 224)):
    """
    Convert ECG signal to GRAYSCALE plot image.
    
    IMPORTANT: Creates grayscale image to match server.py preprocessing!
    
    Args:
        signal: ECG signal array (time_steps, num_leads) e.g., (1000, 12)
        img_size: Output image size (height, width)
    
    Returns:
        Grayscale image array (H, W, 1) normalized to [0, 1]
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        num_leads = min(signal.shape[1], 12)
        
        # Create figure with white background
        # Use a layout that shows ECG clearly
        fig, axes = plt.subplots(4, 3, figsize=(12, 8), dpi=30)
        axes = axes.flatten()
        
        for i in range(num_leads):
            ax = axes[i]
            
            # Plot ECG signal
            ax.plot(signal[:, i], 'k-', linewidth=0.8)  # Black line
            
            # Set limits
            ax.set_xlim(0, len(signal))
            
            # Remove axes for cleaner image
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add lead name
            ax.set_title(LEAD_NAMES[i] if i < len(LEAD_NAMES) else f'L{i+1}', 
                        fontsize=8, fontweight='bold')
            
            # Add grid (like real ECG paper)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('white')
        
        # Hide unused subplots
        for i in range(num_leads, 12):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='white', edgecolor='none', dpi=50)
        plt.close(fig)
        buf.seek(0)
        
        # Load and convert to grayscale
        img = Image.open(buf)
        img = img.resize(img_size, Image.LANCZOS)
        img = img.convert('L')  # GRAYSCALE - matches server.py!
        
        # Normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add channel dimension: (H, W) -> (H, W, 1)
        img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
        
    except Exception as e:
        print(f"Error converting signal to image: {e}")
        # Return blank grayscale image
        return np.zeros((img_size[0], img_size[1], 1), dtype=np.float32)


def signal_to_single_lead_image(signal, lead_idx=1, img_size=(224, 224)):
    """
    Convert single ECG lead to grayscale image.
    Simpler visualization, often used for quick analysis.
    
    Args:
        signal: ECG signal array (time_steps, num_leads)
        lead_idx: Which lead to use (default 1 = Lead II, commonly used)
        img_size: Output image size
    
    Returns:
        Grayscale image array (H, W, 1) normalized to [0, 1]
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        # Get single lead
        lead_signal = signal[:, lead_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4), dpi=50)
        
        # Plot
        ax.plot(lead_signal, 'k-', linewidth=1)
        ax.set_xlim(0, len(lead_signal))
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        # Load and convert
        img = Image.open(buf)
        img = img.resize(img_size, Image.LANCZOS)
        img = img.convert('L')
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
        
    except Exception as e:
        print(f"Error: {e}")
        return np.zeros((img_size[0], img_size[1], 1), dtype=np.float32)


# ==============================================================================
#                           DATASET LOADING
# ==============================================================================

def find_ptbxl_dataset():
    """
    Find PTB-XL dataset in various possible locations.
    
    Returns:
        tuple: (dataset_path, metadata_path) or (None, None)
    """
    possible_names = [
        'ptb-xl',
        'ptbxl',
        'PTB-XL',
        'PTBXL',
        'ptb_xl',
        'physionet.org/files/ptb-xl/1.0.3'
    ]
    
    for name in possible_names:
        check_dir = os.path.join(DATASET_DIR, name)
        if os.path.exists(check_dir):
            # Look for metadata file
            metadata_paths = [
                os.path.join(check_dir, 'ptbxl_database.csv'),
                os.path.join(check_dir, 'ptbxl_database.csv'),
            ]
            
            for meta_path in metadata_paths:
                if os.path.exists(meta_path):
                    print(f"  Found PTB-XL dataset at: {check_dir}")
                    return check_dir, meta_path
            
            # Check if metadata is in subdirectory
            for subdir in os.listdir(check_dir):
                subpath = os.path.join(check_dir, subdir)
                if os.path.isdir(subpath):
                    meta_path = os.path.join(subpath, 'ptbxl_database.csv')
                    if os.path.exists(meta_path):
                        print(f"  Found PTB-XL dataset at: {subpath}")
                        return subpath, meta_path
    
    return None, None


def load_ptbxl_dataset(sampling_rate=100, max_samples=3000, img_size=224):
    """
    Load PTB-XL dataset and convert ECG signals to grayscale images.
    
    Args:
        sampling_rate: 100 or 500 Hz
        max_samples: Maximum number of samples to load
        img_size: Output image size
    
    Returns:
        X_images, y_labels, class_weights or None if dataset not found
    """
    if not WFDB_AVAILABLE:
        print("❌ wfdb library required. Run: pip install wfdb")
        return None
    
    # Find dataset
    ptbxl_dir, metadata_path = find_ptbxl_dataset()
    
    if ptbxl_dir is None:
        print(f"❌ PTB-XL dataset not found")
        print("\n📥 Download from: https://physionet.org/content/ptb-xl/1.0.3/")
        print("   Or: https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset")
        print(f"\n📁 Extract to: {PTBXL_DIR}")
        print("\nExpected structure:")
        print("  ptb-xl/")
        print("  ├── records100/")
        print("  │   ├── 00000/")
        print("  │   │   ├── 00001_lr.dat")
        print("  │   │   ├── 00001_lr.hea")
        print("  │   │   └── ...")
        print("  │   └── ...")
        print("  ├── records500/")
        print("  ├── ptbxl_database.csv")
        print("  └── scp_statements.csv")
        return None
    
    print(f"📂 Loading PTB-XL dataset...")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Image size: {img_size}x{img_size}")
    
    # Load metadata
    try:
        metadata = pd.read_csv(metadata_path, index_col='ecg_id')
        print(f"  Total records in metadata: {len(metadata)}")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None
    
    # Parse SCP codes
    try:
        metadata['scp_codes'] = metadata['scp_codes'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
        )
    except Exception as e:
        print(f"⚠️ Error parsing SCP codes: {e}")
        metadata['scp_codes'] = metadata['scp_codes'].apply(lambda x: {})
    
    # Limit samples if needed
    if max_samples and len(metadata) > max_samples:
        metadata = metadata.sample(n=max_samples, random_state=42)
        print(f"  Limited to {max_samples} samples")
    
    # Determine records folder
    records_folder = 'records100' if sampling_rate == 100 else 'records500'
    expected_length = 1000 if sampling_rate == 100 else 5000
    
    records_path = os.path.join(ptbxl_dir, records_folder)
    if not os.path.exists(records_path):
        print(f"❌ Records folder not found: {records_path}")
        print(f"  Available folders: {os.listdir(ptbxl_dir)}")
        return None
    
    print(f"  Records folder: {records_folder}")
    print(f"  Loading {len(metadata)} records...")
    
    X_images = []
    y_labels = []
    
    loaded = 0
    errors = 0
    
    for idx, (ecg_id, row) in enumerate(metadata.iterrows()):
        if idx % 500 == 0 and idx > 0:
            print(f"    Processed {idx}/{len(metadata)} ({loaded} loaded, {errors} errors)")
        
        try:
            # Get filename
            filename = row['filename_lr'] if sampling_rate == 100 else row['filename_hr']
            
            # Build full path
            file_path = os.path.join(ptbxl_dir, filename)
            
            # Remove extension if present (wfdb.rdsamp adds it)
            if file_path.endswith('.dat') or file_path.endswith('.hea'):
                file_path = file_path.rsplit('.', 1)[0]
            
            # Check if file exists
            if not os.path.exists(file_path + '.dat'):
                errors += 1
                continue
            
            # Load signal using wfdb
            signal, meta = wfdb.rdsamp(file_path)
            
            # Ensure correct length
            if len(signal) >= expected_length:
                signal = signal[:expected_length]
            else:
                # Pad if too short
                padding = np.zeros((expected_length - len(signal), signal.shape[1]))
                signal = np.vstack([signal, padding])
            
            # Normalize signal
            signal = (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)
            
            # Get class from SCP codes
            scp_codes = row.get('scp_codes', {})
            primary_class = 0  # Default to normal
            max_likelihood = 0
            
            if isinstance(scp_codes, dict):
                for code, likelihood in scp_codes.items():
                    code_upper = code.upper() if isinstance(code, str) else str(code)
                    if code_upper in SCP_TO_CLASS:
                        if likelihood > max_likelihood:
                            primary_class = SCP_TO_CLASS[code_upper]
                            max_likelihood = likelihood
            
            # Convert signal to grayscale image
            img = signal_to_grayscale_image(signal, (img_size, img_size))
            
            X_images.append(img)
            y_labels.append(primary_class)
            loaded += 1
            
        except Exception as e:
            errors += 1
            if errors < 5:  # Only show first few errors
                print(f"    Error loading record {ecg_id}: {e}")
            continue
    
    if loaded == 0:
        print("❌ No records could be loaded!")
        return None
    
    X_images = np.array(X_images)
    y_labels = np.array(y_labels)
    
    print(f"\n  ✓ Loaded {loaded} ECG records ({errors} errors)")
    print(f"  Image shape: {X_images.shape}")
    print(f"  Class distribution: {Counter(y_labels)}")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_labels),
        y=y_labels
    )
    class_weight_dict = {i: 1.0 for i in range(5)}  # Default
    for cls, weight in zip(np.unique(y_labels), class_weights):
        class_weight_dict[cls] = weight
    
    print(f"  Class weights: {class_weight_dict}")
    
    return X_images, y_labels, class_weight_dict


def create_synthetic_ecg_data(n_samples=500, img_size=224):
    """
    Create synthetic ECG images for testing when real data is unavailable.
    
    WARNING: This is for demonstration only!
    Real model should be trained on actual ECG data.
    """
    print("⚠️ Creating synthetic ECG images for demonstration...")
    print("   For accurate results, please use PTB-XL dataset!")
    
    np.random.seed(42)
    X = []
    y = []
    
    samples_per_class = n_samples // 5
    
    for class_idx in range(5):
        for _ in range(samples_per_class):
            # Create synthetic ECG signal
            t = np.linspace(0, 10, 1000)
            signal = np.zeros((1000, 12))
            
            for lead in range(12):
                # Base heartbeat pattern
                heart_rate = 1.0 + np.random.uniform(-0.2, 0.2)
                base = np.sin(2 * np.pi * heart_rate * t)
                
                # Add class-specific characteristics
                if class_idx == 0:  # Normal
                    ecg = 0.5 * base + 0.2 * np.sin(2 * np.pi * 2 * heart_rate * t)
                    
                elif class_idx == 1:  # MI - ST elevation
                    ecg = 0.5 * base
                    # Add ST elevation
                    for i in range(0, 1000, 100):
                        if i + 30 < 1000:
                            ecg[i:i+30] += 0.4
                    
                elif class_idx == 2:  # Arrhythmia - irregular rhythm
                    ecg = 0.5 * np.sin(2 * np.pi * (heart_rate + 0.3 * np.sin(0.5 * t)) * t)
                    # Add premature beats
                    ecg[300:320] += 0.8
                    ecg[700:720] += 0.8
                    
                elif class_idx == 3:  # Heart Failure - wide QRS
                    ecg = 0.4 * np.sin(2 * np.pi * 0.8 * heart_rate * t)
                    ecg += 0.3 * np.sin(2 * np.pi * 1.6 * heart_rate * t)
                    
                else:  # Hypertrophy - high voltage
                    ecg = 0.9 * base + 0.4 * np.sin(2 * np.pi * 2 * heart_rate * t)
                
                # Add noise
                noise = np.random.randn(1000) * 0.05
                signal[:, lead] = ecg * (1 + lead * 0.03) + noise
            
            # Normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            # Convert to grayscale image
            img = signal_to_grayscale_image(signal, (img_size, img_size))
            
            X.append(img)
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    print(f"  ✓ Created {len(X)} synthetic ECG images")
    print(f"  Shape: {X.shape}")
    
    return X, y


# ==============================================================================
#                           TRAINING FUNCTION
# ==============================================================================

def train_heart_image_model():
    """
    Train heart condition model on ECG IMAGES.
    
    Uses grayscale images to match server.py preprocessing.
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow required")
        return None
    
    print("\n" + "=" * 70)
    print("  HEART CONDITION DETECTION - IMAGE MODEL (FIXED)")
    print("  For analyzing ECG plot images")
    print("=" * 70)
    
    num_classes = 5
    
    # Try to load PTB-XL dataset
    data = load_ptbxl_dataset(sampling_rate=100, max_samples=3000, img_size=IMG_SIZE)
    
    if data is not None:
        X, y, class_weight_dict = data
        using_real_data = True
    else:
        # Fall back to synthetic data
        print("\n⚠️ Using synthetic data for demonstration...")
        X, y = create_synthetic_ecg_data(600, IMG_SIZE)
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(enumerate(class_weights))
        using_real_data = False
    
    # One-hot encode
    y_onehot = to_categorical(y, num_classes=num_classes)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Image shape: {X_train.shape}")
    print(f"  Using real data: {using_real_data}")
    
    # Create model
    print("\n🔧 Creating grayscale image-based model...")
    model = create_heart_image_model(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),  # Grayscale!
        num_classes=num_classes
    )
    model.summary()
    
    # Data augmentation (light - ECG images shouldn't be heavily augmented)
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=False,  # Don't flip ECGs!
        fill_mode='constant',
        cval=1.0  # White background
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            HEART_IMAGE_MODEL_PATH,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    print("\n" + "-" * 50)
    print("🚀 Training...")
    print("-" * 50)
    
    epochs = 30 if using_real_data else 15
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "-" * 50)
    print("📊 Evaluation...")
    print("-" * 50)
    
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Loss: {results[0]:.4f}")
    print(f"  Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall: {results[3]:.4f}")
    print(f"  AUC: {results[4]:.4f}")
    
    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\n  Per-class accuracy:")
    class_names = ['Normal', 'MI', 'Arrhythmia', 'Heart Failure', 'Hypertrophy']
    for cls in np.unique(y_true_classes):
        mask = y_true_classes == cls
        if mask.sum() > 0:
            acc = (y_pred_classes[mask] == cls).mean()
            print(f"    {class_names[cls]}: {acc:.4f} ({mask.sum()} samples)")
    
    # Prediction distribution
    print(f"\n  Prediction distribution:")
    for cls in range(num_classes):
        count = (y_pred_classes == cls).sum()
        print(f"    {class_names[cls]}: {count} ({count/len(y_pred_classes)*100:.1f}%)")
    
    # Save model
    model.save(HEART_IMAGE_MODEL_PATH)
    print(f"\n✓ Model saved: {HEART_IMAGE_MODEL_PATH}")
    
    # Save config
    config = {
        'model_path': HEART_IMAGE_MODEL_PATH,
        'model_type': 'image',
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'preprocessing': 'Grayscale, normalize to [0,1]',
        'num_classes': num_classes,
        'classes': {str(k): v for k, v in HEART_CLASSES.items()},
        'class_names': class_names,
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4]),
        'using_real_data': using_real_data
    }
    
    with open(HEART_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {HEART_CONFIG_PATH}")
    
    # Important reminders
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT REMINDERS:")
    print("=" * 70)
    print("1. This model uses GRAYSCALE preprocessing")
    print("2. Input size: 224x224")
    print("3. Output classes: 5 (Normal, MI, Arrhythmia, HF, Hypertrophy)")
    print("4. Server.py already uses grayscale for heart images ✓")
    if not using_real_data:
        print("\n⚠️  WARNING: Model was trained on SYNTHETIC data!")
        print("   For accurate predictions, retrain with PTB-XL dataset.")
    print("=" * 70)
    
    return model


# ==============================================================================
#                           MAIN ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    # Check dependencies
    if not WFDB_AVAILABLE:
        print("\n" + "=" * 70)
        print("⚠️  WFDB library not installed!")
        print("   Run: pip install wfdb")
        print("   This is required to load PTB-XL ECG data.")
        print("=" * 70)
    
    if not TF_AVAILABLE:
        print("\n❌ TensorFlow required. Install with: pip install tensorflow")
    else:
        train_heart_image_model()