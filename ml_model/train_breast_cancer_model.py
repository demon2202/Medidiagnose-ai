import os
import numpy as np
import json
import warnings
import glob
warnings.filterwarnings('ignore')

# TensorFlow imports
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} available")
except ImportError:
    print("✗ TensorFlow not available")

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')

BREAST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'breast_cancer_model.h5')
BREAST_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'breast_cancer_config.json')

# ============== IMPORTANT: Image size MUST match server.py ==============
IMG_SIZE = 224

# ==============================================================================
#                           CLASS DEFINITIONS
# ==============================================================================

# For Breast Ultrasound Dataset (3 classes)
BREAST_CLASSES_3 = {
    0: {
        'code': 'normal',
        'name': 'Normal',
        'severity': 'healthy',
        'description': 'No abnormalities detected in breast tissue.',
        'recommendation': 'Continue regular screening schedule.'
    },
    1: {
        'code': 'benign',
        'name': 'Benign Tumor',
        'severity': 'low',
        'description': 'Non-cancerous growth detected. Common and usually not dangerous.',
        'recommendation': 'Follow-up imaging recommended. Consult with specialist.'
    },
    2: {
        'code': 'malignant',
        'name': 'Malignant Tumor',
        'severity': 'critical',
        'description': 'Cancerous growth detected. Immediate medical attention required.',
        'recommendation': 'Urgent consultation with oncologist required.'
    }
}

# Extended BI-RADS style classification (6 classes) - for server compatibility
# Maps 3-class predictions to 6-class output
BREAST_CLASSES_6 = {
    0: {'code': 'normal', 'name': 'Normal', 'birads': 'BI-RADS 1', 'severity': 'healthy'},
    1: {'code': 'benign', 'name': 'Benign Finding', 'birads': 'BI-RADS 2', 'severity': 'low'},
    2: {'code': 'probably_benign', 'name': 'Probably Benign', 'birads': 'BI-RADS 3', 'severity': 'low'},
    3: {'code': 'suspicious', 'name': 'Suspicious', 'birads': 'BI-RADS 4', 'severity': 'moderate'},
    4: {'code': 'highly_suggestive', 'name': 'Highly Suggestive', 'birads': 'BI-RADS 5', 'severity': 'high'},
    5: {'code': 'malignant', 'name': 'Malignant', 'birads': 'BI-RADS 6', 'severity': 'critical'}
}

# Mapping from 3-class to 6-class (for server compatibility)
CLASS_3_TO_6_MAPPING = {
    0: 0,  # normal -> normal (BI-RADS 1)
    1: 1,  # benign -> benign (BI-RADS 2)
    2: 5   # malignant -> malignant (BI-RADS 6)
}


# ==============================================================================
#                           MODEL ARCHITECTURES
# ==============================================================================

def create_breast_cancer_model_3class(input_shape=(224, 224, 1), num_classes=3):
    """
    Create CNN model for breast ultrasound classification - 3 classes.
    
    IMPORTANT: Uses GRAYSCALE input to match server.py preprocessing!
    
    Input: Grayscale image (224, 224, 1)
    Output: 3 classes (normal, benign, malignant)
    """
    
    # For grayscale, we can't use ImageNet pretrained weights directly
    # We'll use a custom architecture optimized for ultrasound
    
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
        
        # Output: 3 classes
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


def create_breast_cancer_model_6class(input_shape=(224, 224, 1), num_classes=6):
    """
    Create CNN model for breast cancer - 6 classes (BI-RADS style).
    
    This is for compatibility with server.py BREAST_CANCER_CLASSES.
    Use this if you have a dataset with 6 severity levels.
    
    Input: Grayscale image (224, 224, 1)
    Output: 6 classes (BI-RADS 1-6)
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
        
        # Output: 6 classes
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


def create_breast_cancer_model_transfer(input_shape=(224, 224, 3), num_classes=3):
    """
    Create model with transfer learning for RGB images.
    
    Use this if you want to use pretrained ImageNet weights.
    NOTE: This requires RGB input - make sure server.py also uses RGB!
    
    Input: RGB image (224, 224, 3)
    Output: 3 or 6 classes
    """
    
    # Use EfficientNetB0 for better accuracy
    try:
        base_model = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    except:
        # Fallback to MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
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
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model


# ==============================================================================
#                           DATA LOADING
# ==============================================================================

def find_breast_ultrasound_dataset():
    """
    Find the breast ultrasound dataset in various possible locations.
    
    Returns:
        tuple: (data_dir, dataset_type) or (None, None) if not found
    """
    possible_names = [
        'breast_ultrasound',
        'Breast_Ultrasound',
        'breast-ultrasound',
        'Dataset_BUSI_with_GT',
        'BUSI',
        'busi',
        'breast_ultrasound_images'
    ]
    
    for name in possible_names:
        check_dir = os.path.join(DATASET_DIR, name)
        if os.path.exists(check_dir):
            # Check for class folders
            subdirs = [d for d in os.listdir(check_dir) 
                      if os.path.isdir(os.path.join(check_dir, d))]
            
            # Check if it has the expected structure
            subdirs_lower = [d.lower() for d in subdirs]
            
            if any('benign' in s for s in subdirs_lower) or \
               any('malignant' in s for s in subdirs_lower) or \
               any('normal' in s for s in subdirs_lower):
                print(f"  Found dataset at: {check_dir}")
                print(f"  Subdirectories: {subdirs}")
                return check_dir, 'ultrasound'
    
    return None, None


def load_breast_ultrasound_data(img_size=224, use_grayscale=True):
    """
    Load Breast Ultrasound Images Dataset.
    
    IMPORTANT: Preprocessing MUST match server.py!
    Server.py uses grayscale=True for breast images.
    
    Dataset: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
    
    Expected structure:
        Dataset/breast_ultrasound/
        ├── benign/
        ├── malignant/
        └── normal/
    
    Args:
        img_size: Target image size (default 224)
        use_grayscale: Use grayscale (True) or RGB (False)
                       MUST match server.py preprocessing!
    
    Returns:
        X_train, X_test, y_train, y_test, class_weights, num_classes
    """
    
    # Find dataset
    data_dir, dataset_type = find_breast_ultrasound_dataset()
    
    if data_dir is None:
        print(f"❌ Breast ultrasound dataset not found")
        print("\n📥 Download from: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print(f"\n📁 Extract to: {os.path.join(DATASET_DIR, 'breast_ultrasound')}")
        print("\nExpected structure:")
        print("  breast_ultrasound/")
        print("  ├── benign/")
        print("  ├── malignant/")
        print("  └── normal/")
        return None
    
    print(f"📂 Loading Breast Ultrasound dataset from {data_dir}...")
    print(f"  Preprocessing: {'Grayscale' if use_grayscale else 'RGB'}")
    
    # Find class folders
    class_folders = {}
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            folder_lower = folder.lower()
            if 'normal' in folder_lower:
                class_folders['normal'] = folder_path
            elif 'benign' in folder_lower:
                class_folders['benign'] = folder_path
            elif 'malignant' in folder_lower:
                class_folders['malignant'] = folder_path
    
    if len(class_folders) < 2:
        print(f"❌ Could not find expected class folders in {data_dir}")
        print(f"  Found: {list(class_folders.keys())}")
        print("  Expected: normal, benign, malignant")
        return None
    
    print(f"  Found class folders: {list(class_folders.keys())}")
    
    # Class mapping for 3-class
    class_to_idx = {
        'normal': 0,
        'benign': 1,
        'malignant': 2
    }
    
    X = []
    y = []
    
    for class_name, folder_path in class_folders.items():
        print(f"  Loading {class_name}...")
        
        # Find all images
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            images.extend(glob.glob(os.path.join(folder_path, ext)))
        
        # Filter out mask images (BUSI dataset has _mask files)
        images = [img for img in images if 'mask' not in img.lower()]
        
        print(f"    Found {len(images)} images (excluding masks)")
        
        loaded = 0
        for img_path in images:
            try:
                # Load image
                img = Image.open(img_path)
                
                # Resize
                img = img.resize((img_size, img_size), Image.LANCZOS)
                
                # Convert based on settings
                if use_grayscale:
                    img = img.convert('L')  # Grayscale
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
                else:
                    img = img.convert('RGB')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                
                X.append(img_array)
                y.append(class_to_idx[class_name])
                loaded += 1
                
            except Exception as e:
                print(f"    Error loading {img_path}: {e}")
                continue
        
        print(f"    Loaded {loaded} images")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n  Total loaded: {len(X)} images")
    print(f"  Image shape: {X.shape}")
    print(f"  Class distribution: {Counter(y)}")
    
    # Check for class imbalance
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    if max_count > 3 * min_count:
        print(f"\n  ⚠️ Class imbalance detected!")
        print(f"    Min class: {min_count}, Max class: {max_count}")
        print(f"    Will use class weights to handle this.")
    
    # One-hot encode
    num_classes = len(class_folders)
    y_onehot = to_categorical(y, num_classes=num_classes)
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Calculate class weights
    y_train_int = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"  Class weights: {class_weight_dict}")
    
    return X_train, X_test, y_train, y_test, class_weight_dict, num_classes


# ==============================================================================
#                           TRAINING FUNCTION
# ==============================================================================

def train_breast_cancer_model(use_grayscale=True, use_6_classes=False):
    """
    Train breast cancer detection model.
    
    Args:
        use_grayscale: Use grayscale preprocessing (must match server.py!)
        use_6_classes: Output 6 classes (BI-RADS) instead of 3
    
    IMPORTANT: 
    - Server.py currently uses grayscale=True for breast images
    - If you change this, update server.py accordingly!
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow required")
        return None
    
    print("\n" + "=" * 70)
    print("  BREAST CANCER DETECTION MODEL TRAINING - FIXED")
    print("  Dataset: Breast Ultrasound Images")
    print("=" * 70)
    
    # Load data
    data = load_breast_ultrasound_data(img_size=IMG_SIZE, use_grayscale=use_grayscale)
    
    if data is None:
        print("\n⚠️ Dataset not found. Cannot train model.")
        return None
    
    X_train, X_test, y_train, y_test, class_weight_dict, num_classes = data
    
    # Determine input shape
    if use_grayscale:
        input_shape = (IMG_SIZE, IMG_SIZE, 1)
    else:
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
    
    print(f"\n  Input shape: {input_shape}")
    print(f"  Output classes: {num_classes}")
    
    # Handle 6-class output if requested
    if use_6_classes and num_classes == 3:
        print("\n  Converting 3-class labels to 6-class for server compatibility...")
        # Map 3-class to 6-class
        y_train_int = np.argmax(y_train, axis=1)
        y_test_int = np.argmax(y_test, axis=1)
        
        y_train_6 = np.array([CLASS_3_TO_6_MAPPING[y] for y in y_train_int])
        y_test_6 = np.array([CLASS_3_TO_6_MAPPING[y] for y in y_test_int])
        
        y_train = to_categorical(y_train_6, num_classes=6)
        y_test = to_categorical(y_test_6, num_classes=6)
        
        # Recalculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_6),
            y=y_train_6
        )
        # Fill in missing classes with 1.0
        class_weight_dict = {i: 1.0 for i in range(6)}
        for cls, weight in zip(np.unique(y_train_6), class_weights):
            class_weight_dict[cls] = weight
        
        num_classes = 6
        print(f"  New class weights: {class_weight_dict}")
    
    # Create model
    print(f"\n🔧 Creating {'grayscale' if use_grayscale else 'RGB'} model with {num_classes} classes...")
    
    if use_grayscale:
        if num_classes == 3:
            model = create_breast_cancer_model_3class(input_shape, num_classes)
        else:
            model = create_breast_cancer_model_6class(input_shape, num_classes)
    else:
        model = create_breast_cancer_model_transfer(input_shape, num_classes)
    
    model.summary()
    
    # Data augmentation for ultrasound images
    if use_grayscale:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,  # Ultrasound can be any orientation
            zoom_range=0.1,
            shear_range=0.05,
            fill_mode='constant',
            cval=0  # Black fill
        )
    else:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.15,
            shear_range=0.1,
            fill_mode='nearest'
        )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            BREAST_MODEL_PATH,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training Phase 1
    print("\n" + "-" * 50)
    print("🚀 Training Phase 1...")
    print("-" * 50)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Fine-tuning for transfer learning
    if not use_grayscale:
        print("\n" + "-" * 50)
        print("🚀 Training Phase 2 (fine-tuning)...")
        print("-" * 50)
        
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Freeze all but last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=16),
            epochs=15,
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
    
    # Detailed predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\n  Per-class accuracy:")
    unique_classes = np.unique(y_true_classes)
    class_names = ['normal', 'benign', 'probably_benign', 'suspicious', 'highly_suggestive', 'malignant'] if num_classes == 6 else ['normal', 'benign', 'malignant']
    
    for cls in unique_classes:
        mask = y_true_classes == cls
        if mask.sum() > 0:
            acc = (y_pred_classes[mask] == cls).mean()
            print(f"    {class_names[cls]}: {acc:.4f} ({mask.sum()} samples)")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(f"\n  Confusion Matrix:")
    print(cm)
    
    # Check prediction distribution
    print(f"\n  Prediction distribution:")
    for cls in range(num_classes):
        count = (y_pred_classes == cls).sum()
        if count > 0:
            print(f"    {class_names[cls]}: {count} ({count/len(y_pred_classes)*100:.1f}%)")
    
    # Save model
    model.save(BREAST_MODEL_PATH)
    print(f"\n✓ Model saved: {BREAST_MODEL_PATH}")
    
    # Save config
    if num_classes == 3:
        classes_config = {str(k): v for k, v in BREAST_CLASSES_3.items()}
    else:
        classes_config = {str(k): v for k, v in BREAST_CLASSES_6.items()}
    
    config = {
        'model_path': BREAST_MODEL_PATH,
        'input_shape': list(input_shape),
        'preprocessing': 'Grayscale, normalize to [0,1]' if use_grayscale else 'RGB, normalize to [0,1]',
        'use_grayscale': use_grayscale,
        'num_classes': num_classes,
        'classes': classes_config,
        'class_names': class_names[:num_classes],
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4])
    }
    
    with open(BREAST_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {BREAST_CONFIG_PATH}")
    
    # Print important reminder
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT REMINDERS:")
    print("=" * 70)
    print(f"1. This model uses {'GRAYSCALE' if use_grayscale else 'RGB'} preprocessing")
    print(f"2. Input size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"3. Output classes: {num_classes}")
    print(f"4. Make sure server.py uses the same preprocessing!")
    print("=" * 70)
    
    return model


# ==============================================================================
#                           MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main training function with options"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow required")
        print("Install with: pip install tensorflow")
        return
    
    print("\n" + "=" * 70)
    print("  BREAST CANCER MODEL TRAINING - CONFIGURATION")
    print("=" * 70)
    
    print("\nOptions:")
    print("  1. Train 3-class model (normal, benign, malignant) - RECOMMENDED")
    print("  2. Train 6-class model (BI-RADS style)")
    print("  3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Train 3-class grayscale model (matches server.py)
        train_breast_cancer_model(use_grayscale=True, use_6_classes=False)
    elif choice == '2':
        # Train 6-class grayscale model
        train_breast_cancer_model(use_grayscale=True, use_6_classes=True)
    else:
        print("Exiting...")
        return
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print("\n⚠️  Restart server.py to load the new model!")


if __name__ == '__main__':
    main()