import os
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Check TensorFlow availability
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} available")
except ImportError:
    print("✗ TensorFlow not available")

from PIL import Image
import glob
from collections import Counter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')

# Output paths
SKIN_MODEL_PATH = os.path.join(SCRIPT_DIR, 'skin_cancer_model.h5')
SKIN_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'skin_cancer_config.json')
PNEUMONIA_MODEL_PATH = os.path.join(SCRIPT_DIR, 'pneumonia_model.h5')
PNEUMONIA_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'pneumonia_config.json')

# ============== IMPORTANT: Image size MUST match server.py ==============
IMG_SIZE = 224

# HAM10000 class mapping - MUST match server.py SKIN_CANCER_CLASSES
HAM10000_CLASSES = {
    'akiec': 0,  # Actinic keratoses - pre-cancerous
    'bcc': 1,    # Basal cell carcinoma - malignant
    'bkl': 2,    # Benign keratosis - benign
    'df': 3,     # Dermatofibroma - benign
    'mel': 4,    # Melanoma - malignant (most dangerous)
    'nv': 5,     # Melanocytic nevi - benign (moles)
    'vasc': 6    # Vascular lesions - benign
}

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_INFO = {
    0: {'name': 'Actinic Keratoses', 'type': 'pre-cancerous', 'severity': 'moderate'},
    1: {'name': 'Basal Cell Carcinoma', 'type': 'malignant', 'severity': 'moderate'},
    2: {'name': 'Benign Keratosis', 'type': 'benign', 'severity': 'low'},
    3: {'name': 'Dermatofibroma', 'type': 'benign', 'severity': 'low'},
    4: {'name': 'Melanoma', 'type': 'malignant', 'severity': 'critical'},
    5: {'name': 'Melanocytic Nevi', 'type': 'benign', 'severity': 'low'},
    6: {'name': 'Vascular Lesions', 'type': 'benign', 'severity': 'low'}
}


# ==============================================================================
#                           SKIN CANCER MODEL
# ==============================================================================

def create_skin_cancer_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Create CNN model for skin cancer classification using transfer learning.
    
    Uses MobileNetV2 which is:
    - Lightweight and fast
    - Good for medical imaging
    - Works well with limited data
    
    Input: RGB image (224, 224, 3)
    Output: 7 classes (HAM10000 categories)
    """
    
    # Use MobileNetV2 as base - pretrained on ImageNet
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially (will unfreeze for fine-tuning)
    base_model.trainable = False
    
    # Build model
    model = models.Sequential([
        # Base model
        base_model,
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers with batch norm and dropout
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
        
        # Output layer - 7 classes with softmax
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


def create_skin_cancer_model_simple(input_shape=(224, 224, 3), num_classes=7):
    """
    Custom CNN model without transfer learning.
    Use this if you have limited GPU memory or want faster training.
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
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model


def load_ham10000_data(img_size=224, max_samples_per_class=None):
    """
    Load HAM10000 dataset with proper preprocessing.
    
    IMPORTANT: Preprocessing here MUST match server.py:
    - RGB images (not grayscale)
    - Normalized to [0, 1]
    - Size: 224x224
    
    Args:
        img_size: Target image size (default 224 to match server)
        max_samples_per_class: Limit samples per class for balanced training
    
    Returns:
        X_train, X_test, y_train, y_test, class_weights
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    
    ham_dir = os.path.join(DATASET_DIR, 'HAM10000')
    metadata_path = os.path.join(ham_dir, 'HAM10000_metadata.csv')
    
    if not os.path.exists(metadata_path):
        print(f"❌ HAM10000 metadata not found at {metadata_path}")
        print("\n📥 Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"\n📁 Extract to: {ham_dir}")
        print("\nExpected structure:")
        print("  HAM10000/")
        print("  ├── HAM10000_images_part_1/")
        print("  ├── HAM10000_images_part_2/")
        print("  └── HAM10000_metadata.csv")
        return None
    
    print("📂 Loading HAM10000 dataset...")
    metadata = pd.read_csv(metadata_path)
    print(f"  Total entries in metadata: {len(metadata)}")
    
    # Find image directories
    image_dirs = []
    possible_folders = ['HAM10000_images_part_1', 'HAM10000_images_part_2',
                        'HAM10000_images', 'images', 'train', 'all_images']
    
    for folder in possible_folders:
        path = os.path.join(ham_dir, folder)
        if os.path.exists(path):
            image_dirs.append(path)
            print(f"  Found image folder: {folder}")
    
    if not image_dirs:
        print("❌ No image directories found")
        return None
    
    # Create image path mapping
    image_paths = {}
    for img_dir in image_dirs:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in glob.glob(os.path.join(img_dir, ext)):
                img_id = os.path.splitext(os.path.basename(img_path))[0]
                image_paths[img_id] = img_path
    
    print(f"  Found {len(image_paths)} images")
    
    # Organize by class
    class_images = {cls: [] for cls in HAM10000_CLASSES.keys()}
    
    for _, row in metadata.iterrows():
        img_id = row['image_id']
        if img_id in image_paths:
            dx = row['dx']
            if dx in class_images:
                class_images[dx].append(image_paths[img_id])
    
    # Print class distribution
    print("\n  Class distribution:")
    total_images = 0
    for cls, images in class_images.items():
        print(f"    {cls}: {len(images)} images")
        total_images += len(images)
    print(f"  Total: {total_images} images")
    
    # Balance classes by limiting samples
    if max_samples_per_class:
        print(f"\n  Balancing classes (max {max_samples_per_class} per class)...")
        for cls in class_images:
            if len(class_images[cls]) > max_samples_per_class:
                np.random.shuffle(class_images[cls])
                class_images[cls] = class_images[cls][:max_samples_per_class]
    
    # Load images
    X = []
    y = []
    
    total = sum(len(imgs) for imgs in class_images.values())
    loaded = 0
    
    print(f"\n  Loading and preprocessing {total} images...")
    
    for cls, img_paths in class_images.items():
        class_idx = HAM10000_CLASSES[cls]
        
        for img_path in img_paths:
            try:
                # Load image
                img = Image.open(img_path)
                
                # Resize to target size
                img = img.resize((img_size, img_size), Image.LANCZOS)
                
                # Convert to RGB (IMPORTANT: must match server.py)
                img = img.convert('RGB')
                
                # Normalize to [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                X.append(img_array)
                y.append(class_idx)
                
                loaded += 1
                if loaded % 500 == 0:
                    print(f"    Loaded {loaded}/{total} images...")
                    
            except Exception as e:
                print(f"    Error loading {img_path}: {e}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n  ✓ Loaded {len(X)} images successfully")
    print(f"  Shape: {X.shape}")
    print(f"  Final class distribution: {Counter(y)}")
    
    # One-hot encode labels
    y_onehot = to_categorical(y, num_classes=7)
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Calculate class weights for imbalanced data
    y_train_int = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n  Class weights: {class_weight_dict}")
    
    return X_train, X_test, y_train, y_test, class_weight_dict


def train_skin_cancer_model(use_transfer_learning=True, max_samples_per_class=1500):
    """
    Train skin cancer detection model.
    
    Args:
        use_transfer_learning: Use MobileNetV2 (True) or custom CNN (False)
        max_samples_per_class: Limit for class balancing
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow is required")
        return None
    
    print("\n" + "=" * 70)
    print("  SKIN CANCER DETECTION MODEL TRAINING")
    print("  Dataset: HAM10000 (Human Against Machine 10000)")
    print("=" * 70)
    
    # Load data
    data = load_ham10000_data(img_size=IMG_SIZE, max_samples_per_class=max_samples_per_class)
    
    if data is None:
        print("\n⚠️ Dataset not found. Cannot train model.")
        print("Please download HAM10000 dataset first.")
        return None
    
    X_train, X_test, y_train, y_test, class_weight_dict = data
    
    # Create model
    if use_transfer_learning:
        print("\n🔧 Creating model with MobileNetV2 transfer learning...")
        model = create_skin_cancer_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=7)
    else:
        print("\n🔧 Creating custom CNN model...")
        model = create_skin_cancer_model_simple(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=7)
    
    model.summary()
    
    # Data augmentation - important for skin lesion images
    datagen = ImageDataGenerator(
        rotation_range=180,  # Skin lesions can be any orientation
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='reflect',  # Better for skin images
        brightness_range=[0.8, 1.2]  # Lighting variation
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
            SKIN_MODEL_PATH,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base
    print("\n" + "-" * 50)
    print("🚀 Training Phase 1 (frozen base)...")
    print("-" * 50)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=25,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Phase 2: Fine-tuning (unfreeze top layers)
    if use_transfer_learning:
        print("\n" + "-" * 50)
        print("🚀 Training Phase 2 (fine-tuning)...")
        print("-" * 50)
        
        # Unfreeze top layers of base model
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
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
    for i, cls_name in enumerate(CLASS_NAMES):
        mask = y_true_classes == i
        if mask.sum() > 0:
            acc = (y_pred_classes[mask] == i).mean()
            print(f"    {cls_name}: {acc:.4f} ({mask.sum()} samples)")
    
    # Save model
    model.save(SKIN_MODEL_PATH)
    print(f"\n✓ Model saved: {SKIN_MODEL_PATH}")
    
    # Save config
    config = {
        'model_path': SKIN_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'preprocessing': 'RGB, normalize to [0,1]',
        'num_classes': 7,
        'class_names': CLASS_NAMES,
        'class_mapping': HAM10000_CLASSES,
        'class_info': CLASS_INFO,
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4])
    }
    
    with open(SKIN_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {SKIN_CONFIG_PATH}")
    
    return model


# ==============================================================================
#                           PNEUMONIA MODEL - FIXED
# ==============================================================================

def create_pneumonia_model(input_shape=(224, 224, 1), num_classes=2):
    """
    CNN model for pneumonia detection - FIXED VERSION.
    
    Changes from original:
    - Output 2 classes (softmax) instead of 1 (sigmoid) for better calibration
    - Added more regularization
    - Better architecture for grayscale X-ray images
    
    Input: Grayscale image (224, 224, 1)
    Output: 2 classes [Normal, Pneumonia]
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
        
        # FIXED: Use 2 classes with softmax instead of 1 with sigmoid
        # This gives better calibrated probabilities
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',  # Changed from binary_crossentropy
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model


def load_chest_xray_data(img_size=224):
    """
    Load Chest X-Ray Pneumonia dataset with proper preprocessing.
    
    IMPORTANT: Preprocessing here MUST match server.py:
    - Grayscale images
    - Normalized to [0, 1]
    - Size: 224x224
    - Shape: (H, W, 1)
    
    FIXES:
    - Proper class balancing
    - Consistent preprocessing
    - Returns arrays instead of generators for better control
    """
    xray_dir = os.path.join(DATASET_DIR, 'chest_xray')
    train_dir = os.path.join(xray_dir, 'train')
    test_dir = os.path.join(xray_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"❌ Chest X-Ray training data not found at {train_dir}")
        print("\n📥 Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print(f"\n📁 Extract to: {xray_dir}")
        print("\nExpected structure:")
        print("  chest_xray/")
        print("  ├── train/")
        print("  │   ├── NORMAL/")
        print("  │   └── PNEUMONIA/")
        print("  ├── test/")
        print("  │   ├── NORMAL/")
        print("  │   └── PNEUMONIA/")
        print("  └── val/")
        return None
    
    print("📂 Loading Chest X-Ray dataset...")
    
    def load_images_from_folder(folder, label, max_samples=None):
        """Load images from a folder with consistent preprocessing"""
        images = []
        labels = []
        
        all_files = glob.glob(os.path.join(folder, '*'))
        all_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if max_samples and len(all_files) > max_samples:
            np.random.shuffle(all_files)
            all_files = all_files[:max_samples]
        
        for img_path in all_files:
            try:
                # Load as grayscale
                img = Image.open(img_path).convert('L')
                
                # Resize
                img = img.resize((img_size, img_size), Image.LANCZOS)
                
                # Normalize to [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Add channel dimension (H, W) -> (H, W, 1)
                img_array = np.expand_dims(img_array, axis=-1)
                
                images.append(img_array)
                labels.append(label)
                
            except Exception as e:
                continue
        
        return images, labels
    
    # Load training data
    train_normal_dir = os.path.join(train_dir, 'NORMAL')
    train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')
    
    # Count available images
    normal_count = len(glob.glob(os.path.join(train_normal_dir, '*')))
    pneumonia_count = len(glob.glob(os.path.join(train_pneumonia_dir, '*')))
    
    print(f"  Training data available:")
    print(f"    NORMAL: {normal_count}")
    print(f"    PNEUMONIA: {pneumonia_count}")
    
    # IMPORTANT: Balance the classes!
    # The chest_xray dataset is heavily imbalanced (much more pneumonia)
    # We need to either undersample pneumonia or oversample normal
    
    # Option 1: Undersample pneumonia to match normal
    max_samples = min(normal_count, pneumonia_count)
    print(f"\n  Balancing classes to {max_samples} samples each...")
    
    print(f"\n  Loading NORMAL images...")
    X_normal, y_normal = load_images_from_folder(train_normal_dir, 0, max_samples)
    print(f"    Loaded {len(X_normal)} NORMAL images")
    
    print(f"  Loading PNEUMONIA images...")
    X_pneumonia, y_pneumonia = load_images_from_folder(train_pneumonia_dir, 1, max_samples)
    print(f"    Loaded {len(X_pneumonia)} PNEUMONIA images")
    
    # Combine
    X_train = np.array(X_normal + X_pneumonia)
    y_train = np.array(y_normal + y_pneumonia)
    
    # Shuffle
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print(f"\n  Total training samples: {len(X_train)}")
    print(f"  Training class distribution: {Counter(y_train)}")
    
    # Load test data
    print(f"\n  Loading test data...")
    test_normal_dir = os.path.join(test_dir, 'NORMAL')
    test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
    
    X_test_normal, y_test_normal = load_images_from_folder(test_normal_dir, 0)
    X_test_pneumonia, y_test_pneumonia = load_images_from_folder(test_pneumonia_dir, 1)
    
    X_test = np.array(X_test_normal + X_test_pneumonia)
    y_test = np.array(y_test_normal + y_test_pneumonia)
    
    # Shuffle test data
    indices = np.random.permutation(len(X_test))
    X_test = X_test[indices]
    y_test = y_test[indices]
    
    print(f"  Total test samples: {len(X_test)}")
    print(f"  Test class distribution: {Counter(y_test)}")
    
    print(f"\n  Data shape: {X_train.shape}")
    
    return X_train, X_test, y_train, y_test


def train_pneumonia_model():
    """
    Train pneumonia detection model - FIXED VERSION.
    
    Fixes:
    - Proper class balancing
    - 2-class output instead of binary
    - Better data augmentation
    - Correct preprocessing
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow is required")
        return None
    
    print("\n" + "=" * 70)
    print("  PNEUMONIA DETECTION MODEL TRAINING - FIXED")
    print("  Dataset: Chest X-Ray Images (Pneumonia)")
    print("=" * 70)
    
    # Load data
    data = load_chest_xray_data(img_size=IMG_SIZE)
    
    if data is None:
        print("\n⚠️ Dataset not found. Cannot train model.")
        return None
    
    X_train, X_test, y_train_int, y_test_int = data
    
    # Convert to one-hot encoding (2 classes)
    y_train = to_categorical(y_train_int, num_classes=2)
    y_test = to_categorical(y_test_int, num_classes=2)
    
    # Calculate class weights (should be balanced but just in case)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n  Class weights: {class_weight_dict}")
    
    # Create model
    print("\n🔧 Creating CNN model...")
    model = create_pneumonia_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=2)
    model.summary()
    
    # Data augmentation for X-ray images
    datagen = ImageDataGenerator(
        rotation_range=10,  # X-rays shouldn't be rotated much
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,  # X-rays can be flipped
        fill_mode='constant',
        cval=0  # Fill with black (like X-ray background)
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
            PNEUMONIA_MODEL_PATH,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "-" * 50)
    print("🚀 Training...")
    print("-" * 50)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "-" * 50)
    print("📊 Evaluation on test set...")
    print("-" * 50)
    
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Loss: {results[0]:.4f}")
    print(f"  Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall: {results[3]:.4f}")
    print(f"  AUC: {results[4]:.4f}")
    
    # Detailed analysis
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_test_int, y_pred_classes)
    print(f"\n  Confusion Matrix:")
    print(f"    TN (Normal correct): {cm[0,0]}")
    print(f"    FP (Normal as Pneumonia): {cm[0,1]}")
    print(f"    FN (Pneumonia as Normal): {cm[1,0]}")
    print(f"    TP (Pneumonia correct): {cm[1,1]}")
    
    # Check prediction distribution
    print(f"\n  Prediction distribution:")
    print(f"    Predicted Normal: {(y_pred_classes == 0).sum()}")
    print(f"    Predicted Pneumonia: {(y_pred_classes == 1).sum()}")
    
    # Check probability distribution
    print(f"\n  Probability statistics:")
    normal_probs = y_pred[:, 0]
    pneumonia_probs = y_pred[:, 1]
    print(f"    Normal prob - mean: {normal_probs.mean():.3f}, std: {normal_probs.std():.3f}")
    print(f"    Pneumonia prob - mean: {pneumonia_probs.mean():.3f}, std: {pneumonia_probs.std():.3f}")
    
    # Save model
    model.save(PNEUMONIA_MODEL_PATH)
    print(f"\n✓ Model saved: {PNEUMONIA_MODEL_PATH}")
    
    # Save config
    config = {
        'model_path': PNEUMONIA_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'preprocessing': 'Grayscale, normalize to [0,1]',
        'num_classes': 2,
        'class_names': ['NORMAL', 'PNEUMONIA'],
        'output_type': 'softmax_2class',  # Important for server.py
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4]),
        'confusion_matrix': {
            'TN': int(cm[0,0]),
            'FP': int(cm[0,1]),
            'FN': int(cm[1,0]),
            'TP': int(cm[1,1])
        }
    }
    
    with open(PNEUMONIA_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {PNEUMONIA_CONFIG_PATH}")
    
    return model


# ==============================================================================
#                           MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main training pipeline"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow is required for training")
        print("Install with: pip install tensorflow")
        return
    
    print("\n" + "=" * 70)
    print("  MediDiagnose-AI: Image Classification Training Pipeline")
    print("  FIXED VERSION - Proper preprocessing and class balancing")
    print("=" * 70)
    
    # Create directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Menu
    print("\nSelect model to train:")
    print("  1. Skin Cancer Detection (HAM10000)")
    print("  2. Pneumonia Detection (Chest X-Ray) - FIXED")
    print("  3. Both models")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        train_skin_cancer_model()
    elif choice == '2':
        train_pneumonia_model()
    elif choice == '3':
        train_skin_cancer_model()
        train_pneumonia_model()
    else:
        print("Exiting...")
        return
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print("\n⚠️ IMPORTANT: After training, restart your server.py to load new models!")


if __name__ == '__main__':
    main()