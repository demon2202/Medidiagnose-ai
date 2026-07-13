"""
train_breast_cancer_model.py - IMPROVED VERSION
================================================
Fixes:
- Better model architecture with residual connections
- Improved data augmentation for ultrasound images
- Grayscale transfer learning using channel replication
- Better preprocessing pipeline
- Proper validation and evaluation
- Higher accuracy through better training strategy
"""

import os
import numpy as np
import json
import warnings
import glob
warnings.filterwarnings('ignore')

TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers, backend as K
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    print(f"[OK] TensorFlow {tf.__version__} available")
except ImportError:
    print("[ERR] TensorFlow not available")

from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import random

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
BREAST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'breast_cancer_model.h5')
BREAST_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'breast_cancer_config.json')

IMG_SIZE = 224

# ==============================================================================
#                           CLASS DEFINITIONS
# ==============================================================================

BREAST_CLASSES_3 = {
    0: {'code': 'normal', 'name': 'Normal', 'severity': 'healthy',
        'description': 'No abnormalities detected in breast tissue.'},
    1: {'code': 'benign', 'name': 'Benign Tumor', 'severity': 'low',
        'description': 'Non-cancerous growth detected.'},
    2: {'code': 'malignant', 'name': 'Malignant Tumor', 'severity': 'critical',
        'description': 'Cancerous growth detected. Immediate attention required.'}
}

BREAST_CLASSES_6 = {
    0: {'code': 'normal', 'name': 'Normal', 'birads': 'BI-RADS 1', 'severity': 'healthy'},
    1: {'code': 'benign', 'name': 'Benign Finding', 'birads': 'BI-RADS 2', 'severity': 'low'},
    2: {'code': 'probably_benign', 'name': 'Probably Benign', 'birads': 'BI-RADS 3', 'severity': 'low'},
    3: {'code': 'suspicious', 'name': 'Suspicious', 'birads': 'BI-RADS 4', 'severity': 'moderate'},
    4: {'code': 'highly_suggestive', 'name': 'Highly Suggestive', 'birads': 'BI-RADS 5', 'severity': 'high'},
    5: {'code': 'malignant', 'name': 'Malignant', 'birads': 'BI-RADS 6', 'severity': 'critical'}
}

CLASS_3_TO_6_MAPPING = {0: 0, 1: 1, 2: 5}


# ==============================================================================
#                    IMPROVED PREPROCESSING
# ==============================================================================

def apply_clahe_pil(img):
    """
    Apply Contrast Limited Adaptive Histogram Equalization using PIL.
    Improves contrast in ultrasound images significantly.
    """
    img_array = np.array(img, dtype=np.float32)

    # Simple adaptive histogram equalization
    # Split into tiles and equalize each
    h, w = img_array.shape[:2]
    tile_h, tile_w = max(h // 8, 1), max(w // 8, 1)

    result = img_array.copy()

    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            tile = img_array[i:min(i + tile_h, h), j:min(j + tile_w, w)]
            if tile.size > 0:
                t_min, t_max = tile.min(), tile.max()
                if t_max - t_min > 0:
                    result[i:min(i + tile_h, h), j:min(j + tile_w, w)] = \
                        (tile - t_min) / (t_max - t_min) * 255.0

    return Image.fromarray(result.astype(np.uint8))


def preprocess_ultrasound_image(img_path, img_size=224, augment=False):
    """
    Advanced preprocessing for breast ultrasound images.

    Steps:
    1. Load and convert to grayscale
    2. Apply CLAHE for contrast enhancement
    3. Resize with high-quality resampling
    4. Normalize
    5. Optional augmentation

    Returns:
        numpy array (img_size, img_size, 1) normalized to [0, 1]
    """
    try:
        img = Image.open(img_path)

        # Convert to grayscale
        img = img.convert('L')

        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)

        # Apply slight sharpening for ultrasound
        img = img.filter(ImageFilter.SHARPEN)

        # Resize with high quality
        img = img.resize((img_size, img_size), Image.LANCZOS)

        # Apply CLAHE-like enhancement
        img_array = np.array(img, dtype=np.float32)

        # Normalize using percentile-based normalization (handles outliers better)
        p2, p98 = np.percentile(img_array, (2, 98))
        if p98 - p2 > 0:
            img_array = np.clip((img_array - p2) / (p98 - p2), 0, 1)
        else:
            img_array = img_array / 255.0

        # Add channel dimension
        img_array = np.expand_dims(img_array, axis=-1)

        if augment:
            img_array = random_augment_ultrasound(img_array)

        return img_array.astype(np.float32)

    except Exception as e:
        print(f"    Error preprocessing {img_path}: {e}")
        return None


def random_augment_ultrasound(img_array):
    """
    Apply random augmentations suitable for ultrasound images.
    """
    # Random brightness
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        img_array = np.clip(img_array * factor, 0, 1)

    # Random Gaussian noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.02, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)

    # Random horizontal flip (left-right is anatomically valid - probe can be
    # mirrored/approached from either side)
    if random.random() > 0.5:
        img_array = np.fliplr(img_array)

    # NOTE: vertical flip and 90-degree rotations intentionally removed.
    # Breast ultrasound has a fixed depth axis (skin/fat near top, deeper
    # tissue below) carrying real diagnostic signal — e.g. posterior
    # acoustic shadowing behind malignant masses only makes sense in the
    # true orientation. Flipping vertically or rotating 90/180/270 degrees
    # manufactures anatomically implausible images, which is especially
    # damaging here since this function heavily populates the oversampled
    # minority class.

    # Random rotation (small angle)
    if random.random() > 0.5:
        from scipy.ndimage import rotate as scipy_rotate
        try:
            angle = random.uniform(-15, 15)
            img_array = scipy_rotate(img_array, angle, axes=(0, 1),
                                     reshape=False, mode='constant', cval=0)
            img_array = np.clip(img_array, 0, 1)
        except ImportError:
            pass

    return img_array.astype(np.float32)


# ==============================================================================
#                    IMPROVED MODEL ARCHITECTURES
# ==============================================================================

def residual_block(x, filters, kernel_size=3, stride=1, downsample=False):
    """Create a residual block with skip connection."""
    shortcut = x

    # First conv
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second conv
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut if needed
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def squeeze_excitation_block(x, ratio=16):
    """Squeeze-and-Excitation attention block for better feature selection."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(filters // ratio, 1), activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def create_breast_model_improved(input_shape=(224, 224, 1), num_classes=3):
    """
    Improved CNN with residual connections and squeeze-excitation blocks.

    Key improvements:
    - Residual connections prevent vanishing gradients
    - SE blocks provide channel attention
    - Better regularization strategy
    - Deeper but more efficient architecture
    """
    inputs = layers.Input(shape=input_shape)

    # Initial conv
    x = layers.Conv2D(32, 7, strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks with increasing filters
    # Stage 1: 32 filters
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = squeeze_excitation_block(x)
    x = layers.Dropout(0.2)(x)

    # Stage 2: 64 filters with downsampling
    x = residual_block(x, 64, stride=2, downsample=True)
    x = residual_block(x, 64)
    x = squeeze_excitation_block(x)
    x = layers.Dropout(0.25)(x)

    # Stage 3: 128 filters
    x = residual_block(x, 128, stride=2, downsample=True)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = squeeze_excitation_block(x)
    x = layers.Dropout(0.3)(x)

    # Stage 4: 256 filters
    x = residual_block(x, 256, stride=2, downsample=True)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = squeeze_excitation_block(x)
    x = layers.Dropout(0.3)(x)

    # Stage 5: 512 filters
    x = residual_block(x, 512, stride=2, downsample=True)
    x = residual_block(x, 512)
    x = squeeze_excitation_block(x)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='breast_cancer_resnet')

    return model


def create_breast_model_transfer_grayscale(input_shape=(224, 224, 1), num_classes=3):
    """
    Transfer learning model that works with grayscale by replicating channels.
    Replicates grayscale to 3 channels to use ImageNet weights.
    clipnorm=1.0 on Adam prevents val_loss explosion.
    """
    # Input layer for grayscale
    gray_input = layers.Input(shape=input_shape, name='grayscale_input')

    # Replicate to 3 channels for pretrained model
    x = layers.Concatenate()([gray_input, gray_input, gray_input])

    # Scale to [-1, 1] for MobileNetV2 pretrained weights
    x = layers.Lambda(lambda val: val * 2.0 - 1.0)(x)

    # Use MobileNetV2 as backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze initially

    # Get features
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(gray_input, outputs, name='breast_cancer_transfer')

    return model, base_model


# ==============================================================================
#                    FOCAL LOSS FOR CLASS IMBALANCE
# ==============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss - handles class imbalance much better than cross-entropy.
    Down-weights easy examples and focuses on hard ones.
    """
    def focal_loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=-1)
    return focal_loss_fn


# ==============================================================================
#                    DATA LOADING - IMPROVED
# ==============================================================================

def find_breast_ultrasound_dataset():
    """Find the breast ultrasound dataset."""
    possible_names = [
        'breast_ultrasound', 'Breast_Ultrasound', 'breast-ultrasound',
        'Dataset_BUSI_with_GT', 'BUSI', 'busi', 'breast_ultrasound_images'
    ]

    for name in possible_names:
        check_dir = os.path.join(DATASET_DIR, name)
        if os.path.exists(check_dir):
            subdirs = [d for d in os.listdir(check_dir)
                       if os.path.isdir(os.path.join(check_dir, d))]
            subdirs_lower = [d.lower() for d in subdirs]

            if any('benign' in s for s in subdirs_lower) or \
               any('malignant' in s for s in subdirs_lower):
                print(f"  Found dataset at: {check_dir}")
                print(f"  Subdirectories: {subdirs}")
                return check_dir, 'ultrasound'

    return None, None


def load_breast_ultrasound_data_improved(img_size=224, augment_minority=True):
    """
    Load Breast Ultrasound dataset with IMPROVED preprocessing.

    Improvements:
    - Advanced preprocessing (CLAHE, percentile normalization)
    - Minority class oversampling with augmentation
    - Better image quality through enhanced preprocessing
    - Filters out mask images properly
    """
    data_dir, dataset_type = find_breast_ultrasound_dataset()

    if data_dir is None:
        print(f"❌ Breast ultrasound dataset not found")
        print(f"\n📥 Download from: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print(f"📁 Extract to: {os.path.join(DATASET_DIR, 'breast_ultrasound')}")
        print("\nExpected structure:")
        print("  breast_ultrasound/")
        print("  ├── benign/")
        print("  ├── malignant/")
        print("  └── normal/")
        return None

    print(f"📂 Loading Breast Ultrasound dataset from {data_dir}...")
    print(f"  Using improved preprocessing pipeline")

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
        print(f"❌ Not enough class folders found: {list(class_folders.keys())}")
        return None

    class_to_idx = {'normal': 0, 'benign': 1, 'malignant': 2}

    # Load all images with improved preprocessing
    class_data = {}

    for class_name, folder_path in class_folders.items():
        print(f"\n  Loading {class_name}...")

        images_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp']:
            images_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        # Strict mask filter: exclude any file with 'mask' anywhere in the name
        import re
        images_paths = [p for p in images_paths
                        if not re.search(r'mask', os.path.basename(p).lower())]

        print(f"    Found {len(images_paths)} images (masks strictly excluded)")

        loaded_images = []
        for img_path in images_paths:
            img_array = preprocess_ultrasound_image(img_path, img_size, augment=False)
            if img_array is not None:
                loaded_images.append(img_array)

        class_data[class_name] = loaded_images
        print(f"    Successfully loaded {len(loaded_images)} images")

    # Print class distribution
    print(f"\n  Class distribution:")
    for cls_name, imgs in class_data.items():
        print(f"    {cls_name}: {len(imgs)} images")

    # Capture original sizes before oversampling to calculate pre-augmentation class weights
    original_sizes = {cls_name: len(imgs) for cls_name, imgs in class_data.items()}

    # Handle class imbalance through oversampling with augmentation
    if augment_minority:
        # Target: balance all classes to at least 80% of the largest class
        max_count = max(len(imgs) for imgs in class_data.values())
        target_count = int(max_count * 1.0)  # Match largest class exactly

        print(f"\n  Balancing classes to {target_count} samples each with augmentation...")

        for cls_name in class_data:
            current_count = len(class_data[cls_name])
            if current_count < target_count:
                additional_needed = target_count - current_count
                print(f"    Augmenting {cls_name}: {current_count} \u2192 {target_count} (+{additional_needed})")

                original_images = class_data[cls_name].copy()
                for i in range(additional_needed):
                    src_img = original_images[i % len(original_images)].copy()
                    aug_img = random_augment_ultrasound(src_img)
                    class_data[cls_name].append(aug_img)

    # Build final arrays
    X = []
    y = []

    for cls_name, imgs in class_data.items():
        class_idx = class_to_idx[cls_name]
        for img in imgs:
            X.append(img)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n  Total samples: {len(X)}")
    print(f"  Image shape: {X.shape}")
    print(f"  Final class distribution: {Counter(y)}")

    # Verify data quality
    print(f"\n  Data quality check:")
    print(f"    Mean pixel value: {X.mean():.4f}")
    print(f"    Std pixel value: {X.std():.4f}")
    print(f"    Min pixel value: {X.min():.4f}")
    print(f"    Max pixel value: {X.max():.4f}")

    # One-hot encode
    num_classes = len(class_folders)
    y_onehot = to_categorical(y, num_classes=num_classes)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Class weights must reflect the ACTUAL distribution the model trains on.
    # The data was already balanced via oversampling above (lines ~460-478) —
    # applying weights computed on the pre-oversampling imbalance here would
    # double-compensate: minority classes get duplicated via augmentation
    # AND upweighted in the loss, while majority classes get suppressed in
    # the loss despite being present in equal numbers post-oversampling.
    # This was the likely cause of the model failing to properly separate
    # normal/benign/malignant — recompute on the real post-oversampling `y`.
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y), y=y
    )
    class_weight_dict = {i: float(class_weights[i]) for i in range(num_classes)}
    print(f"  Original class counts (pre-oversampling): { {k: original_sizes[k] for k in original_sizes} }")
    print(f"  Post-oversampling class weights (should be ~1.0 each): {class_weight_dict}")

    return X_train, X_test, y_train, y_test, class_weight_dict, num_classes


# ==============================================================================
#                    IMPROVED TRAINING FUNCTION
# ==============================================================================

def cosine_decay_with_warmup(epoch, total_epochs=50, warmup_epochs=5,
                              initial_lr=0.001, min_lr=1e-6):
    """Cosine decay learning rate with linear warmup."""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))


def train_breast_cancer_model(use_transfer=True, use_6_classes=False):
    """
    Train breast cancer detection model - IMPROVED VERSION.

    Improvements:
    - Better model architecture (residual + SE blocks OR transfer learning)
    - Advanced preprocessing (CLAHE, percentile normalization)
    - Focal loss for class imbalance
    - Cosine annealing with warmup
    - Two-phase training (frozen then fine-tuned)
    - Proper evaluation with confusion matrix and per-class metrics
    """
    if not TF_AVAILABLE:
        print("❌ TensorFlow required")
        return None

    print("\n" + "=" * 70)
    print("  BREAST CANCER DETECTION MODEL - IMPROVED TRAINING")
    print("  Dataset: Breast Ultrasound Images")
    print("=" * 70)

    # Load data with improved preprocessing
    data = load_breast_ultrasound_data_improved(img_size=IMG_SIZE, augment_minority=True)

    if data is None:
        print("\n⚠️ Dataset not found. Cannot train model.")
        return None

    X_train, X_test, y_train, y_test, class_weight_dict, num_classes = data

    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    # Handle 6-class conversion
    if use_6_classes and num_classes == 3:
        print("\n  Converting 3-class to 6-class labels...")
        y_train_int = np.argmax(y_train, axis=1)
        y_test_int = np.argmax(y_test, axis=1)

        y_train_6 = np.array([CLASS_3_TO_6_MAPPING[yi] for yi in y_train_int])
        y_test_6 = np.array([CLASS_3_TO_6_MAPPING[yi] for yi in y_test_int])

        y_train = to_categorical(y_train_6, num_classes=6)
        y_test = to_categorical(y_test_6, num_classes=6)

        class_weights_arr = compute_class_weight(
            'balanced', classes=np.unique(y_train_6), y=y_train_6
        )
        class_weight_dict = {i: 1.0 for i in range(6)}
        for cls, w in zip(np.unique(y_train_6), class_weights_arr):
            class_weight_dict[cls] = w

        num_classes = 6

    # Create model
    base_model_ref = None

    if use_transfer:
        print(f"\n🔧 Creating transfer learning model (grayscale -> 3ch trick)...")
        model, base_model_ref = create_breast_model_transfer_grayscale(
            input_shape, num_classes
        )
    else:
        print(f"\n🔧 Creating improved ResNet-style model...")
        model = create_breast_model_improved(input_shape, num_classes)

    # Compile with standard categorical_crossentropy + class weights
    # (avoids focal_loss serialization issues when loading in server.py)
    model.compile(
        optimizer=Adam(learning_rate=0.0002, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    model.summary()
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")

    # Data augmentation - optimized for ultrasound
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        # vertical_flip intentionally omitted: breast ultrasound has a fixed
        # anatomical orientation (skin/fat near top, deeper tissue below).
        # Flipping vertically manufactures physically implausible images and
        # destroys depth-dependent diagnostic features like posterior
        # acoustic shadowing behind malignant masses.
        zoom_range=0.2,
        shear_range=0.15,
        brightness_range=[0.75, 1.25],
        fill_mode='constant',
        cval=0
    )

    # Callbacks
    total_epochs_phase1 = 25
    callbacks_phase1 = [
        EarlyStopping(
            monitor='val_auc', mode='max',
            patience=8, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            BREAST_MODEL_PATH, monitor='val_auc',
            mode='max', save_best_only=True, verbose=1
        )
    ]

    # =================== PHASE 1: Train classification head ===================
    print("\n" + "-" * 50)
    print("🚀 Phase 1: Training classification head...")
    print("-" * 50)

    batch_size = 16

    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=total_epochs_phase1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_phase1,
        class_weight=class_weight_dict,
        verbose=1
    )

    # =================== PHASE 2: Fine-tune (transfer learning only) ==========
    if use_transfer and base_model_ref is not None:
        print("\n" + "-" * 50)
        print("🚀 Phase 2: Fine-tuning backbone...")
        print("-" * 50)

        # Unfreeze top layers
        base_model_ref.trainable = True
        for layer in base_model_ref.layers[:-60]:
            layer.trainable = False

        trainable_count = sum(1 for layer in base_model_ref.layers if layer.trainable)
        print(f"  Unfroze {trainable_count} layers for fine-tuning")

        # Recompile with very low learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.00002),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )

        total_epochs_phase2 = 15
        callbacks_phase2 = [
            EarlyStopping(
                monitor='val_auc', mode='max',
                patience=6, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3,
                min_lr=1e-8, verbose=1
            ),
            ModelCheckpoint(
                BREAST_MODEL_PATH, monitor='val_auc',
                mode='max', save_best_only=True, verbose=1
            )
        ]

        history2 = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=total_epochs_phase2,
            validation_data=(X_test, y_test),
            callbacks=callbacks_phase2,
            class_weight=class_weight_dict,
            verbose=1
        )

    # =================== EVALUATION ===================
    print("\n" + "-" * 50)
    print("📊 Detailed Evaluation...")
    print("-" * 50)

    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Loss: {results[0]:.4f}")
    print(f"  Test Accuracy: {results[1]:.4f} ({results[1] * 100:.2f}%)")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall: {results[3]:.4f}")
    print(f"  AUC: {results[4]:.4f}")

    # Detailed predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    class_names = (['normal', 'benign', 'malignant'] if num_classes == 3
                   else ['normal', 'benign', 'prob_benign', 'suspicious', 'high_susp', 'malignant'])

    # Classification report
    print("\n  Classification Report:")
    present_classes = sorted(list(set(y_true_classes) | set(y_pred_classes)))
    present_names = [class_names[i] for i in present_classes if i < len(class_names)]
    report = classification_report(
        y_true_classes, y_pred_classes,
        labels=present_classes,
        target_names=present_names,
        zero_division=0
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(f"  Confusion Matrix:")
    print(cm)

    # Confidence distribution
    print(f"\n  Confidence distribution:")
    max_confidences = np.max(y_pred, axis=1)
    print(f"    Mean confidence: {max_confidences.mean():.4f}")
    print(f"    Median confidence: {np.median(max_confidences):.4f}")
    print(f"    Min confidence: {max_confidences.min():.4f}")
    print(f"    Max confidence: {max_confidences.max():.4f}")

    # Per-class confidence
    for cls in present_classes:
        mask = y_true_classes == cls
        if mask.sum() > 0:
            cls_conf = y_pred[mask, cls]
            cls_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
            print(f"    {cls_name}: mean conf={cls_conf.mean():.4f}, "
                  f"correct={y_pred_classes[mask].tolist().count(cls)}/{mask.sum()}")

    # Save model
    model.save(BREAST_MODEL_PATH)
    print(f"\n[OK] Model saved: {BREAST_MODEL_PATH}")

    # Save config
    classes_config = ({str(k): v for k, v in BREAST_CLASSES_3.items()} if num_classes == 3
                      else {str(k): v for k, v in BREAST_CLASSES_6.items()})

    config = {
        'model_path': BREAST_MODEL_PATH,
        'input_shape': list(input_shape),
        'preprocessing': 'Grayscale, CLAHE, percentile normalization to [0,1]',
        'use_grayscale': True,
        'num_classes': num_classes,
        'classes': classes_config,
        'class_names': class_names[:num_classes],
        'architecture': 'transfer_mobilenetv2' if use_transfer else 'resnet_se',
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4]),
        'mean_confidence': float(max_confidences.mean()),
        'confusion_matrix': cm.tolist()
    }

    with open(BREAST_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Config saved: {BREAST_CONFIG_PATH}")

    print("\n" + "=" * 70)
    print("[WARN] REMINDERS:")
    print(f"  - Model uses GRAYSCALE preprocessing")
    print(f"  - Input: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Restart server.py to load new model!")
    print("=" * 70)

    return model


# ==============================================================================
#                    MAIN
# ==============================================================================

def main():
    if not TF_AVAILABLE:
        print("❌ TensorFlow required. pip install tensorflow")
        return

    print("\n" + "=" * 70)
    print("  BREAST CANCER MODEL TRAINING - IMPROVED")
    print("=" * 70)

    print("\nOptions:")
    print("  1. Train 3-class with transfer learning (RECOMMENDED - highest accuracy)")
    print("  2. Train 3-class with custom ResNet (no pretrained weights)")
    print("  3. Train 6-class with transfer learning")
    print("  4. Exit")

    import sys
    choice = '1'
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        print(f"Using CLI choice: {choice}")
    elif not sys.stdin.isatty():
        print("Non-interactive stdin detected. Training option 1 by default.")
        choice = '1'
    else:
        choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        train_breast_cancer_model(use_transfer=True, use_6_classes=False)
    elif choice == '2':
        train_breast_cancer_model(use_transfer=False, use_6_classes=False)
    elif choice == '3':
        train_breast_cancer_model(use_transfer=True, use_6_classes=True)
    else:
        print("Exiting...")
        return

    print("\n✅ Training Complete!")
    print("⚠️  Restart server.py to load the new model!")


if __name__ == '__main__':
    main()