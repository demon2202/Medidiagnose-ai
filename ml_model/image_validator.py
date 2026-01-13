import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} available")
except ImportError:
    print("✗ TensorFlow not available")

from PIL import Image
import json
from collections import Counter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATOR_MODEL_PATH = os.path.join(SCRIPT_DIR, 'image_validator_model.h5')
VALIDATOR_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'image_validator_config.json')

# Image size
IMG_SIZE = 224

# Image type classes
IMAGE_TYPES = {
    0: {'code': 'skin_lesion', 'name': 'Skin Lesion/Dermoscopy', 'valid_for': ['skin']},
    1: {'code': 'xray_chest', 'name': 'Chest X-Ray', 'valid_for': ['xray', 'pneumonia']},
    2: {'code': 'mammogram', 'name': 'Mammogram/Breast Ultrasound', 'valid_for': ['breast']},
    3: {'code': 'ecg', 'name': 'ECG/Heart Scan', 'valid_for': ['heart']},
    4: {'code': 'other', 'name': 'Non-Medical/Unrecognized', 'valid_for': []}
}

# Characteristics for rule-based detection
IMAGE_CHARACTERISTICS = {
    'skin_lesion': {
        'color_range': 'high',  # Colorful (brown, black, pink, red)
        'typical_aspect': 'square-ish',
        'texture': 'varied',
        'background': 'skin-toned'
    },
    'xray_chest': {
        'color_range': 'grayscale',
        'typical_aspect': 'portrait',
        'texture': 'smooth gradients',
        'features': 'ribs, lungs, heart silhouette'
    },
    'mammogram': {
        'color_range': 'grayscale',
        'typical_aspect': 'varies',
        'texture': 'dense tissue patterns',
        'background': 'black'
    },
    'ecg': {
        'color_range': 'low',  # Usually white/light with dark lines
        'typical_aspect': 'landscape',
        'texture': 'grid with wave patterns',
        'features': 'regular peaks, grid lines'
    }
}


class ImageValidator:
    """
    Validates if an uploaded image matches the expected medical image type.
    Uses both rule-based heuristics and optional ML model.
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.use_ml = False
        
        # Try to load ML model
        if model_path and os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                self.use_ml = True
                print("✓ Image validator ML model loaded")
            except Exception as e:
                print(f"⚠ Could not load validator model: {e}")
        
        if not self.use_ml:
            print("ℹ Using rule-based image validation")
    
    def analyze_image_statistics(self, img_array):
        """
        Analyze image statistics to help determine image type.
        
        Args:
            img_array: numpy array of image (H, W, C) normalized to [0, 1]
        
        Returns:
            dict with image statistics
        """
        stats = {}
        
        # Color analysis
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # RGB image
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Check if grayscale (R≈G≈B)
            rgb_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
            stats['is_grayscale'] = rgb_diff < 0.05
            stats['rgb_variance'] = float(rgb_diff)
            
            # Color statistics
            stats['mean_r'] = float(np.mean(r))
            stats['mean_g'] = float(np.mean(g))
            stats['mean_b'] = float(np.mean(b))
            stats['overall_brightness'] = float(np.mean(img_array))
            
            # Saturation (color intensity)
            max_rgb = np.maximum(np.maximum(r, g), b)
            min_rgb = np.minimum(np.minimum(r, g), b)
            saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-7), 0)
            stats['mean_saturation'] = float(np.mean(saturation))
            
        else:
            stats['is_grayscale'] = True
            stats['overall_brightness'] = float(np.mean(img_array))
            stats['mean_saturation'] = 0.0
        
        # Edge detection (simple gradient)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Sobel-like edge detection
        gx = np.abs(gray[1:, :] - gray[:-1, :])
        gy = np.abs(gray[:, 1:] - gray[:, :-1])
        stats['edge_intensity'] = float(np.mean(gx) + np.mean(gy))
        
        # Histogram analysis
        hist, _ = np.histogram(gray.flatten(), bins=50, range=(0, 1))
        hist = hist / (hist.sum() + 1e-7)
        stats['histogram_entropy'] = float(-np.sum(hist * np.log(hist + 1e-7)))
        
        # Check for grid patterns (ECG characteristic)
        # Look for regular vertical lines
        col_variance = np.var(np.mean(gray, axis=0))
        row_variance = np.var(np.mean(gray, axis=1))
        stats['has_grid_pattern'] = float(col_variance + row_variance)
        
        # Check for large dark regions (X-ray/mammogram characteristic)
        dark_pixels = np.mean(gray < 0.2)
        stats['dark_region_ratio'] = float(dark_pixels)
        
        # Check for skin tones (skin lesion characteristic)
        if not stats['is_grayscale']:
            # Skin tone detection (simplified)
            skin_mask = (
                (r > 0.3) & (r < 0.9) &
                (g > 0.2) & (g < 0.8) &
                (b > 0.1) & (b < 0.7) &
                (r > g) & (g > b)
            )
            stats['skin_tone_ratio'] = float(np.mean(skin_mask))
        else:
            stats['skin_tone_ratio'] = 0.0
        
        return stats
    
    def predict_image_type_rules(self, img_array):
        """
        Rule-based image type prediction.
        
        Args:
            img_array: numpy array (H, W, C) normalized to [0, 1]
        
        Returns:
            tuple: (predicted_type_idx, confidence, all_scores)
        """
        stats = self.analyze_image_statistics(img_array)
        
        scores = {
            'skin_lesion': 0.0,
            'xray_chest': 0.0,
            'mammogram': 0.0,
            'ecg': 0.0,
            'other': 0.2  # Base score for "other"
        }
        
        # === SKIN LESION DETECTION ===
        if not stats['is_grayscale']:
            scores['skin_lesion'] += 0.3
        if stats['mean_saturation'] > 0.1:
            scores['skin_lesion'] += 0.2
        if stats['skin_tone_ratio'] > 0.3:
            scores['skin_lesion'] += 0.3
        if 0.3 < stats['overall_brightness'] < 0.7:
            scores['skin_lesion'] += 0.1
        if stats['rgb_variance'] > 0.02:
            scores['skin_lesion'] += 0.1
        
        # === X-RAY DETECTION ===
        if stats['is_grayscale'] or stats['rgb_variance'] < 0.02:
            scores['xray_chest'] += 0.25
        if stats['dark_region_ratio'] > 0.2 and stats['dark_region_ratio'] < 0.6:
            scores['xray_chest'] += 0.25
        if 0.3 < stats['overall_brightness'] < 0.6:
            scores['xray_chest'] += 0.2
        if stats['histogram_entropy'] > 2.5:
            scores['xray_chest'] += 0.15
        if stats['edge_intensity'] > 0.05 and stats['edge_intensity'] < 0.15:
            scores['xray_chest'] += 0.15
        
        # === MAMMOGRAM DETECTION ===
        if stats['is_grayscale'] or stats['rgb_variance'] < 0.02:
            scores['mammogram'] += 0.2
        if stats['dark_region_ratio'] > 0.4:
            scores['mammogram'] += 0.3
        if stats['overall_brightness'] < 0.4:
            scores['mammogram'] += 0.2
        if stats['histogram_entropy'] < 3.0:
            scores['mammogram'] += 0.15
        if stats['edge_intensity'] < 0.1:
            scores['mammogram'] += 0.15
        
        # === ECG DETECTION ===
        if stats['is_grayscale'] or stats['mean_saturation'] < 0.1:
            scores['ecg'] += 0.15
        if stats['overall_brightness'] > 0.6:
            scores['ecg'] += 0.2  # ECGs usually have white/light background
        if stats['has_grid_pattern'] > 0.01:
            scores['ecg'] += 0.3
        if stats['edge_intensity'] > 0.08:
            scores['ecg'] += 0.2
        if stats['dark_region_ratio'] < 0.2:
            scores['ecg'] += 0.15
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        # Get prediction
        type_mapping = {
            'skin_lesion': 0,
            'xray_chest': 1,
            'mammogram': 2,
            'ecg': 3,
            'other': 4
        }
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # If confidence is too low, mark as "other"
        if confidence < 0.35:
            best_type = 'other'
            confidence = scores['other']
        
        return type_mapping[best_type], confidence, scores, stats
    
    def predict_image_type_ml(self, img_array):
        """
        ML-based image type prediction.
        
        Args:
            img_array: numpy array (H, W, C) normalized to [0, 1]
        
        Returns:
            tuple: (predicted_type_idx, confidence, all_probabilities)
        """
        if self.model is None:
            return self.predict_image_type_rules(img_array)
        
        # Prepare input
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        
        return predicted_idx, confidence, predictions
    
    def validate_image(self, img_array, expected_type):
        """
        Validate if image matches expected type.
        
        Args:
            img_array: numpy array (H, W, C) normalized to [0, 1]
            expected_type: str - 'skin', 'xray', 'breast', 'heart'
        
        Returns:
            dict with validation results
        """
        # Get image type prediction
        if self.use_ml:
            pred_idx, confidence, all_probs = self.predict_image_type_ml(img_array)
            stats = self.analyze_image_statistics(img_array)
        else:
            pred_idx, confidence, all_scores, stats = self.predict_image_type_rules(img_array)
            all_probs = all_scores
        
        predicted_type_info = IMAGE_TYPES.get(pred_idx, IMAGE_TYPES[4])
        
        # Check if prediction matches expected type
        type_mapping = {
            'skin': ['skin_lesion'],
            'xray': ['xray_chest'],
            'pneumonia': ['xray_chest'],
            'breast': ['mammogram'],
            'heart': ['ecg'],
            'ecg': ['ecg']
        }
        
        expected_codes = type_mapping.get(expected_type.lower(), [])
        is_valid = predicted_type_info['code'] in expected_codes
        
        # Also check confidence threshold
        confidence_threshold = 0.4
        is_confident = confidence >= confidence_threshold
        
        # Final validation
        if not is_valid or not is_confident:
            # Image doesn't match expected type
            if pred_idx == 4:  # "other"
                message = f"This image does not appear to be a valid medical image. Please upload a proper {expected_type} image."
            else:
                message = f"This image appears to be a {predicted_type_info['name']}, not a {expected_type} image. Please upload the correct image type."
            
            return {
                'is_valid': False,
                'predicted_type': predicted_type_info['name'],
                'predicted_code': predicted_type_info['code'],
                'expected_type': expected_type,
                'confidence': confidence,
                'message': message,
                'suggestion': f"Please upload a valid {expected_type} image for accurate analysis.",
                'image_stats': stats
            }
        
        return {
            'is_valid': True,
            'predicted_type': predicted_type_info['name'],
            'predicted_code': predicted_type_info['code'],
            'expected_type': expected_type,
            'confidence': confidence,
            'message': 'Image type validated successfully.',
            'image_stats': stats
        }


def create_validator_model(input_shape=(224, 224, 3), num_classes=5):
    """Create a lightweight CNN for image type classification"""
    
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
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def generate_synthetic_validation_data(n_samples_per_class=200, img_size=224):
    """
    Generate synthetic training data for image type classifier.
    In production, you'd use real samples from each category.
    """
    print("⚠ Generating synthetic validation training data...")
    print("  For best results, collect real samples of each image type.")
    
    np.random.seed(42)
    X = []
    y = []
    
    for class_idx in range(5):
        for _ in range(n_samples_per_class):
            if class_idx == 0:  # Skin lesion - colorful, skin-toned
                img = np.random.rand(img_size, img_size, 3) * 0.3
                # Add skin tone base
                img[:,:,0] += 0.4  # More red
                img[:,:,1] += 0.25  # Some green
                img[:,:,2] += 0.15  # Less blue
                # Add lesion spot
                cx, cy = img_size//2 + np.random.randint(-30, 30), img_size//2 + np.random.randint(-30, 30)
                for i in range(img_size):
                    for j in range(img_size):
                        dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                        if dist < 40:
                            img[i, j] = [0.3 + np.random.rand()*0.2, 0.15, 0.1]
                
            elif class_idx == 1:  # X-ray - grayscale, dark regions
                gray = np.random.rand(img_size, img_size) * 0.3 + 0.3
                # Add lung-like dark regions
                cx1, cx2 = img_size//3, 2*img_size//3
                cy = img_size//2
                for i in range(img_size):
                    for j in range(img_size):
                        dist1 = np.sqrt((i-cy)**2 + (j-cx1)**2)
                        dist2 = np.sqrt((i-cy)**2 + (j-cx2)**2)
                        if dist1 < 50 or dist2 < 50:
                            gray[i, j] *= 0.5
                img = np.stack([gray]*3, axis=-1)
                
            elif class_idx == 2:  # Mammogram - grayscale, dark background
                gray = np.random.rand(img_size, img_size) * 0.2
                # Add bright tissue region
                cx, cy = img_size//2 + np.random.randint(-20, 20), img_size//2
                for i in range(img_size):
                    for j in range(img_size):
                        dist = np.sqrt((i-cy)**2 + (j-cx)**2)
                        if dist < 60:
                            gray[i, j] = 0.5 + np.random.rand() * 0.3
                img = np.stack([gray]*3, axis=-1)
                
            elif class_idx == 3:  # ECG - light background, wave patterns
                img = np.ones((img_size, img_size, 3)) * 0.9
                # Add grid
                for i in range(0, img_size, 20):
                    img[i:i+1, :] = [0.8, 0.85, 0.8]
                    img[:, i:i+1] = [0.8, 0.85, 0.8]
                # Add ECG wave
                wave_y = img_size // 2
                for x in range(img_size):
                    wave_offset = int(30 * np.sin(x * 0.1) * np.exp(-((x % 50) - 25)**2 / 100))
                    y_pos = wave_y + wave_offset
                    if 0 <= y_pos < img_size:
                        img[max(0, y_pos-1):min(img_size, y_pos+2), x] = [0.1, 0.1, 0.1]
                
            else:  # Other - random patterns
                img = np.random.rand(img_size, img_size, 3)
            
            # Ensure valid range
            img = np.clip(img, 0, 1).astype(np.float32)
            X.append(img)
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    print(f"  ✓ Generated {len(X)} synthetic images")
    return X, y


def train_validator_model():
    """Train the image type validator model"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow required")
        return None
    
    print("\n" + "=" * 60)
    print("  IMAGE TYPE VALIDATOR - TRAINING")
    print("=" * 60)
    
    # Generate synthetic data (replace with real data collection)
    X, y = generate_synthetic_validation_data(200, IMG_SIZE)
    
    # One-hot encode
    y_onehot = to_categorical(y, num_classes=5)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Create model
    model = create_validator_model()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(VALIDATOR_MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    ]
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Train
    print("\n  Training validator model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=15,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  ✓ Validation Accuracy: {results[1]*100:.1f}%")
    
    # Save
    model.save(VALIDATOR_MODEL_PATH)
    print(f"  ✓ Model saved: {VALIDATOR_MODEL_PATH}")
    
    # Save config
    config = {
        'model_path': VALIDATOR_MODEL_PATH,
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'num_classes': 5,
        'classes': {str(k): v for k, v in IMAGE_TYPES.items()},
        'accuracy': float(results[1])
    }
    
    with open(VALIDATOR_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    return model


# Singleton validator instance
_validator_instance = None

def get_validator():
    """Get or create the image validator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ImageValidator(VALIDATOR_MODEL_PATH)
    return _validator_instance


def validate_medical_image(img_array, expected_type):
    """
    Convenience function to validate a medical image.
    
    Args:
        img_array: numpy array (H, W, C) normalized to [0, 1]
        expected_type: str - 'skin', 'xray', 'breast', 'heart'
    
    Returns:
        dict with validation results
    """
    validator = get_validator()
    return validator.validate_image(img_array, expected_type)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Image Validator Training & Testing")
    print("=" * 60)
    
    # Train the model
    model = train_validator_model()
    
    # Test the validator
    print("\n" + "-" * 40)
    print("  Testing Validator...")
    print("-" * 40)
    
    validator = ImageValidator(VALIDATOR_MODEL_PATH)
    
    # Create test images
    test_cases = [
        ('skin', np.random.rand(224, 224, 3) * 0.3 + np.array([0.4, 0.25, 0.15])),
        ('xray', np.stack([np.random.rand(224, 224) * 0.3 + 0.3]*3, axis=-1)),
        ('breast', np.stack([np.random.rand(224, 224) * 0.2]*3, axis=-1)),
    ]
    
    for expected, img in test_cases:
        img = np.clip(img, 0, 1).astype(np.float32)
        result = validator.validate_image(img, expected)
        print(f"\n  Expected: {expected}")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Predicted: {result['predicted_type']}")
        print(f"  Confidence: {result['confidence']:.2f}")
    
    print("\n" + "=" * 60)
    print("  ✓ Image Validator Ready!")
    print("=" * 60)